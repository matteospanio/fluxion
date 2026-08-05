"""Python covers the op registry, exactly — and the Wave/chain surface behaves as documented.

`fluxion.filter` and `fluxion.effect` are generated from `fluxion.ops_table()`, so in principle
they cannot drift. These tests are what makes that true in practice: they fail if the generated
files are stale, if a class was hand-edited, or if the generator's idea of a signature and the
registry's stop agreeing.
"""

import inspect

import numpy as np
import pytest

import fluxion as fx

FS = 48_000


def _cls(op: dict) -> type:
    """The generated class for a registry row."""
    return getattr(getattr(fx, op["group"]), op["class"])


def test_every_registry_op_is_a_class_in_its_group():
    """Pre: ops_table() is the Rust registry. Post: every row has a class, with matching params."""
    for op in fx.ops_table():
        cls = _cls(op)  # AttributeError here means the generated module is stale
        assert issubclass(cls, fx.Chain), f"{op['class']} does not extend Chain"
        params = list(inspect.signature(cls.__new__).parameters)
        assert params[0] == "cls"
        if op["variadic"]:
            # A variadic op's parameters are the vector, so it takes *args named after the spec.
            assert params[1:] == [f"{op['params'][0]['name']}s"]
        else:
            assert params[1:] == [p["name"] for p in op["params"]], (
                f"{op['class']} parameters drifted from the registry"
            )


def test_no_class_exists_without_a_registry_row():
    """The other direction: nothing hand-added, nothing left behind after a rename."""
    for group in ("filter", "effect"):
        expected = {op["class"] for op in fx.ops_table() if op["group"] == group}
        assert set(getattr(fx, group).__all__) == expected


def test_class_defaults_are_the_registry_defaults():
    """A bare class call means the same thing as the bare op name in chain text."""
    for op in fx.ops_table():
        cls = _cls(op)
        if not op["variadic"]:
            signature = inspect.signature(cls.__new__)
            for param in op["params"]:
                assert signature.parameters[param["name"]].default == pytest.approx(
                    param["default"]
                ), f"{op['class']}.{param['name']} default drifted"
        assert str(cls()) == str(fx.chain(op["name"])), f"{op['class']}() != chain('{op['name']}')"


def test_error_names_the_class_the_parameter_and_the_range():
    with pytest.raises(ValueError, match=r"fluxion\.filter\.Lowpass.*cutoff.*out of range"):
        fx.filter.Lowpass(-1.0)
    with pytest.raises(ValueError, match=r"fluxion\.effect\.Compand.*ratio.*\[1, 100\]"):
        fx.effect.Compand(0.01, 0.1, -20.0, 0.5, 6.0, 0.0)
    # An unknown name suggests the real one rather than just failing.
    with pytest.raises(ValueError, match="did you mean 'highpass'"):
        fx.Chain("hipass", 80.0)


def test_chain_text_is_the_same_string_every_interface_uses():
    built = fx.filter.Highpass(80, order=4) | fx.effect.Gain(0.5)
    assert str(built) == "highpass(80, 4) | gain(0.5)"
    assert str(fx.chain(str(built))) == str(built)
    # Named parameters, the shorthand and the dB suffix all land on the same graph.
    assert str(fx.chain("highpass(cutoff=80, order=4)")) == "highpass(80, 4)"
    assert str(fx.chain("gain=-3dB")) == str(fx.effect.Gain(fx.db(-3)))


def test_a_syntax_error_points_at_the_problem():
    with pytest.raises(ValueError, match=r"\^+ did you mean 'highpass'"):
        fx.chain("hipass(80) | gain(2)")


def test_a_chain_is_callable_on_a_bare_array():
    x = np.zeros(480, dtype=np.float32)
    x[0] = 1.0
    chain = fx.filter.Highpass(80, order=4) | fx.effect.Gain(0.5)
    y = chain(x, FS)
    assert y.shape == x.shape and y.dtype == np.float32
    # Calling and .process are the same thing.
    assert np.array_equal(y, chain.process(x, FS))


# --- Wave -----------------------------------------------------------------------------------


def test_wave_pipe_defers_into_a_single_pass():
    """`w | a | b` accumulates one two-leaf chain rather than running twice."""
    wave = fx.Wave(np.zeros((1, 480), dtype=np.float32), FS)
    piped = wave | fx.filter.Highpass(80) | fx.effect.Gain(0.7)
    assert str(piped._plan) == "highpass(80, 2) | gain(0.7)"
    _ = piped.ys  # materialize
    assert piped._plan is None
    # The source Wave is untouched: piping returns a new one.
    assert wave._plan is None


def test_wave_result_matches_calling_the_chain_directly():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 1024)).astype(np.float32)
    chain = fx.filter.Highpass(80, order=4) | fx.effect.Gain(0.5)
    assert np.allclose((fx.Wave(x, FS) | chain).ys, chain(x, FS))


def test_wave_promotes_mono_and_rejects_higher_rank():
    assert fx.Wave(np.zeros(64, dtype=np.float32), FS).ys.shape == (1, 64)
    with pytest.raises(ValueError, match="channels, frames"):
        fx.Wave(np.zeros((2, 2, 64), dtype=np.float32), FS)


def test_wave_rejects_something_that_is_not_an_effect():
    with pytest.raises(TypeError, match="expected a fluxion effect"):
        _ = fx.Wave(np.zeros(64, dtype=np.float32), FS) | "highpass(80)"


def test_wave_round_trips_through_a_file(tmp_path):
    rng = np.random.default_rng(1)
    x = (rng.standard_normal((2, 500)) * 0.1).astype(np.float32)
    path = tmp_path / "round.wav"
    fx.Wave(x, FS).save(path)

    back = fx.Wave.from_file(path)
    assert back.fs == FS
    assert back.channels() == 2
    assert len(back) == 500
    assert back.duration() == pytest.approx(500 / FS)
    assert np.allclose(back.ys, x, atol=1e-6)


def test_wave_ensure_fs_pins_the_project_rate(tmp_path):
    """ROADMAP R2: any rate in, the project rate out, at the length the two rates imply."""
    for source in (8_000, 22_050, 44_100, 96_000):
        w = fx.Wave(np.zeros((2, source // 10), dtype=np.float32), source)  # 100 ms
        out = w.ensure_fs(FS)
        assert out.fs == FS
        assert out.channels() == 2
        assert len(out) == round(len(w) * FS / source)
        assert out.metadata["source_fs"] == source


def test_wave_ensure_fs_at_the_same_rate_changes_nothing():
    w = fx.Wave(np.linspace(-1, 1, 64, dtype=np.float32), FS)
    assert w.ensure_fs(FS) is w
    with pytest.raises(ValueError, match="must be positive"):
        w.ensure_fs(0)


def test_wave_from_file_can_convert_on_the_way_in(tmp_path):
    """A host sets its rate once: files at it are untouched, the rest arrive converted."""
    path = tmp_path / "at44k.wav"
    tone = np.sin(2 * np.pi * 1000 * np.arange(44_100) / 44_100).astype(np.float32)
    fx.Wave(tone, 44_100).save(path)

    w = fx.Wave.from_file(path, fs=FS)
    assert w.fs == FS
    assert len(w) == FS
    # Still a 1 kHz tone, and still where the new rate says it is (the ends taper).
    want = np.sin(2 * np.pi * 1000 * np.arange(FS) / FS).astype(np.float32)
    assert np.allclose(w.ys[0][2000:-2000], want[2000:-2000], atol=1e-3)


def test_wave_channels_split_and_merge(tmp_path):
    left = fx.Wave(np.full((1, 8), 1.0, dtype=np.float32), FS)
    right = fx.Wave(np.full((1, 8), 2.0, dtype=np.float32), FS)

    stereo = fx.Wave.merge([left, right], split_channels=True)
    assert stereo.channels() == 2
    assert np.array_equal(stereo.get_channel(1).ys, right.ys)

    summed = fx.Wave.merge([left, right])
    assert summed.channels() == 1
    assert np.allclose(summed.ys, 3.0)

    with pytest.raises(ValueError, match="different sample rates"):
        fx.Wave.merge([left, fx.Wave(np.zeros((1, 8), dtype=np.float32), 44_100)])


def test_db_matches_the_chain_text_suffix():
    assert fx.db(0.0) == pytest.approx(1.0)
    assert fx.db(-6.0) == pytest.approx(0.5011872)
    assert str(fx.effect.Gain(fx.db(-3))) == str(fx.chain("gain=-3dB"))


def test_chain_takes_side_inputs():
    """ROADMAP S1/S3: a keyed gate follows the key, not the programme."""
    loud = (np.sin(2 * np.pi * 1000 * np.arange(FS) / FS) * 0.5).astype(np.float32)
    silent = np.zeros(FS, dtype=np.float32)

    keyed = fx.chain("gate(-40, 60, 0.001, 0, 0.005) < side(0)")
    shut = keyed(loud, FS, sides=[silent])
    assert np.abs(shut[FS // 2 :]).max() < 0.01

    # The same gate written without a key listens to itself and stays open.
    plain = fx.chain("gate(-40, 60, 0.001, 0, 0.005)")(loud, FS)
    assert np.abs(plain[FS // 4 :]).max() > 0.4


def test_taps_report_without_touching_the_audio():
    """ROADMAP A1/A2/A3: analysers in the chain, and the same samples out."""
    x = (np.sin(2 * np.pi * 375 * np.arange(FS) / FS) * 0.4).astype(np.float32)

    plain = fx.chain("gain(0.5)")(x, FS)
    audio, readings = fx.chain("meter | gain(0.5) | spectrum(2048, 0.5)").taps(x, FS)

    assert np.array_equal(plain, audio)
    assert [r["kind"] for r in readings] == ["meter", "spectrum"]
    # -6.02 dBFS is 0.5 linear, before the gain halves it.
    assert readings[0]["peak_db"] == pytest.approx(-7.96, abs=0.05)
    # 375 Hz is bin 16 of a 2048-point FFT at 48 kHz; the gain halved the 0.4 partial.
    spectrum = readings[1]
    bin_index = round(375 / spectrum["bin_hz"])
    assert spectrum["magnitude"][bin_index] == pytest.approx(0.2, abs=0.005)


def test_automation_and_region_rendering():
    """ROADMAP D2/D4: curves drive parameters, and a window is exactly that window."""
    x = np.ones(FS, dtype=np.float32)
    fade = fx.chain("fade: gain(1)")

    # A 60 dB fade drawn in decibels is -30 dB half way through, not -6.
    a = fx.Automation().db_ramp("fade", "gain", 1.0, 0.001, 1.0)
    assert len(a) == 1
    out = fade.automate(x, FS, a)
    assert 20 * np.log10(out[FS // 2]) == pytest.approx(-30.0, abs=0.05)

    # A stateful chain rendered in pieces is bit-identical to the whole render.
    stateful = fx.chain("highpass(80, 4) | echo(0.05, 0.4, 0.3)")
    noise = np.sin(np.arange(20_000, dtype=np.float32) * 0.05) * 0.5
    whole = stateful(noise, FS)
    pieces = np.empty_like(whole)
    for start, end in [(10_000, 20_000), (0, 3), (3, 9_999), (9_999, 10_000)]:
        pieces[start:end] = stateful.render_region(noise, FS, start, end)
    assert np.array_equal(whole, pieces)

    # An op that needs the whole signal says so rather than returning a plausible window.
    with pytest.raises(ValueError, match="normalize"):
        fx.chain("normalize(1)").render_region(noise, FS, 0, 100)

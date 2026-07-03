"""Python tests for the fluxion bindings (eager API + differentiable primitives)."""

import numpy as np
import pytest

import fluxion

FS = 48_000


def test_eager_chain_shape_and_dtype():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(4000).astype(np.float32)
    chain = fluxion.lowpass(1000.0, 4) | fluxion.gain(0.5)
    y = chain.process(x, FS)
    assert y.shape == x.shape
    assert y.dtype == np.float32


def test_lowpass_attenuates_high_frequency():
    t = np.arange(8000, dtype=np.float32) / FS
    hi = np.sin(2 * np.pi * 12_000 * t).astype(np.float32)
    y = fluxion.lowpass(1000.0, 6).process(hi, FS)
    rms = lambda v: float(np.sqrt(np.mean(v**2)))
    # A 12 kHz tone through a 1 kHz low-pass should lose most of its energy.
    assert rms(y) < 0.1 * rms(hi)


def test_gain_scales_exactly():
    x = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    y = fluxion.gain(0.5).process(x, FS)
    np.testing.assert_allclose(y, [0.5, -1.0, 1.5], atol=1e-6)


def test_parallel_sums():
    x = np.array([1.0, 10.0], dtype=np.float32)
    y = (fluxion.gain(2.0) + fluxion.gain(3.0)).process(x, FS)
    np.testing.assert_allclose(y, [5.0, 50.0], atol=1e-6)


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        fluxion.lowpass(-1.0, 4)  # negative cutoff


def test_dlpack_numpy_input():
    # numpy 2.x arrays are DLPack producers; process() consumes them via from_dlpack.
    x = np.arange(8, dtype=np.float32)
    np.testing.assert_allclose(fluxion.gain(2.0).process(x, FS), 2.0 * x, atol=1e-6)
    # non-float32 input is accepted (converted): float64 → float32.
    xd = np.arange(8, dtype=np.float64)
    np.testing.assert_allclose(fluxion.gain(2.0).process(xd, FS), 2.0 * xd.astype(np.float32), atol=1e-6)


def test_array_api_strict_input():
    """Array-API consumer conformance: accept arrays from any conforming library (strict ref impl)."""
    xp = pytest.importorskip("array_api_strict")
    src = np.linspace(-1.0, 1.0, 200, dtype=np.float32)
    x = xp.asarray(src)
    chain = fluxion.lowpass(1000.0, 4) | fluxion.gain(0.5)
    y = chain.process(x, FS)  # strict Array-API array in
    ref = chain.process(src, FS)
    np.testing.assert_allclose(np.asarray(y), ref, atol=1e-6)


def test_output_is_array_api_compliant():
    """fluxion's numpy output participates in the Array API (consumable by any conforming lib)."""
    compat = pytest.importorskip("array_api_compat")
    y = fluxion.gain(2.0).process(np.arange(8, dtype=np.float32), FS)
    xp = compat.array_namespace(y)
    assert xp is not None
    # round-trip a fluxion output back into a chain via the Array-API namespace.
    np.testing.assert_allclose(np.asarray(fluxion.gain(0.5).process(xp.asarray(y), FS)), y * 0.5, atol=1e-6)


def test_gpu_batch_matches_cpu():
    """GPU batch SOS filter == per-row CPU (only runs in the CUDA-built wheel on a GPU host)."""
    if not fluxion.cuda_available():
        pytest.skip("not a CUDA build")
    batch, frames = 64, 512
    coeffs = np.array(
        [0.2929, 0.5858, 0.2929, 0.0, 0.1716, 0.5, 0.3, -0.1, -0.2, 0.05], dtype=np.float32
    )  # two stable sections
    x = np.random.default_rng(0).standard_normal(batch * frames).astype(np.float32)

    y_gpu = fluxion.sos_filter_batch_gpu(x, frames, coeffs)
    y_cpu = np.concatenate(
        [fluxion.sos_forward(x[r * frames : (r + 1) * frames], coeffs) for r in range(batch)]
    )
    assert float(np.max(np.abs(y_gpu - y_cpu))) < 1e-4


def test_dlpack_torch_roundtrip():
    """Consume a torch tensor (DLPack) directly; hand the output back to torch via from_dlpack."""
    torch = pytest.importorskip("torch")
    x = torch.linspace(-1.0, 1.0, 1000, dtype=torch.float32)
    y = fluxion.lowpass(1000.0, 4).process(x, FS)  # torch tensor in, no .numpy()
    yt = torch.from_dlpack(y)  # numpy out → torch, zero-copy
    ref = fluxion.lowpass(1000.0, 4).process(x.numpy(), FS)
    np.testing.assert_allclose(yt.numpy(), ref, atol=1e-6)
    assert yt.shape[0] == 1000


# --- differentiable primitives: finite-difference gradcheck --------------------------------------

ONE_BIQUAD = np.array([0.3, 0.5, 0.2, -0.2, 0.05], dtype=np.float32)  # stable section


def _loss(x, coeffs, seed):
    return float(np.dot(seed, fluxion.sos_forward(x, coeffs)))


def test_sos_backward_gradchecks():
    rng = np.random.default_rng(1)
    n = 64
    x = rng.standard_normal(n).astype(np.float32)
    seed = rng.standard_normal(n).astype(np.float32)  # grad_out = dL/dy

    gx, gc = fluxion.sos_backward(seed, x, ONE_BIQUAD)
    eps = 1e-3

    # d loss / d x
    for i in (0, 5, 20, 63):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        fd = (_loss(xp, ONE_BIQUAD, seed) - _loss(xm, ONE_BIQUAD, seed)) / (2 * eps)
        assert abs(gx[i] - fd) < 1e-2, (i, gx[i], fd)

    # d loss / d coeffs
    for j in range(5):
        cp, cm = ONE_BIQUAD.copy(), ONE_BIQUAD.copy()
        cp[j] += eps
        cm[j] -= eps
        fd = (_loss(x, cp, seed) - _loss(x, cm, seed)) / (2 * eps)
        assert abs(gc[j] - fd) < 1e-2 * (1 + abs(gc[j])), (j, gc[j], fd)


def test_torch_adapter_matches_primitives():
    """The shipped fluxion.torch adapter's forward/backward agree with the (gradchecked) primitives."""
    torch = pytest.importorskip("torch")
    import fluxion.torch as fxt

    xv = np.random.default_rng(2).standard_normal(48).astype(np.float32)
    seed = np.random.default_rng(3).standard_normal(48).astype(np.float32)
    x = torch.tensor(xv, requires_grad=True)
    c = torch.tensor(ONE_BIQUAD, requires_grad=True)

    y = fxt.sos_filter(x, c)
    np.testing.assert_allclose(y.detach().numpy(), fluxion.sos_forward(xv, ONE_BIQUAD), atol=1e-6)

    y.backward(torch.from_numpy(seed))
    gx, gc = fluxion.sos_backward(seed, xv, ONE_BIQUAD)
    np.testing.assert_allclose(x.grad.numpy(), gx, atol=1e-5)
    np.testing.assert_allclose(c.grad.numpy(), gc, atol=1e-5)


def test_jax_adapter_gradchecks():
    """The shipped fluxion.jax custom_vjp adapter's gradient matches the analytic VJP."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    import fluxion.jax as fxj

    xv = np.random.default_rng(4).standard_normal(32).astype(np.float32)
    seed = np.random.default_rng(5).standard_normal(32).astype(np.float32)

    def loss(x, c):
        return jnp.sum(jnp.asarray(seed) * fxj.sos_filter(x, c))

    gx, gc = jax.grad(loss, argnums=(0, 1))(jnp.asarray(xv), jnp.asarray(ONE_BIQUAD))
    rgx, rgc = fluxion.sos_backward(seed, xv, ONE_BIQUAD)
    np.testing.assert_allclose(np.asarray(gx), rgx, atol=1e-4)
    np.testing.assert_allclose(np.asarray(gc), rgc, atol=1e-4)


def test_lowpass_parity_with_scipy():
    """fluxion's Butterworth low-pass matches scipy's design+filter to f32 precision."""
    sig = pytest.importorskip("scipy.signal")

    rng = np.random.default_rng(7)
    x = rng.standard_normal(2000).astype(np.float32)
    fs, fc, order = 48_000, 1000.0, 6
    y = fluxion.lowpass(fc, order).process(x, fs)

    sos = sig.butter(order, fc / (fs / 2), btype="low", output="sos")
    ref = sig.sosfilt(sos, x.astype(np.float64)).astype(np.float32)

    rel_rms = float(np.sqrt(np.mean((y - ref) ** 2)) / np.sqrt(np.mean(ref**2)))
    assert rel_rms < 1e-2, f"relative RMS vs scipy = {rel_rms}"


def test_cheby2_parity_with_scipy():
    """fluxion's Chebyshev II low-pass matches scipy's cheby2 design+filter to f32 precision."""
    sig = pytest.importorskip("scipy.signal")

    rng = np.random.default_rng(11)
    x = rng.standard_normal(2000).astype(np.float32)
    fs, fc, order, rs = 48_000, 2000.0, 6, 40.0
    y = fluxion.cheby2_lowpass(fc, order, rs).process(x, fs)

    sos = sig.cheby2(order, rs, fc / (fs / 2), btype="low", output="sos")
    ref = sig.sosfilt(sos, x.astype(np.float64)).astype(np.float32)

    rel_rms = float(np.sqrt(np.mean((y - ref) ** 2)) / np.sqrt(np.mean(ref**2)))
    assert rel_rms < 1e-2, f"relative RMS vs scipy = {rel_rms}"


def test_process_2d_multichannel():
    # A 2-D (C, T) input is one multichannel signal: each channel equals the 1-D result (J10).
    chain = fluxion.lowpass(2000.0, 4)
    rng = np.random.default_rng(3)
    x2 = rng.standard_normal((2, 500)).astype(np.float32)
    y2 = chain.process(x2, FS)
    assert y2.shape == (2, 500)
    for c in range(2):
        np.testing.assert_allclose(y2[c], chain.process(x2[c], FS), atol=1e-5)


def test_process_rejects_3d():
    with pytest.raises(ValueError):
        fluxion.gain(1.0).process(np.zeros((2, 2, 2), dtype=np.float32), FS)


def test_fir_impulse_response_is_the_taps():
    taps = [0.2, -0.5, 0.3]
    impulse = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    y = fluxion.fir(taps).process(impulse, FS)
    np.testing.assert_allclose(y[:3], np.array(taps, dtype=np.float32), atol=1e-6)


def test_sos_coeffs_seeds_from_a_design():
    coeffs = fluxion.lowpass(3000.0, 4).sos_coeffs(FS)  # 4th order = 2 sections × 5
    assert coeffs.shape == (10,)
    with pytest.raises(ValueError):  # not a pure cascade
        (fluxion.lowpass(1000.0, 2) | fluxion.gain(0.5)).sos_coeffs(FS)


def test_torch_sos_module_is_trainable():
    torch = pytest.importorskip("torch")
    import fluxion.torch as fxt

    m = fxt.SosModule.from_chain(fluxion.lowpass(2000.0, 4), FS)
    assert isinstance(m.coeffs, torch.nn.Parameter)
    x = torch.randn(256, requires_grad=True)
    m(x).pow(2).sum().backward()
    assert m.coeffs.grad is not None and x.grad is not None


# --- J9: batched CPU path ------------------------------------------------------------------------


def test_process_batch_matches_per_row():
    """Chain.process_batch on a (B, T) array == stacking per-row Chain.process (the batch contract)."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, 500)).astype(np.float32)
    chain = fluxion.lowpass(1500.0, 6) | fluxion.highpass(200.0, 2)  # pure filter → SIMD batch path
    y = chain.process_batch(x, FS)
    assert y.shape == x.shape and y.dtype == np.float32
    for r in range(x.shape[0]):
        np.testing.assert_allclose(y[r], chain.process(x[r], FS), atol=1e-5)


def test_process_batch_fallback_nonfilter():
    """A non-filter chain (gain) still batches correctly via the per-signal fallback."""
    x = np.array([[1.0, 10.0], [2.0, 3.0]], dtype=np.float32)
    y = fluxion.gain(2.0).process_batch(x, FS)
    np.testing.assert_allclose(y, 2.0 * x, atol=1e-6)


def test_process_batch_rejects_1d():
    with pytest.raises(ValueError):
        fluxion.gain(1.0).process_batch(np.zeros(10, dtype=np.float32), FS)


# --- J12: augmentation ---------------------------------------------------------------------------


def test_augment_determinism_and_range():
    from fluxion.augment import Compose, RandomChain

    # Two identically-seeded pipelines produce bit-identical output (reproducibility).
    x = np.random.default_rng(0).standard_normal(1000).astype(np.float32)
    make = lambda: Compose(
        [RandomChain(fluxion.lowpass, cutoff=(200.0, 8000.0), order=4, p=1.0, rng=42)]
    )
    np.testing.assert_array_equal(make()(x, FS), make()(x, FS))

    # Sampled params land in the requested range; the fixed param stays fixed.
    rc = RandomChain(fluxion.lowpass, cutoff=(200.0, 8000.0), order=4, rng=7)
    for _ in range(200):
        p = rc.sample()
        assert 200.0 <= p["cutoff"] < 8000.0
        assert p["order"] == 4


def test_augment_p_zero_is_identity():
    from fluxion.augment import RandomChain

    x = np.arange(16, dtype=np.float32)
    rc = RandomChain(fluxion.gain, value=(0.1, 0.9), p=0.0, rng=1)
    np.testing.assert_array_equal(rc(x, FS), x)  # never applied → untouched


def test_augment_integer_range_samples_ints():
    from fluxion.augment import RandomChain

    rc = RandomChain(fluxion.lowpass, cutoff=1000.0, order=(2, 8), rng=3)
    for _ in range(50):
        o = rc.sample()["order"]
        assert isinstance(o, int) and 2 <= o <= 8


# --- J13: FLAMO/DDSP checkpoint import ------------------------------------------------------------


def test_interop_coeff_layout_feeds_sos_forward():
    from fluxion.interop import load_flamo_sos

    # Fake FLAMO checkpoint: two biquad sections as (N, 3) numerator/denominator (a0 != 1).
    b = np.array([[0.3, 0.5, 0.2], [1.0, -1.5, 0.7]], dtype=np.float64)
    a = np.array([[2.0, -0.4, 0.1], [1.0, -0.2, 0.05]], dtype=np.float64)  # first row a0 = 2
    sos6 = load_flamo_sos({"filters.0.b": b, "filters.0.a": a})
    assert sos6.shape == (2, 6) and sos6.dtype == np.float32
    np.testing.assert_allclose(sos6[:, 3], [1.0, 1.0], atol=1e-6)  # a0 normalised to 1
    np.testing.assert_allclose(sos6[0, :3], b[0] / 2.0, atol=1e-6)  # b row scaled by 1/a0

    # (N, 5) fluxion layout feeds sos_forward directly.
    coeffs = load_flamo_sos({"b": b, "a": a}, layout="fluxion").ravel()
    assert coeffs.shape == (10,)
    impulse = np.zeros(8, dtype=np.float32)
    impulse[0] = 1.0
    y = fluxion.sos_forward(impulse, coeffs)
    assert y.shape == (8,) and np.isfinite(y).all()


def test_interop_rbj_peaking_matches_hand_math():
    from fluxion.interop import load_flamo_sos

    freq, gain_db, q, fs = 1000.0, 6.0, 0.707, 48_000.0
    sd = {
        "freq": np.array([freq]),
        "gain_db": np.array([gain_db]),
        "Q": np.array([q]),
        "fs": fs,
    }
    sos = load_flamo_sos(sd)
    # Recompute the RBJ peaking biquad by hand and compare.
    amp = 10.0 ** (gain_db / 40.0)
    w0 = 2 * np.pi * freq / fs
    alpha = np.sin(w0) / (2 * q)
    a0 = 1 + alpha / amp
    expect = (
        np.array(
            [1 + alpha * amp, -2 * np.cos(w0), 1 - alpha * amp, a0, -2 * np.cos(w0), 1 - alpha / amp]
        )
        / a0
    )
    np.testing.assert_allclose(sos[0], expect, rtol=1e-5)


def test_interop_svf_raises_clear_error():
    from fluxion.interop import load_flamo_sos

    with pytest.raises(ValueError, match="SVF"):
        load_flamo_sos({"m_lp": np.zeros(2), "m_bp": np.zeros(2), "m_hp": np.zeros(2)})


def test_interop_unknown_layout_raises():
    from fluxion.interop import load_flamo_sos

    with pytest.raises(ValueError, match="unrecognised"):
        load_flamo_sos({"weight": np.zeros((2, 2))})


def test_interop_safetensors_path(tmp_path):
    """A .safetensors path loads via the lazy safetensors import (the 'interop' extra)."""
    st = pytest.importorskip("safetensors.numpy")
    b = np.array([[0.3, 0.5, 0.2]], dtype=np.float32)
    a = np.array([[1.0, -0.2, 0.05]], dtype=np.float32)
    path = tmp_path / "cascade.safetensors"
    st.save_file({"b": b, "a": a}, str(path))
    from fluxion.interop import load_flamo_sos

    sos = load_flamo_sos(str(path))
    assert sos.shape == (1, 6)
    np.testing.assert_allclose(sos[0], [0.3, 0.5, 0.2, 1.0, -0.2, 0.05], atol=1e-6)


def test_dataset_parquet_roundtrip(tmp_path):
    pa = pytest.importorskip("pyarrow")
    from fluxion import dataset

    items = [
        (np.array([[0.0, 0.5, -0.5, 1.0], [0.1, -0.1, 0.2, -0.2]], dtype=np.float32), 48_000),
        (np.array([0.25, -0.25, 0.75], dtype=np.float32), 16_000),  # mono (T,)
    ]
    path = tmp_path / "ds.parquet"
    n = dataset.write_parquet(str(path), items)
    assert n == 2

    # Schema must match the Rust fluxion_io::arrow contract (uint32 / uint16 / list<float32>).
    schema = pa.parquet.read_schema(str(path))
    assert schema.field("fs").type == pa.uint32()
    assert schema.field("channels").type == pa.uint16()
    assert schema.field("audio").type == pa.list_(pa.float32())

    back = dataset.read_parquet(str(path))
    assert len(back) == 2
    # Stereo comes back (2, T) verbatim; mono (T,) comes back (1, T).
    np.testing.assert_array_equal(back[0][0], items[0][0])
    assert back[0][1] == 48_000
    np.testing.assert_array_equal(back[1][0], items[1][0][None, :])
    assert back[1][1] == 16_000


def test_dataset_streaming_is_bounded_and_lazy(tmp_path):
    pytest.importorskip("pyarrow")
    from fluxion import dataset

    # A generator in, a generator out: write 500 clips in small row groups, stream them back.
    rng = np.random.default_rng(0)
    src = ((rng.standard_normal((1, 64)).astype(np.float32), 8_000) for _ in range(500))
    path = tmp_path / "big.parquet"
    n = dataset.write_parquet(str(path), src, row_group_size=32)
    assert n == 500

    seen = 0
    for audio, fs in dataset.iter_parquet(str(path), batch_size=32):
        assert audio.shape == (1, 64)
        assert fs == 8_000
        seen += 1
    assert seen == 500


def test_dataset_augment_pipeline_composes(tmp_path):
    pytest.importorskip("pyarrow")
    import fluxion
    from fluxion import dataset

    rng = np.random.default_rng(1)
    items = [(rng.standard_normal((1, 128)).astype(np.float32), 48_000) for _ in range(4)]
    in_path, out_path = tmp_path / "in.parquet", tmp_path / "out.parquet"
    dataset.write_parquet(str(in_path), items)

    # End-to-end streaming augmentation: read -> gain(0.5) -> write, all lazy.
    g = fluxion.gain(0.5)
    dataset.write_parquet(
        str(out_path),
        ((g.process(x, fs), fs) for x, fs in dataset.iter_parquet(str(in_path))),
    )
    back = dataset.read_parquet(str(out_path))
    assert len(back) == 4
    for (orig, _), (aug, _) in zip(items, back):
        np.testing.assert_allclose(aug, orig * 0.5, atol=1e-6)


def test_dataset_empty_writes_valid_file(tmp_path):
    pa = pytest.importorskip("pyarrow")
    from fluxion import dataset

    path = tmp_path / "empty.parquet"
    assert dataset.write_parquet(str(path), []) == 0
    assert dataset.read_parquet(str(path)) == []
    # Still a schema-carrying, readable Parquet file.
    assert pa.parquet.read_schema(str(path)).field("fs").type == pa.uint32()

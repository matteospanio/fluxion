"""Tests for the realtime bindings (RtChain): streaming, hot-swap, and guard rails."""

import numpy as np
import pytest

import fluxion

FS = 48_000


def _stream(rt: "fluxion.RtChain", x: np.ndarray, block: int) -> np.ndarray:
    """Push `x` through `rt` in `block`-sized chunks, reusing one output buffer."""
    y = np.empty_like(x)
    out = np.empty(block, np.float32)
    for start in range(0, len(x), block):
        chunk = np.ascontiguousarray(x[start : start + block])
        o = out[: len(chunk)]
        rt.process(chunk, o)
        y[start : start + len(chunk)] = o
    return y


def test_streaming_matches_whole_signal_iir():
    """Chunked RtChain output equals the one-shot batch Chain.process (state carried across blocks)."""
    chain = fluxion.highpass(150.0, 4) | fluxion.lowpass(15_000.0, 4) | fluxion.peaking(500.0, -5.0, 1.4)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(5000).astype(np.float32)

    whole = chain.process(x, FS)
    rt = fluxion.RtChain.from_chain(chain, FS, max_block=333)
    streamed = _stream(rt, x, 333)  # odd block that doesn't divide the length

    np.testing.assert_allclose(streamed, whole, atol=1e-5)


def test_streaming_matches_whole_signal_fir_and_gain():
    """A chain with a long FIR and a gain (the soundlamp shape) streams identically to batch."""
    rng = np.random.default_rng(1)
    taps = (rng.standard_normal(1024) / 64.0).astype(np.float32)
    chain = fluxion.lowpass(4000.0, 4) | fluxion.fir(taps.tolist()) | fluxion.gain(0.5)
    x = rng.standard_normal(4096).astype(np.float32)

    whole = chain.process(x, FS)
    rt = fluxion.RtChain.from_chain(chain, FS, max_block=256)
    streamed = _stream(rt, x, 256)

    peak = float(np.abs(whole).max())
    np.testing.assert_allclose(streamed, whole, atol=1e-4 * max(peak, 1.0))


def test_state_is_carried_and_reset():
    """Two identical blocks give different outputs (state), and reset() restores the start."""
    chain = fluxion.lowpass(1000.0, 4)
    rt = fluxion.RtChain.from_chain(chain, FS, max_block=64)
    x = np.ones(64, np.float32)
    a, b, c = (np.empty(64, np.float32) for _ in range(3))

    rt.process(x, a)
    rt.process(x, b)  # carries the filter state from the first block
    assert not np.allclose(a, b), "state must persist across blocks"

    rt.reset()
    rt.process(x, c)
    np.testing.assert_array_equal(a, c)


def test_from_sections_matches_from_chain():
    """RtChain.from_sections(chain.sos_coeffs) behaves exactly like from_chain for a pure cascade."""
    chain = fluxion.lowpass(2000.0, 6)
    sections = chain.sos_coeffs(FS).reshape(-1, 5)
    rng = np.random.default_rng(2)
    x = rng.standard_normal(2000).astype(np.float32)

    a = _stream(fluxion.RtChain.from_chain(chain, FS, max_block=128), x, 128)
    rt = fluxion.RtChain.from_sections(sections, max_block=128)
    assert rt.fs is None
    b = _stream(rt, x, 128)
    np.testing.assert_array_equal(a, b)


def test_set_coeffs_crossfades_to_the_new_filter():
    """Swapping lp→hp on DC slides the output 1→0 without a click and settles near 0."""
    rt = fluxion.RtChain.from_chain(fluxion.lowpass(2000.0, 2), FS, max_block=200)
    hp = fluxion.highpass(2000.0, 2).sos_coeffs(FS).reshape(-1, 5)

    dc = np.ones(200, np.float32)
    out = np.empty(200, np.float32)
    y = []
    for i in range(20):
        if i == 10:
            rt.set_coeffs(0, hp, fade_samples=1000)
        rt.process(dc, out)
        y.append(out.copy())
    y = np.concatenate(y)

    assert abs(y[1999] - 1.0) < 0.02, "should sit at the low-pass DC gain before the swap"
    jump = np.abs(np.diff(y[2000:3000])).max()
    assert jump < 0.02, f"click during the crossfade: {jump}"
    assert np.abs(y[3500:]).max() < 0.02, "should settle at the high-pass DC gain (0)"


def test_set_coeffs_routes_to_the_addressed_node_in_series():
    """Swapping node 1 (the second filter, depth-first) to a high-pass must kill DC."""
    chain = fluxion.lowpass(2000.0, 2) | fluxion.lowpass(2000.0, 2)
    rt = fluxion.RtChain.from_chain(chain, FS, max_block=256)
    assert rt.filter_count == 2

    hp = fluxion.highpass(2000.0, 2).sos_coeffs(FS).reshape(-1, 5)
    rt.set_coeffs(1, hp, fade_samples=1)
    dc = np.ones(256, np.float32)
    out = np.empty(256, np.float32)
    for _ in range(30):
        rt.process(dc, out)
    assert np.abs(out).max() < 1e-3, "node 1 swap must reach the second series filter"


def test_parallel_chain_filter_count_and_node_routing():
    """A parallel chain (lp + hp) exposes both branches as addressable filter nodes.

    On DC only the low branch passes (sum ≈ 1); swapping node 1 — the right/high-pass branch,
    depth-first — to another low-pass doubles the DC gain (sum ≈ 2), proving the index routed
    to the right branch and not the left.
    """
    chain = fluxion.lowpass(2000.0, 2) + fluxion.highpass(2000.0, 2)
    rt = fluxion.RtChain.from_chain(chain, FS, max_block=256)
    assert rt.filter_count == 2

    dc = np.ones(256, np.float32)
    out = np.empty(256, np.float32)
    for _ in range(30):
        rt.process(dc, out)
    assert abs(out[-1] - 1.0) < 0.02, "before the swap only the low branch passes DC"

    lp = fluxion.lowpass(2000.0, 2).sos_coeffs(FS).reshape(-1, 5)
    rt.set_coeffs(1, lp, fade_samples=1)
    for _ in range(30):
        rt.process(dc, out)
    assert abs(out[-1] - 2.0) < 0.02, "after the swap both branches pass DC"


def test_guard_rails():
    chain = fluxion.lowpass(1000.0, 4)
    rt = fluxion.RtChain.from_chain(chain, FS, max_block=64)
    x64, y64 = np.zeros(64, np.float32), np.zeros(64, np.float32)

    with pytest.raises(ValueError, match="max_block"):
        rt.process(np.zeros(65, np.float32), np.zeros(65, np.float32))
    with pytest.raises(ValueError, match="same length"):
        rt.process(x64, np.zeros(32, np.float32))
    with pytest.raises(TypeError):  # strict dtype: no silent float64 → float32 copies
        rt.process(x64.astype(np.float64), y64)
    with pytest.raises(IndexError):
        rt.set_coeffs(5, chain.sos_coeffs(FS).reshape(-1, 5))
    with pytest.raises(ValueError, match="max_block must be >= 1"):
        fluxion.RtChain.from_chain(chain, FS, max_block=0)


def test_rejects_non_realtime_chain():
    """normalize needs the whole signal's peak — no realtime lowering."""
    with pytest.raises(ValueError, match="realtime"):
        fluxion.RtChain.from_chain(fluxion.lowpass(1000.0, 4) | fluxion.normalize(1.0), FS)


def test_rejects_unstable_sections():
    """Poles outside the unit circle are refused at construction and on a live swap."""
    unstable = np.array([[1.0, 0.0, 0.0, 0.0, 1.5]], np.float32)
    with pytest.raises(ValueError, match="unstable"):
        fluxion.RtChain.from_sections(unstable)

    rt = fluxion.RtChain.from_chain(fluxion.lowpass(1000.0, 2), FS)
    with pytest.raises(ValueError, match="unstable"):
        rt.set_coeffs(0, unstable)


def test_introspection_getters():
    chain = fluxion.lowpass(1000.0, 4) | fluxion.peaking(500.0, -5.0, 1.4) | fluxion.gain(0.5)
    rt = fluxion.RtChain.from_chain(chain, FS, max_block=512)
    # lowpass and peaking design to SOS filter nodes; gain is not addressable.
    assert rt.filter_count == 2
    assert rt.max_block == 512
    assert rt.fs == FS
    assert rt.verdict == "certified-stable"
    assert rt.margin > 0.0
    assert "RtChain" in repr(rt)

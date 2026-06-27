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

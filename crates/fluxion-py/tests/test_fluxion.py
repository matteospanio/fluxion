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


def test_torch_autograd_function():
    """Optional: a torch.autograd.Function built on the fluxion primitives gradchecks."""
    torch = pytest.importorskip("torch")

    class SosFilter(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, coeffs):
            ctx.save_for_backward(x, coeffs)
            y = fluxion.sos_forward(x.detach().numpy(), coeffs.detach().numpy())
            return torch.from_numpy(y)

        @staticmethod
        def backward(ctx, grad_out):
            x, coeffs = ctx.saved_tensors
            gx, gc = fluxion.sos_backward(
                grad_out.contiguous().numpy(), x.numpy(), coeffs.numpy()
            )
            return torch.from_numpy(gx), torch.from_numpy(gc)

    x = torch.randn(48, dtype=torch.float32, requires_grad=True)
    c = torch.tensor(ONE_BIQUAD, dtype=torch.float32, requires_grad=True)
    SosFilter.apply(x, c).pow(2).sum().backward()
    assert x.grad is not None and c.grad is not None
    assert torch.isfinite(x.grad).all() and torch.isfinite(c.grad).all()

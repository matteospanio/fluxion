"""``torch.autograd.Function`` adapter for fluxion's differentiable SOS cascade.

Wraps the analytic forward/backward (``fluxion.sos_forward`` / ``sos_backward``) so gradients flow
through PyTorch's tape — no recursion unrolling, the IIR adjoint is computed in closed form.

    >>> import torch, fluxion.torch as fxt
    >>> x = torch.randn(256, requires_grad=True)
    >>> coeffs = torch.tensor([0.3, 0.5, 0.2, -0.2, 0.05], requires_grad=True)  # one biquad
    >>> y = fxt.sos_filter(x, coeffs)
    >>> y.pow(2).sum().backward()  # x.grad and coeffs.grad are now populated
"""

from __future__ import annotations

import numpy as np
import torch

from ._fluxion import sos_backward, sos_forward


def _np(t: "torch.Tensor") -> np.ndarray:
    return t.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)


class _SosFilter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: "torch.Tensor", coeffs: "torch.Tensor") -> "torch.Tensor":
        ctx.save_for_backward(x, coeffs)
        y = sos_forward(_np(x), _np(coeffs))
        return torch.from_numpy(y).to(x.device, x.dtype)

    @staticmethod
    def backward(ctx, grad_out: "torch.Tensor"):
        x, coeffs = ctx.saved_tensors
        gx, gc = sos_backward(_np(grad_out), _np(x), _np(coeffs))
        return (
            torch.from_numpy(gx).to(x.device, x.dtype),
            torch.from_numpy(gc).to(coeffs.device, coeffs.dtype),
        )


def sos_filter(x: "torch.Tensor", coeffs: "torch.Tensor") -> "torch.Tensor":
    """Apply an SOS cascade to ``x``, differentiable in both ``x`` and ``coeffs``.

    ``coeffs`` is a flat ``[b0, b1, b2, a1, a2] · n_sections`` tensor.
    """
    return _SosFilter.apply(x, coeffs)

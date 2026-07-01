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


class SosModule(torch.nn.Module):
    """A trainable SOS cascade as an ``nn.Module``: its coefficients are an ``nn.Parameter`` and
    ``forward`` runs the analytic-VJP :func:`sos_filter`, so it drops into a torch model / optimizer
    and its coefficients train through fluxion's closed-form IIR adjoint.

        >>> import fluxion, torch, fluxion.torch as fxt
        >>> m = fxt.SosModule.from_chain(fluxion.lowpass(2000.0, 4), fs=48_000)  # seed from a design
        >>> y = m(torch.randn(512))                # differentiable; m.coeffs is an nn.Parameter
        >>> list(m.parameters())[0].shape          # trainable in an optimizer
        torch.Size([10])
    """

    def __init__(self, coeffs) -> None:
        super().__init__()
        c = torch.as_tensor(coeffs, dtype=torch.float32).flatten()
        if c.numel() == 0 or c.numel() % 5 != 0:
            raise ValueError(
                "coeffs must be a non-empty multiple of 5 ([b0,b1,b2,a1,a2] per section)"
            )
        self.coeffs = torch.nn.Parameter(c)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return sos_filter(x, self.coeffs)

    @classmethod
    def from_chain(cls, chain, fs: int) -> "SosModule":
        """Seed a trainable module from a designed pure-filter :class:`fluxion.Chain` at ``fs`` (via
        ``chain.sos_coeffs(fs)``)."""
        return cls(chain.sos_coeffs(fs))

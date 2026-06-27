"""``jax.custom_vjp`` adapter for fluxion's differentiable SOS cascade.

fluxion runs on the host (CPU/NumPy), so the forward and backward are invoked via ``jax.pure_callback``
and wired together with ``jax.custom_vjp`` to expose the analytic gradient to JAX.

    >>> import jax.numpy as jnp, fluxion.jax as fxj
    >>> x = jnp.asarray(...)          # float32
    >>> coeffs = jnp.asarray([0.3, 0.5, 0.2, -0.2, 0.05])
    >>> y = fxj.sos_filter(x, coeffs)
"""

from __future__ import annotations

import jax
import numpy as np

from ._fluxion import sos_backward, sos_forward


def _forward(x, coeffs):
    out = jax.ShapeDtypeStruct(x.shape, x.dtype)
    return jax.pure_callback(
        lambda xv, cv: sos_forward(
            np.asarray(xv, np.float32), np.asarray(cv, np.float32)
        ),
        out,
        x,
        coeffs,
    )


@jax.custom_vjp
def sos_filter(x, coeffs):
    """Apply an SOS cascade to ``x``, differentiable in both ``x`` and ``coeffs``."""
    return _forward(x, coeffs)


def _fwd(x, coeffs):
    return _forward(x, coeffs), (x, coeffs)


def _bwd(res, grad_out):
    x, coeffs = res
    shapes = (
        jax.ShapeDtypeStruct(x.shape, x.dtype),
        jax.ShapeDtypeStruct(coeffs.shape, coeffs.dtype),
    )
    gx, gc = jax.pure_callback(
        lambda g, xv, cv: sos_backward(
            np.asarray(g, np.float32),
            np.asarray(xv, np.float32),
            np.asarray(cv, np.float32),
        ),
        shapes,
        grad_out,
        x,
        coeffs,
    )
    return gx, gc


sos_filter.defvjp(_fwd, _bwd)

"""fluxion — GPU-accelerated, differentiable audio DSP.

An eager, torchaudio-style API (build a :class:`Chain`, apply it to an array) plus differentiable
SOS primitives. Framework autograd adapters live in the optional submodules :mod:`fluxion.torch`
(``torch.autograd.Function``) and :mod:`fluxion.jax` (``jax.custom_vjp``) — import them only if you
have torch / jax installed.

**Array-API interop.** fluxion is an Array-API *consumer*: every function accepts an array from any
conforming library (NumPy, PyTorch, JAX, CuPy, ``array_api_strict``, …) via the DLPack interchange —
zero-copy when it is already float32 + C-contiguous — and returns a NumPy array, which is itself
Array-API-compliant (and a DLPack producer, so frameworks consume it zero-copy). fluxion is not an
Array-API *namespace provider*: it's a transform library, not a general array library.
"""

from ._fluxion import __cuda__
from ._fluxion import (
    Chain,
    allpass,
    bandpass,
    cheby1_highpass,
    cheby1_lowpass,
    cheby2_highpass,
    cheby2_lowpass,
    delay,
    echo,
    fir,
    gain,
    high_shelf,
    highpass,
    low_shelf,
    lowpass,
    normalize,
    notch,
    peaking,
    reverb,
    sos_backward,
    sos_forward,
)

__all__ = [
    "Chain",
    "lowpass",
    "highpass",
    "cheby1_lowpass",
    "cheby1_highpass",
    "cheby2_lowpass",
    "cheby2_highpass",
    "peaking",
    "low_shelf",
    "high_shelf",
    "notch",
    "bandpass",
    "allpass",
    "reverb",
    "fir",
    "gain",
    "normalize",
    "delay",
    "echo",
    "sos_forward",
    "sos_backward",
    "cuda_available",
]


def cuda_available() -> bool:
    """True if this wheel was built with CUDA support (the "GPU wheel")."""
    return bool(__cuda__)


# The GPU batch filter exists only in the CUDA-built wheel.
if __cuda__:
    from ._fluxion import sos_filter_batch_gpu  # noqa: F401

    __all__.append("sos_filter_batch_gpu")

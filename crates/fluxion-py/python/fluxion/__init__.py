"""fluxion — GPU-accelerated, differentiable audio DSP.

An eager, torchaudio-style API (build a :class:`Chain`, apply it to a NumPy array) plus differentiable
SOS primitives. Framework autograd adapters live in the optional submodules :mod:`fluxion.torch`
(``torch.autograd.Function``) and :mod:`fluxion.jax` (``jax.custom_vjp``) — import them only if you
have torch / jax installed.
"""

from ._fluxion import (
    Chain,
    allpass,
    bandpass,
    cheby1_highpass,
    cheby1_lowpass,
    delay,
    echo,
    gain,
    high_shelf,
    highpass,
    low_shelf,
    lowpass,
    normalize,
    notch,
    peaking,
    sos_backward,
    sos_forward,
)

__all__ = [
    "Chain",
    "lowpass",
    "highpass",
    "cheby1_lowpass",
    "cheby1_highpass",
    "peaking",
    "low_shelf",
    "high_shelf",
    "notch",
    "bandpass",
    "allpass",
    "gain",
    "normalize",
    "delay",
    "echo",
    "sos_forward",
    "sos_backward",
]

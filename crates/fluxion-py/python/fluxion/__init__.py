"""fluxion — GPU-accelerated, differentiable audio DSP.

Build a chain out of effects, apply it to audio::

    import fluxion as fx

    wave = fx.Wave.from_file("in.wav")
    (wave | fx.filter.Highpass(80, order=4) | fx.effect.Gain(fx.db(-3))).save("out.wav")

``|`` is series, ``+`` is parallel (outputs summed) — the same algebra as the Rust library, the
CLI, and the browser. Effects live in :mod:`fluxion.filter` and :mod:`fluxion.effect`, both
generated from the one op registry, so a name here is the same name everywhere. The same chain can
be written as text and shared with any other interface::

    fx.chain("highpass(80, 4) | gain(-3dB)")

A chain is callable on a bare array too, for notebooks and batch jobs where a :class:`Wave` would
be in the way::

    y = fx.chain("highpass(80, 4)")(x, fs=48_000)

**Array-API interop.** fluxion is an Array-API *consumer*: every function accepts an array from any
conforming library (NumPy, PyTorch, JAX, CuPy, ``array_api_strict``, …) via the DLPack interchange —
zero-copy when it is already float32 + C-contiguous — and returns a NumPy array, which is itself
Array-API-compliant (and a DLPack producer, so frameworks consume it zero-copy). fluxion is not an
Array-API *namespace provider*: it's a transform library, not a general array library.

Framework autograd adapters live in the optional submodules :mod:`fluxion.torch`
(``torch.autograd.Function``) and :mod:`fluxion.jax` (``jax.custom_vjp``) — import them only if you
have torch / jax installed.
"""

from . import augment, dataset, effect, interop
from . import filter  # noqa: A004  (shadows the builtin on purpose, so `fx.filter.Lowpass` reads)
from .augment import Compose, RandomChain
from .wave import Wave
from ._fluxion import (
    Automation,
    Chain,
    RtChain,
    __cuda__,
    chain,
    ensure_fs,
    ops_table,
    read_audio,
    sos_backward,
    sos_forward,
    write_audio,
)

__all__ = [
    "Automation",
    "Chain",
    "RtChain",
    "Wave",
    "chain",
    "db",
    "ensure_fs",
    "ops_table",
    "read_audio",
    "write_audio",
    "sos_forward",
    "sos_backward",
    "cuda_available",
    "effect",
    "filter",
    "augment",
    "dataset",
    "interop",
    "Compose",
    "RandomChain",
]


def db(decibels: float) -> float:
    """Decibels as a linear amplitude ratio: ``db(-3)`` is 0.708.

    Ops whose parameter is a plain ratio — ``Gain``, ``Normalize`` — take linear values, because
    that is what the DSP multiplies by. This makes the intent explicit at the call site:
    ``fx.effect.Gain(fx.db(-3))``. In chain text the same thing is written ``gain(-3dB)``.
    """
    return 10.0 ** (decibels / 20.0)


def cuda_available() -> bool:
    """True if this wheel was built with CUDA support (the "GPU wheel")."""
    return bool(__cuda__)


# The GPU batch filter exists only in the CUDA-built wheel.
if __cuda__:
    from ._fluxion import sos_filter_batch_gpu  # noqa: F401

    __all__.append("sos_filter_batch_gpu")

"""Data-augmentation transforms built on fluxion chains.

Pure Python, numpy-only. Two small composables for stochastic audio augmentation:

* :class:`Compose` — apply a sequence of transforms (fluxion :class:`~fluxion.Chain` objects and/or
  plain ``(x, fs) -> x`` callables) one after another.
* :class:`RandomChain` — build a fluxion chain with *fresh, randomly sampled* parameters on every
  call, gated by an apply-probability ``p``. Seed it (or pass a NumPy ``Generator``) for reproducible
  augmentation pipelines.

    >>> import numpy as np, fluxion
    >>> from fluxion.augment import Compose, RandomChain
    >>> aug = Compose([
    ...     RandomChain(fluxion.lowpass, cutoff=(500.0, 8000.0), order=4, p=0.5, rng=0),
    ...     fluxion.gain(0.9),
    ... ])
    >>> x = np.random.default_rng(0).standard_normal(1000).astype(np.float32)
    >>> y = aug(x, 48_000)          # same seed → same output every run
"""

from __future__ import annotations

from typing import Any, Callable, Sequence, Union

import numpy as np

# A transform is either a fluxion Chain (has ``.process(x, fs)``) or a ``(x, fs) -> x`` callable.
Transform = Union[Any, Callable[[Any, int], Any]]

__all__ = ["Compose", "RandomChain", "Transform"]


def _apply(t: Transform, x: Any, fs: int) -> Any:
    """Apply one transform: a fluxion ``Chain`` (via ``.process``) or a ``(x, fs)`` callable."""
    process = getattr(t, "process", None)
    if callable(process):
        return process(x, fs)
    return t(x, fs)


class Compose:
    """Apply a list of transforms in sequence (left to right).

    Each transform is either a fluxion :class:`~fluxion.Chain` (applied via ``chain.process(x, fs)``)
    or any ``(x, fs) -> x`` callable (e.g. a :class:`RandomChain`, or your own augmentation).
    """

    def __init__(self, transforms: Sequence[Transform]) -> None:
        self.transforms: list[Transform] = list(transforms)

    def __call__(self, x: Any, fs: int) -> Any:
        for t in self.transforms:
            x = _apply(t, x, fs)
        return x

    def __repr__(self) -> str:
        return f"Compose({self.transforms!r})"


class RandomChain:
    """Sample a fluxion chain's parameters afresh on every call, for stochastic augmentation.

    Pass a chain constructor and its parameters. A parameter given as a ``(lo, hi)`` 2-tuple is
    sampled uniformly on each call — an integer range (both ends ``int``) draws an integer in
    ``[lo, hi]`` inclusive, otherwise a float in ``[lo, hi)``. Any other value is passed through
    unchanged (a fixed parameter). With probability ``1 - p`` the whole transform is skipped and the
    input is returned untouched.

    Seed reproducibly with ``rng`` — an ``int`` seed, a NumPy ``Generator``, or ``None`` (fresh,
    non-deterministic).

        >>> import fluxion
        >>> from fluxion.augment import RandomChain
        >>> rc = RandomChain(fluxion.lowpass, cutoff=(200.0, 8000.0), order=4, p=0.8, rng=0)
        >>> params = rc.sample()           # e.g. {'cutoff': 5123.4, 'order': 4}
        >>> chain = rc.build()             # a fresh fluxion.Chain with sampled params
    """

    def __init__(
        self,
        constructor: Callable[..., Any],
        *,
        p: float = 1.0,
        rng: Union[int, np.random.Generator, None] = None,
        **params: Any,
    ) -> None:
        self.constructor = constructor
        self.p = float(p)
        self.params = params
        self._rng = np.random.default_rng(rng)

    def sample(self) -> dict[str, Any]:
        """Draw one concrete parameter dict from the configured ranges/fixed values."""
        out: dict[str, Any] = {}
        for name, spec in self.params.items():
            if _is_range(spec):
                lo, hi = spec
                if isinstance(lo, int) and isinstance(hi, int):
                    out[name] = int(self._rng.integers(lo, hi + 1))
                else:
                    out[name] = float(self._rng.uniform(lo, hi))
            else:
                out[name] = spec
        return out

    def build(self) -> Any:
        """Sample parameters and construct a fresh chain (``constructor(**sample())``)."""
        return self.constructor(**self.sample())

    def __call__(self, x: Any, fs: int) -> Any:
        # Gate first, then sample: a skipped call still advances the rng deterministically.
        if self._rng.random() >= self.p:
            return x
        return self.build().process(x, fs)

    def __repr__(self) -> str:
        return f"RandomChain({self.constructor!r}, p={self.p}, params={self.params!r})"


def _is_range(spec: Any) -> bool:
    """True if ``spec`` is a ``(lo, hi)`` numeric 2-tuple (a sampling range), not a fixed value."""
    return (
        isinstance(spec, tuple)
        and len(spec) == 2
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in spec)
    )

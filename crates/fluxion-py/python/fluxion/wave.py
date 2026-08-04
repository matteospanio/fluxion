"""``Wave`` — audio that knows its own sample rate.

``fs`` is the single most repeated argument in an audio API, and the single easiest one to get
wrong. A ``Wave`` carries it, so it disappears from user code::

    import fluxion as fx

    wave = fx.Wave.from_file("in.wav")
    (wave | fx.filter.Highpass(80, order=4) | fx.effect.Gain(fx.db(-3))).save("out.wav")

``|`` does **not** process immediately. It accumulates the chain and runs it once, when the samples
are actually asked for — so ``w | a | b | c`` is one pass over the audio with the filters fused,
not three. Nothing about that is visible except in how long it takes.

Samples are planar and channel-first, ``(channels, frames)``, matching ``Signal`` on the Rust side.
A 1-D array is promoted to one channel.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np

from . import _fluxion
from ._fluxion import Chain

__all__ = ["Wave"]


class Wave:
    """A block of audio and its sample rate.

    Build one from a file with :meth:`from_file`, or directly from an array: ``Wave(ys, fs)``.
    """

    __slots__ = ("_ys", "_plan", "fs", "metadata")

    def __init__(self, ys, fs: int, metadata: dict | None = None) -> None:
        self._ys = self._as_channels_first(ys)
        self.fs = int(fs)
        self.metadata: dict = dict(metadata or {})
        # Effects piped in but not yet run. `None` means "the samples are current".
        self._plan: Chain | None = None

    # --- construction -------------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: str | os.PathLike, fs: int | None = None) -> "Wave":
        """Read an audio file. WAV, FLAC, MP3, OGG — whatever the CLI reads, decoded identically.

        Pass ``fs`` to pin the result to a project rate: files already at it are not touched, the
        rest are converted on the way in. ``metadata["source_fs"]`` remembers what the file was.
        """
        ys, file_fs = _fluxion.read_audio(os.fspath(path))
        wave = cls(ys, file_fs, {"path": os.fspath(path)})
        return wave if fs is None else wave.ensure_fs(fs)

    @staticmethod
    def _as_channels_first(ys) -> np.ndarray:
        arr = np.ascontiguousarray(ys, dtype=np.float32)
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        if arr.ndim == 2:
            return arr
        raise ValueError(
            f"a Wave holds (channels, frames) or a 1-D signal, got a {arr.ndim}-D array "
            f"with shape {arr.shape}"
        )

    # --- samples ------------------------------------------------------------------------------

    @property
    def ys(self) -> np.ndarray:
        """The samples, ``(channels, frames)`` float32. Runs any pending effects first."""
        if self._plan is not None:
            self._ys = np.ascontiguousarray(self._plan(self._ys, self.fs), dtype=np.float32)
            if self._ys.ndim == 1:
                self._ys = self._ys.reshape(1, -1)
            self._plan = None
        return self._ys

    def channels(self) -> int:
        """Number of channels."""
        return self._ys.shape[0]

    def __len__(self) -> int:
        """Number of frames (samples per channel)."""
        return self._ys.shape[1]

    def duration(self) -> float:
        """Length in seconds."""
        return len(self) / self.fs

    # --- composition --------------------------------------------------------------------------

    def __or__(self, effect: Chain) -> "Wave":
        """``wave | effect`` — queue an effect, returning a new Wave. Nothing runs yet."""
        if not isinstance(effect, Chain):
            raise TypeError(
                f"expected a fluxion effect or chain on the right of '|', got "
                f"{type(effect).__name__}"
            )
        out = Wave.__new__(Wave)
        out._ys = self._ys
        out.fs = self.fs
        out.metadata = dict(self.metadata)
        # Compose rather than apply: one pass at the end instead of one per effect.
        out._plan = effect if self._plan is None else (self._plan | effect)
        return out

    # --- sample rate --------------------------------------------------------------------------

    def ensure_fs(self, fs: int) -> "Wave":
        """This wave at ``fs``, converting only if it is not already there.

        A host pins one project rate and puts every input through here; ``fs`` never has to be
        thought about again. The result has exactly ``round(len(self) * fs / self.fs)`` frames.
        """
        fs = int(fs)
        if fs <= 0:
            raise ValueError(f"a sample rate must be positive, got {fs}")
        if fs == self.fs:
            return self
        out = Wave(_fluxion.ensure_fs(self.ys, self.fs, fs), fs, self.metadata)
        out.metadata["source_fs"] = self.fs
        return out

    # --- channels -----------------------------------------------------------------------------

    def get_channel(self, index: int) -> "Wave":
        """One channel as its own mono Wave — the read half of per-channel routing."""
        return Wave(self.ys[index : index + 1], self.fs, self.metadata)

    @classmethod
    def merge(cls, waves: Sequence["Wave"], split_channels: bool = False) -> "Wave":
        """Combine waves: summed by default, or stacked as separate channels.

        Summing zero-pads to the longest input; stacking requires equal lengths. Either way the
        sample rates must agree — resampling is an explicit step, not something a merge does behind
        your back.
        """
        if not waves:
            raise ValueError("merge needs at least one Wave")
        fs = waves[0].fs
        if any(w.fs != fs for w in waves):
            rates = sorted({w.fs for w in waves})
            raise ValueError(f"cannot merge waves at different sample rates: {rates}")

        arrays = [w.ys for w in waves]
        if split_channels:
            if len({a.shape[1] for a in arrays}) > 1:
                raise ValueError("stacking channels needs equal lengths; trim or pad first")
            return cls(np.concatenate(arrays, axis=0), fs)

        if len({a.shape[0] for a in arrays}) > 1:
            raise ValueError("summing waves needs the same channel count in each")
        frames = max(a.shape[1] for a in arrays)
        total = np.zeros((arrays[0].shape[0], frames), dtype=np.float32)
        for a in arrays:
            total[:, : a.shape[1]] += a
        return cls(total, fs)

    # --- output -------------------------------------------------------------------------------

    def save(self, path: str | os.PathLike, bits: int | None = None) -> None:
        """Write a WAV file. ``bits`` is ``None`` for 32-bit float, or 16 / 24 / 32 for PCM."""
        _fluxion.write_audio(os.fspath(path), self.ys, self.fs, bits)

    def __repr__(self) -> str:
        pending = f", pending={str(self._plan)!r}" if self._plan is not None else ""
        return (
            f"Wave({self._ys.shape[0]} ch, {self._ys.shape[1]} frames, "
            f"{self.fs} Hz{pending})"
        )

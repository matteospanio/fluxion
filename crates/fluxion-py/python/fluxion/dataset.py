"""Columnar audio-dataset IO (Parquet) for the augmentation pipeline.

Reads and writes the **same schema** as the Rust ``fluxion_io::arrow`` module, so a dataset written
by the Rust library/CLI reads here and vice versa. One row per clip:

======== =============== =========================================
column   type            meaning
======== =============== =========================================
fs       uint32          sample rate (Hz)
channels uint16          channel count
audio    list<float32>   samples, **interleaved** (frame-major)
======== =============== =========================================

A dataset item is a ``(audio, fs)`` pair, where ``audio`` is a 1-D ``(T,)`` mono array or a 2-D
``(C, T)`` channels-first array — the same layout :meth:`fluxion.Chain.process` accepts. Reads always
return ``(C, T)`` (mono comes back as ``(1, T)``).

This module needs `pyarrow`; install the extra::

    pip install "fluxion[dataset]"

It composes with :mod:`fluxion.augment` and :meth:`~fluxion.Chain.process` — streaming end to end,
bounded memory (a generator in, a generator out)::

    import fluxion
    from fluxion.dataset import iter_parquet, write_parquet
    aug = fluxion.augment.RandomChain(fluxion.lowpass, cutoff=(500.0, 8000.0), rng=0)
    write_parquet("out.parquet", ((aug(x, fs), fs) for x, fs in iter_parquet("in.parquet")))
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator, Tuple

import numpy as np

# One dataset row: (audio, fs). `audio` is (T,) or (C, T) on write; always (C, T) on read.
Item = Tuple[Any, int]

__all__ = ["write_parquet", "read_parquet", "iter_parquet", "Item"]


def _pyarrow():
    """Import pyarrow with a clear message pointing at the extra when it's missing."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "fluxion.dataset needs pyarrow — install it with:  pip install 'fluxion[dataset]'"
        ) from e
    return pa, pq


def _schema(pa):
    return pa.schema(
        [
            ("fs", pa.uint32()),
            ("channels", pa.uint16()),
            ("audio", pa.list_(pa.float32())),
        ]
    )


def _as_ct(audio: Any) -> np.ndarray:
    """Coerce an item's audio to a 2-D ``(C, T)`` float32 array (mono ``(T,)`` -> ``(1, T)``)."""
    a = np.ascontiguousarray(audio, dtype=np.float32)
    if a.ndim == 1:
        a = a[None, :]
    elif a.ndim != 2:
        raise ValueError(f"audio must be 1-D (T,) or 2-D (C, T), got {a.ndim}-D")
    return a


def _interleave(a_ct: np.ndarray) -> np.ndarray:
    """(C, T) channels-first -> frame-major interleaved 1-D [c0f0, c1f0, …, c0f1, …]."""
    return np.ascontiguousarray(a_ct.T).reshape(-1)


def write_parquet(
    path: Any,
    items: Iterable[Item],
    *,
    row_group_size: int = 256,
    compression: str = "snappy",
) -> int:
    """Write ``(audio, fs)`` items to a Parquet file and return the row count.

    ``items`` may be any iterable — a list or a lazy generator; rows are written in groups of
    ``row_group_size`` so a generator input stays bounded-memory. ``compression`` is any codec
    pyarrow supports (``"snappy"`` default, ``"zstd"``, ``"none"``); the Rust reader handles
    snappy/zstd/uncompressed.
    """
    pa, pq = _pyarrow()
    schema = _schema(pa)
    writer = None
    fs_buf: list[int] = []
    ch_buf: list[int] = []
    au_buf: list[np.ndarray] = []
    count = 0

    def flush() -> None:
        nonlocal writer
        if not fs_buf:
            return
        table = pa.table(
            {
                "fs": pa.array(fs_buf, type=pa.uint32()),
                "channels": pa.array(ch_buf, type=pa.uint16()),
                "audio": pa.array(au_buf, type=pa.list_(pa.float32())),
            },
            schema=schema,
        )
        if writer is None:
            writer = pq.ParquetWriter(str(path), schema, compression=compression)
        writer.write_table(table)
        fs_buf.clear()
        ch_buf.clear()
        au_buf.clear()

    try:
        for audio, fs in items:
            a = _as_ct(audio)
            fs_buf.append(int(fs))
            ch_buf.append(int(a.shape[0]))
            au_buf.append(_interleave(a))
            count += 1
            if len(fs_buf) >= row_group_size:
                flush()
        flush()
        # An empty dataset still writes a valid, schema-carrying (0-row) file.
        if writer is None:
            writer = pq.ParquetWriter(str(path), schema, compression=compression)
    finally:
        if writer is not None:
            writer.close()
    return count


def iter_parquet(path: Any, *, batch_size: int = 256) -> Iterator[Item]:
    """Stream ``(audio, fs)`` items from a Parquet file one row at a time (bounded memory).

    ``audio`` is a ``(C, T)`` float32 array. Reads the schema written by :func:`write_parquet` or by
    the Rust ``fluxion_io::arrow`` writer.
    """
    _, pq = _pyarrow()
    pf = pq.ParquetFile(str(path))
    for batch in pf.iter_batches(batch_size=batch_size):
        fs_col = batch.column("fs").to_numpy(zero_copy_only=False)
        ch_col = batch.column("channels").to_numpy(zero_copy_only=False)
        audio_col = batch.column("audio")
        for i in range(batch.num_rows):
            flat = audio_col[i].values.to_numpy(zero_copy_only=False).astype(
                np.float32, copy=False
            )
            nch = int(ch_col[i])
            audio = flat[None, :] if nch <= 1 else flat.reshape(-1, nch).T
            yield np.ascontiguousarray(audio), int(fs_col[i])


def read_parquet(path: Any) -> list[Item]:
    """Read an entire Parquet dataset into a list of ``(audio, fs)`` items.

    Convenience over :func:`iter_parquet`; use the iterator for datasets that don't fit in memory.
    """
    return list(iter_parquet(path))

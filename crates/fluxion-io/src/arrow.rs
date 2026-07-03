//! Columnar dataset IO (plan task H4): [`Signal`] ↔ Apache Arrow `RecordBatch` ↔ Parquet.
//!
//! Behind the optional `parquet` feature — the arrow/parquet stack is heavy, so only dataset
//! consumers (the augmentation pipeline, goal 3) pull it in. One row per [`Signal`]; the schema is
//!
//! | column     | type            | meaning                                        |
//! |------------|-----------------|------------------------------------------------|
//! | `fs`       | `UInt32`        | sample rate (Hz)                               |
//! | `channels` | `UInt16`        | channel count                                  |
//! | `audio`    | `List<Float32>` | samples, **interleaved** (frame-major)         |
//!
//! Interleaving keeps each clip in one variable-length list cell (channels of a clip share a rate
//! and length), so a whole dataset is a flat, filterable table that round-trips exactly.

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, Float32Array, Float32Builder, ListArray, ListBuilder, UInt16Array, UInt32Array,
};
use arrow::error::ArrowError;
use arrow::record_batch::RecordBatch;
use fluxion_core::Signal;

use crate::planar_from_interleaved;

/// Interleave a signal's channels frame-major, zero-padding short channels to the longest so the
/// row is rectangular (mirrors the WAV encoder's contract).
fn interleave(signal: &Signal) -> Vec<f32> {
    let frames = signal.frames();
    let nch = signal.channel_count();
    let mut out = Vec::with_capacity(frames * nch);
    for f in 0..frames {
        for ch in &signal.channels {
            out.push(ch.get(f).copied().unwrap_or(0.0));
        }
    }
    out
}

/// Build an Arrow [`RecordBatch`] (one row per signal) from a slice of [`Signal`]s.
///
/// ```
/// # use fluxion_core::Signal;
/// let sig = Signal::new(48_000, vec![vec![0.0, 1.0], vec![-1.0, 0.5]]);
/// let batch = fluxion_io::arrow::signals_to_batch(&[sig]).unwrap();
/// assert_eq!(batch.num_rows(), 1);
/// ```
pub fn signals_to_batch(signals: &[Signal]) -> Result<RecordBatch, ArrowError> {
    let fs = UInt32Array::from_iter_values(signals.iter().map(|s| s.fs));
    let channels = UInt16Array::from_iter_values(signals.iter().map(|s| s.channel_count() as u16));

    let mut audio = ListBuilder::new(Float32Builder::new());
    for s in signals {
        audio.values().append_slice(&interleave(s));
        audio.append(true);
    }
    let audio: ListArray = audio.finish();

    RecordBatch::try_from_iter(vec![
        ("fs", Arc::new(fs) as ArrayRef),
        ("channels", Arc::new(channels) as ArrayRef),
        ("audio", Arc::new(audio) as ArrayRef),
    ])
}

/// Reconstruct [`Signal`]s from a [`RecordBatch`] produced by [`signals_to_batch`] (or a Parquet
/// file written by [`write_parquet`]). Errors if a required column is missing or mistyped.
pub fn batch_to_signals(batch: &RecordBatch) -> Result<Vec<Signal>, ArrowError> {
    let fs = downcast::<UInt32Array>(batch, "fs")?;
    let channels = downcast::<UInt16Array>(batch, "channels")?;
    let audio = downcast::<ListArray>(batch, "audio")?;

    let mut out = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        let nch = channels.value(row) as usize;
        let cell = audio.value(row);
        let samples = cell
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| ArrowError::CastError("audio list values are not Float32".into()))?;
        let planar = planar_from_interleaved(samples.values(), nch);
        out.push(Signal::new(fs.value(row), planar));
    }
    Ok(out)
}

/// Downcast a named column to a concrete array type, with a clear error on absence/mismatch.
fn downcast<'a, A: Array + 'static>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<&'a A, ArrowError> {
    batch
        .column_by_name(name)
        .ok_or_else(|| ArrowError::SchemaError(format!("missing column '{name}'")))?
        .as_any()
        .downcast_ref::<A>()
        .ok_or_else(|| ArrowError::CastError(format!("column '{name}' has an unexpected type")))
}

/// Write a slice of [`Signal`]s to a Parquet file (one row per signal). See the module docs for the
/// schema.
pub fn write_parquet(
    path: impl AsRef<std::path::Path>,
    signals: &[Signal],
) -> Result<(), parquet::errors::ParquetError> {
    let batch = signals_to_batch(signals)?;
    let file = std::fs::File::create(path)?;
    let mut writer = parquet::arrow::ArrowWriter::try_new(file, batch.schema(), None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

/// Read every [`Signal`] back from a Parquet file written by [`write_parquet`].
pub fn read_parquet(
    path: impl AsRef<std::path::Path>,
) -> Result<Vec<Signal>, parquet::errors::ParquetError> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    let file = std::fs::File::open(path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
    let mut signals = Vec::new();
    for batch in reader {
        signals.extend(batch_to_signals(&batch?)?);
    }
    Ok(signals)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_dataset() -> Vec<Signal> {
        vec![
            Signal::new(
                48_000,
                vec![vec![0.0, 0.5, -0.5, 1.0], vec![0.1, -0.1, 0.2, -0.2]],
            ),
            Signal::new(16_000, vec![vec![0.25, -0.25, 0.75]]), // different fs, channels, length
        ]
    }

    #[test]
    fn batch_roundtrip_is_exact() {
        let ds = sample_dataset();
        let batch = signals_to_batch(&ds).unwrap();
        assert_eq!(batch.num_rows(), 2);
        let back = batch_to_signals(&batch).unwrap();

        assert_eq!(back.len(), 2);
        for (a, b) in ds.iter().zip(&back) {
            assert_eq!(a.fs, b.fs);
            assert_eq!(a.channel_count(), b.channel_count());
            assert_eq!(a.channels, b.channels); // f32 stored verbatim
        }
    }

    #[test]
    fn parquet_roundtrip_is_exact() {
        let ds = sample_dataset();
        let path =
            std::env::temp_dir().join(format!("fluxion_io_arrow_{}.parquet", std::process::id()));
        write_parquet(&path, &ds).unwrap();
        let back = read_parquet(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(back.len(), ds.len());
        for (a, b) in ds.iter().zip(&back) {
            assert_eq!(a.fs, b.fs);
            assert_eq!(a.channels, b.channels);
        }
    }
}

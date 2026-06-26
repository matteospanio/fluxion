//! `fluxion-io` — audio and batch IO.
//!
//! Today: WAV read/write via [`hound`], decoding to / encoding from the planar
//! [`Signal`](fluxion_core::Signal) buffer the DSP engine works in. Pure Rust — no
//! libsndfile/ffmpeg. Symphonia decode (flac/mp3/ogg/aac) and Arrow/Parquet batch IO land later
//! (see `PROJECT.md` §7).

use std::path::Path;

use fluxion_core::Signal;
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};

/// Read a WAV file into a planar [`Signal`].
///
/// Integer PCM is normalized by the format's full-scale value; float WAV is passed through.
pub fn read_wav(path: impl AsRef<Path>) -> Result<Signal, hound::Error> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();
    let n = (spec.channels as usize).max(1);
    let mut channels: Vec<Vec<f32>> = vec![Vec::new(); n];

    match spec.sample_format {
        SampleFormat::Float => {
            for (i, s) in reader.samples::<f32>().enumerate() {
                channels[i % n].push(s?);
            }
        }
        SampleFormat::Int => {
            // Full-scale for signed N-bit PCM is 2^(N-1).
            let scale = (1u64 << (spec.bits_per_sample - 1)) as f32;
            for (i, s) in reader.samples::<i32>().enumerate() {
                channels[i % n].push(s? as f32 / scale);
            }
        }
    }

    Ok(Signal::new(spec.sample_rate, channels))
}

/// Write a planar [`Signal`] to a 32-bit float WAV (lossless round-trip).
///
/// Shorter channels are zero-padded to the longest so the interleaved stream stays rectangular.
pub fn write_wav(path: impl AsRef<Path>, signal: &Signal) -> Result<(), hound::Error> {
    let spec = WavSpec {
        channels: signal.channel_count() as u16,
        sample_rate: signal.fs,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };
    let mut writer = WavWriter::create(path, spec)?;
    let frames = signal.frames();
    for f in 0..frames {
        for ch in &signal.channels {
            writer.write_sample(ch.get(f).copied().unwrap_or(0.0))?;
        }
    }
    writer.finalize()
}

#[cfg(test)]
mod tests {
    use super::{read_wav, write_wav};
    use fluxion_core::Signal;

    #[test]
    fn wav_float_roundtrip() {
        let original = Signal::new(
            48_000,
            vec![vec![0.0, 0.5, -0.5, 1.0], vec![0.1, -0.1, 0.25, -0.25]],
        );
        // ponytail: fixed temp path — fine for a single serial test, revisit if tests parallelize.
        let path = std::env::temp_dir().join("fluxion_io_roundtrip.wav");
        write_wav(&path, &original).expect("write");
        let read_back = read_wav(&path).expect("read");
        let _ = std::fs::remove_file(&path);

        assert_eq!(read_back.fs, original.fs);
        assert_eq!(read_back.channels, original.channels); // f32 WAV is bit-exact
    }
}

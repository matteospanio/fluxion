//! `fluxion-io` — audio and batch IO.
//!
//! Today: WAV read/write via [`hound`], decoding to / encoding from a simple planar [`Audio`]
//! buffer (channel-first, the layout the DSP engine works in). Pure Rust — no libsndfile/ffmpeg.
//! Symphonia decode (flac/mp3/ogg/aac) and Arrow/Parquet batch IO land later (see `PROJECT.md` §7).

use std::path::Path;

use hound::{SampleFormat, WavReader, WavSpec, WavWriter};

/// A decoded audio buffer in **planar, channel-first** layout: `channels[c]` holds every sample
/// for channel `c`, all channels the same length.
#[derive(Clone, Debug, PartialEq)]
pub struct Audio {
    /// Sample rate in Hz.
    pub fs: u32,
    /// One `Vec<f32>` of samples per channel (normalized to roughly `[-1.0, 1.0]`).
    pub channels: Vec<Vec<f32>>,
}

impl Audio {
    /// Number of channels.
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    /// Number of frames (samples per channel); `0` if there are no channels.
    pub fn frames(&self) -> usize {
        self.channels.first().map_or(0, Vec::len)
    }
}

/// Read a WAV file into a planar [`Audio`] buffer.
///
/// Integer PCM is normalized by the format's full-scale value; float WAV is passed through.
pub fn read_wav(path: impl AsRef<Path>) -> Result<Audio, hound::Error> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();
    let n = spec.channels as usize;
    let mut channels: Vec<Vec<f32>> = vec![Vec::new(); n.max(1)];

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

    Ok(Audio {
        fs: spec.sample_rate,
        channels,
    })
}

/// Write a planar [`Audio`] buffer to a 32-bit float WAV (lossless round-trip).
///
/// Shorter channels are zero-padded to the longest so the interleaved stream stays rectangular.
pub fn write_wav(path: impl AsRef<Path>, audio: &Audio) -> Result<(), hound::Error> {
    let spec = WavSpec {
        channels: audio.channel_count() as u16,
        sample_rate: audio.fs,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };
    let mut writer = WavWriter::create(path, spec)?;
    let frames = audio.frames();
    for f in 0..frames {
        for ch in &audio.channels {
            writer.write_sample(ch.get(f).copied().unwrap_or(0.0))?;
        }
    }
    writer.finalize()
}

#[cfg(test)]
mod tests {
    use super::{Audio, read_wav, write_wav};

    #[test]
    fn wav_float_roundtrip() {
        let original = Audio {
            fs: 48_000,
            channels: vec![vec![0.0, 0.5, -0.5, 1.0], vec![0.1, -0.1, 0.25, -0.25]],
        };
        // ponytail: fixed temp path — fine for a single serial test, revisit if tests parallelize.
        let path = std::env::temp_dir().join("fluxion_io_roundtrip.wav");
        write_wav(&path, &original).expect("write");
        let read_back = read_wav(&path).expect("read");
        let _ = std::fs::remove_file(&path);

        assert_eq!(read_back.fs, original.fs);
        assert_eq!(read_back.channels, original.channels); // f32 WAV is bit-exact
    }
}

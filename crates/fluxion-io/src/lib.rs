//! `fluxion-io` — audio and batch IO.
//!
//! WAV read/write/probe via [`hound`], and [`decode`] for FLAC/MP3/OGG/AAC/… via Symphonia — both
//! to/from the planar [`Signal`](fluxion_core::Signal) buffer the DSP engine works in. Pure Rust, no
//! libsndfile/ffmpeg. Arrow/Parquet batch IO lands later (see `PROJECT.md` §7).

use std::io::{Cursor, Read, Write};
use std::path::Path;

use fluxion_core::Signal;
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Read a WAV file into a planar [`Signal`].
///
/// Integer PCM is normalized by the format's full-scale value; float WAV is passed through.
pub fn read_wav(path: impl AsRef<Path>) -> Result<Signal, hound::Error> {
    decode_wav(WavReader::open(path)?)
}

/// Read a WAV from any reader (e.g. stdin) into a planar [`Signal`].
pub fn read_wav_from(reader: impl Read) -> Result<Signal, hound::Error> {
    decode_wav(WavReader::new(reader)?)
}

/// Shared WAV decode body: planarize + normalize integer PCM, pass float through.
fn decode_wav<R: Read>(mut reader: WavReader<R>) -> Result<Signal, hound::Error> {
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
    encode_wav(WavWriter::create(path, wav_spec(signal))?, signal)
}

/// Write a planar [`Signal`] as 32-bit float WAV to any writer (e.g. stdout). The WAV is buffered in
/// memory first because the format needs a seekable sink (the header carries the length); stdout is
/// not seekable.
pub fn write_wav_to(mut writer: impl Write, signal: &Signal) -> Result<(), hound::Error> {
    let mut buf = Cursor::new(Vec::new());
    encode_wav(WavWriter::new(&mut buf, wav_spec(signal))?, signal)?;
    writer.write_all(&buf.into_inner())?;
    Ok(())
}

fn wav_spec(signal: &Signal) -> WavSpec {
    WavSpec {
        channels: signal.channel_count() as u16,
        sample_rate: signal.fs,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    }
}

/// Shared WAV encode body: interleave channels (zero-padding short ones) and finalize.
fn encode_wav<W: Write + std::io::Seek>(
    mut writer: WavWriter<W>,
    signal: &Signal,
) -> Result<(), hound::Error> {
    for f in 0..signal.frames() {
        for ch in &signal.channels {
            writer.write_sample(ch.get(f).copied().unwrap_or(0.0))?;
        }
    }
    writer.finalize()
}

/// Metadata about a WAV file, without decoding its samples.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WavInfo {
    /// Sample rate in Hz.
    pub fs: u32,
    /// Channel count.
    pub channels: u16,
    /// Bits per sample.
    pub bits: u16,
    /// `true` for IEEE-float samples, `false` for integer PCM.
    pub float: bool,
    /// Number of frames (samples per channel).
    pub frames: u32,
}

impl WavInfo {
    /// Duration in seconds.
    pub fn seconds(&self) -> f64 {
        if self.fs == 0 {
            0.0
        } else {
            self.frames as f64 / self.fs as f64
        }
    }
}

/// Read a WAV file's header metadata without decoding samples.
pub fn probe_wav(path: impl AsRef<Path>) -> Result<WavInfo, hound::Error> {
    let reader = WavReader::open(path)?;
    let spec = reader.spec();
    Ok(WavInfo {
        fs: spec.sample_rate,
        channels: spec.channels,
        bits: spec.bits_per_sample,
        float: spec.sample_format == SampleFormat::Float,
        frames: reader.duration(),
    })
}

/// Decode any Symphonia-supported audio file (FLAC, MP3, OGG/Vorbis, AAC, WAV, …) into a planar
/// [`Signal`], normalized to `f32`. The format is detected from the file's content and extension.
pub fn decode(path: impl AsRef<Path>) -> Result<Signal, SymphoniaError> {
    let path = path.as_ref();
    let file = std::fs::File::open(path).map_err(SymphoniaError::IoError)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;
    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or(SymphoniaError::Unsupported("no decodable audio track"))?;
    let track_id = track.id;
    let mut decoder =
        symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;

    let mut fs = track.codec_params.sample_rate.unwrap_or(0);
    let mut planar: Vec<Vec<f32>> = Vec::new();
    let mut buf: Option<SampleBuffer<f32>> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            // End of stream is signalled as an UnexpectedEof IO error.
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break;
            }
            Err(SymphoniaError::ResetRequired) => break,
            Err(e) => return Err(e),
        };
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(decoded) => {
                let spec = *decoded.spec();
                let ch = spec.channels.count();
                if fs == 0 {
                    fs = spec.rate;
                }
                if planar.is_empty() {
                    planar = vec![Vec::new(); ch.max(1)];
                }
                if buf.is_none() {
                    buf = Some(SampleBuffer::<f32>::new(decoded.capacity() as u64, spec));
                }
                let sb = buf.as_mut().unwrap();
                sb.copy_interleaved_ref(decoded);
                for (i, &s) in sb.samples().iter().enumerate() {
                    planar[i % ch].push(s);
                }
            }
            // Skip a corrupt packet rather than aborting the whole decode.
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(e) => return Err(e),
        }
    }

    Ok(Signal::new(fs.max(1), planar))
}

#[cfg(test)]
mod tests {
    use super::{decode, read_wav, write_wav};
    use fluxion_core::Signal;

    #[test]
    fn decode_reads_pcm_wav() {
        // Write a known 16-bit stereo PCM WAV, then decode it through Symphonia.
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 44_100,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let path = std::env::temp_dir().join("fluxion_decode_pcm.wav");
        let mut w = hound::WavWriter::create(&path, spec).unwrap();
        let frames: Vec<(i16, i16)> = (0..256)
            .map(|i| (i as i16 * 100 - 12_800, -(i as i16) * 80))
            .collect();
        for (l, r) in &frames {
            w.write_sample(*l).unwrap();
            w.write_sample(*r).unwrap();
        }
        w.finalize().unwrap();

        let sig = decode(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(sig.fs, 44_100);
        assert_eq!(sig.channel_count(), 2);
        assert_eq!(sig.frames(), 256);
        // Sample 10 normalized to [-1, 1).
        assert!((sig.channels[0][10] - frames[10].0 as f32 / 32_768.0).abs() < 1e-3);
        assert!((sig.channels[1][10] - frames[10].1 as f32 / 32_768.0).abs() < 1e-3);
    }

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

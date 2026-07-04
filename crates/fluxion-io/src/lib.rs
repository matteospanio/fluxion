//! `fluxion-io` — audio and batch IO.
//!
//! WAV read/write/probe via [`hound`], and [`decode`]/[`probe`] for FLAC/MP3/OGG/AAC/… via
//! Symphonia — both to/from the planar [`fluxion_core::Signal`] buffer the DSP engine works in.
//! Pure Rust, no libsndfile/ffmpeg. Writing defaults to 32-bit float ([`write_wav`]); integer PCM
//! (16/24/32-bit, with TPDF dither) is selectable via [`WavEncoding`] and [`write_wav_encoded`].
//! For large files, [`read_wav_blocks`] / [`decode_blocks`] stream fixed-size [`Signal`] chunks
//! with bounded memory. Columnar dataset IO ([`Signal`] ↔ Arrow/Parquet) is behind the optional
//! `parquet` feature (the `arrow` module).

#[cfg(feature = "parquet")]
pub mod arrow;

use std::io::{Cursor, Read, Write};
use std::path::Path;

use fluxion_core::Signal;
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{CODEC_TYPE_NULL, CodecType, DecoderOptions};
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
            // Full-scale for signed N-bit PCM is 2^(N-1). Clamp bits to a sane range so a malformed
            // header can't underflow/overflow the shift.
            let bits = spec.bits_per_sample.clamp(1, 64);
            let scale = (1u64 << (bits - 1)) as f32;
            for (i, s) in reader.samples::<i32>().enumerate() {
                channels[i % n].push(s? as f32 / scale);
            }
        }
    }

    Ok(Signal::new(spec.sample_rate, channels))
}

/// How to encode samples when writing a WAV file.
///
/// The [`Default`] is 32-bit IEEE float — bit-exact for the `[-1, 1]`-normalized [`Signal`] buffer
/// and the behavior of [`write_wav`]. Integer PCM (16/24/32-bit) is what most tools and hardware
/// actually consume; choose it to match SoX's default output widths.
///
/// **Clipping:** integer PCM has a bounded range, so each sample is clamped to `[-1.0, 1.0]` before
/// it is scaled to the integer full-scale. Values outside that range are hard-clipped (not wrapped).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WavEncoding {
    /// Bits per sample. Float: only `32`. Integer PCM: `16`, `24`, or `32`.
    pub bits: u16,
    /// `true` = IEEE float samples; `false` = signed integer PCM.
    pub float: bool,
    /// Apply 1-LSB triangular-PDF (TPDF) dither before integer quantization. Ignored for float,
    /// and skipped at 32-bit integer PCM (the f32 sample pipeline carries ~24 significant bits,
    /// coarser than a 32-bit LSB, so dither cannot survive the rounding there).
    ///
    /// On by default: adding a triangular ±1-LSB noise floor before rounding **decorrelates** the
    /// quantization error from the signal, replacing signal-dependent harmonic distortion (audible
    /// on quiet fades) with a benign, constant broadband hiss. Disable it only when you need
    /// bit-exact, reproducible quantization (e.g. round-trip tests).
    pub dither: bool,
}

impl Default for WavEncoding {
    /// 32-bit float, dither irrelevant — the lossless default matching [`write_wav`].
    fn default() -> Self {
        Self {
            bits: 32,
            float: true,
            dither: true,
        }
    }
}

impl WavEncoding {
    /// Signed integer PCM at `bits` (16/24/32), TPDF dither on. Convenience for the common case.
    pub fn pcm(bits: u16) -> Self {
        Self {
            bits,
            float: false,
            dither: true,
        }
    }
}

/// Write a planar [`Signal`] to a 32-bit float WAV (lossless round-trip).
///
/// Shorter channels are zero-padded to the longest so the interleaved stream stays rectangular.
/// For integer PCM or other widths, use [`write_wav_encoded`].
pub fn write_wav(path: impl AsRef<Path>, signal: &Signal) -> Result<(), hound::Error> {
    write_wav_encoded(path, signal, WavEncoding::default())
}

/// Write a planar [`Signal`] as 32-bit float WAV to any writer (e.g. stdout).
///
/// See [`write_wav_encoded_to`] for the buffering rationale and other encodings.
pub fn write_wav_to(writer: impl Write, signal: &Signal) -> Result<(), hound::Error> {
    write_wav_encoded_to(writer, signal, WavEncoding::default())
}

/// Write a planar [`Signal`] to a WAV file with an explicit [`WavEncoding`] (bit depth / float vs
/// integer PCM / dither).
///
/// Shorter channels are zero-padded to the longest so the interleaved stream stays rectangular.
pub fn write_wav_encoded(
    path: impl AsRef<Path>,
    signal: &Signal,
    enc: WavEncoding,
) -> Result<(), hound::Error> {
    let spec = wav_spec(signal, enc)?;
    encode_wav(WavWriter::create(path, spec)?, signal, enc)
}

/// Write a planar [`Signal`] with an explicit [`WavEncoding`] to any writer (e.g. stdout).
///
/// The WAV is buffered in memory first because the format needs a seekable sink (the RIFF header
/// carries the total length, patched on finalize); stdout is not seekable.
pub fn write_wav_encoded_to(
    mut writer: impl Write,
    signal: &Signal,
    enc: WavEncoding,
) -> Result<(), hound::Error> {
    let spec = wav_spec(signal, enc)?;
    let mut buf = Cursor::new(Vec::new());
    encode_wav(WavWriter::new(&mut buf, spec)?, signal, enc)?;
    writer.write_all(&buf.into_inner())?;
    Ok(())
}

/// Build (and validate) a hound [`WavSpec`] from a [`Signal`] and [`WavEncoding`].
fn wav_spec(signal: &Signal, enc: WavEncoding) -> Result<WavSpec, hound::Error> {
    let sample_format = match (enc.float, enc.bits) {
        (true, 32) => SampleFormat::Float,
        (false, 16 | 24 | 32) => SampleFormat::Int,
        // hound would reject these too, but a message naming the option is friendlier.
        (true, _) => return Err(hound::Error::Unsupported),
        (false, _) => return Err(hound::Error::Unsupported),
    };
    Ok(WavSpec {
        channels: signal.channel_count() as u16,
        sample_rate: signal.fs,
        bits_per_sample: enc.bits,
        sample_format,
    })
}

/// Fixed seed for the dither PRNG: deterministic dither makes encoding reproducible (bit-exact
/// output across runs), which matters for content-addressed pipelines and testable behavior. The
/// per-sample noise is still spectrally white, which is all TPDF dither needs.
const DITHER_SEED: u32 = 0x_C0FF_EE11;

/// Tiny non-cryptographic PRNG (xorshift32) driving TPDF dither, inline to avoid a `rand` dep.
struct XorShift(u32);

impl XorShift {
    /// Next 32-bit state. Seed is nonzero, so the sequence never collapses to 0.
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }

    /// Uniform float in `[0, 1)` from the top 24 bits (f32 has 24 mantissa bits).
    fn next_unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    /// One triangular-PDF sample in `(-1, 1)` LSB: the difference of two independent uniforms.
    fn tpdf(&mut self) -> f32 {
        self.next_unit() - self.next_unit()
    }
}

/// Quantize one `[-1, 1]` sample to signed integer PCM per `enc` (clamp, optional TPDF
/// dither, round). The single source of truth for both the whole-file and streaming writers.
#[inline]
fn quantize_pcm(x: f32, enc: WavEncoding, rng: &mut XorShift) -> i32 {
    // Signed N-bit PCM spans [-2^(N-1), 2^(N-1) - 1]; full-scale is 2^(N-1).
    let full_scale = (1u64 << (enc.bits - 1)) as f32;
    let max_code = (1i64 << (enc.bits - 1)) - 1;
    let min_code = -(1i64 << (enc.bits - 1));
    // Clamp to [-1, 1] first (documented clipping), then scale to code units.
    let mut v = x.clamp(-1.0, 1.0) * full_scale;
    // ponytail: the f32 pipeline caps 32-bit int PCM at ~24 significant bits and makes
    // 1-LSB dither a no-op there, so skip it; an f64 encode path is the upgrade if
    // true 32-bit int resolution is ever needed.
    if enc.dither && enc.bits < 32 {
        v += rng.tpdf();
    }
    (v.round() as i64).clamp(min_code, max_code) as i32
}

/// Shared WAV encode body: interleave channels (zero-padding short ones), quantize per `enc`, and
/// finalize.
fn encode_wav<W: Write + std::io::Seek>(
    mut writer: WavWriter<W>,
    signal: &Signal,
    enc: WavEncoding,
) -> Result<(), hound::Error> {
    let frames = signal.frames();
    if enc.float {
        for f in 0..frames {
            for ch in &signal.channels {
                writer.write_sample(ch.get(f).copied().unwrap_or(0.0))?;
            }
        }
    } else {
        let mut rng = XorShift(DITHER_SEED);
        for f in 0..frames {
            for ch in &signal.channels {
                let x = ch.get(f).copied().unwrap_or(0.0);
                writer.write_sample(quantize_pcm(x, enc, &mut rng))?;
            }
        }
    }
    writer.finalize()
}

/// Incremental WAV writer — the streaming counterpart to [`write_wav_encoded`], for
/// bounded-memory pipelines (plan task H5): feed planar blocks in order, then
/// [`finalize`](Self::finalize) (hound patches the RIFF length header). Quantization is
/// sample-for-sample identical to the whole-file writer: same clamp/scale/round and the same
/// deterministic TPDF dither stream carried across blocks.
pub struct WavBlockWriter {
    writer: WavWriter<std::io::BufWriter<std::fs::File>>,
    enc: WavEncoding,
    channels: usize,
    rng: XorShift,
}

impl WavBlockWriter {
    /// Create the output file with the given rate/channel/encoding spec.
    pub fn create(
        path: impl AsRef<Path>,
        fs: u32,
        channels: u16,
        enc: WavEncoding,
    ) -> Result<Self, hound::Error> {
        let sample_format = match (enc.float, enc.bits) {
            (true, 32) => SampleFormat::Float,
            (false, 16 | 24 | 32) => SampleFormat::Int,
            _ => return Err(hound::Error::Unsupported),
        };
        let spec = WavSpec {
            channels,
            sample_rate: fs,
            bits_per_sample: enc.bits,
            sample_format,
        };
        Ok(Self {
            writer: WavWriter::create(path, spec)?,
            enc,
            channels: channels as usize,
            rng: XorShift(DITHER_SEED),
        })
    }

    /// Append one planar block (`block[c]` = channel `c`). Channels must match the spec;
    /// short channels are zero-padded within the block, like the whole-file writer.
    pub fn write_block(&mut self, block: &[Vec<f32>]) -> Result<(), hound::Error> {
        assert_eq!(block.len(), self.channels, "channel count changed mid-stream");
        let frames = block.iter().map(Vec::len).max().unwrap_or(0);
        if self.enc.float {
            for f in 0..frames {
                for ch in block {
                    self.writer.write_sample(ch.get(f).copied().unwrap_or(0.0))?;
                }
            }
        } else {
            for f in 0..frames {
                for ch in block {
                    let x = ch.get(f).copied().unwrap_or(0.0);
                    let code = quantize_pcm(x, self.enc, &mut self.rng);
                    self.writer.write_sample(code)?;
                }
            }
        }
        Ok(())
    }

    /// Finish the file (writes the final RIFF header).
    pub fn finalize(self) -> Result<(), hound::Error> {
        self.writer.finalize()
    }
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

/// Container-level metadata about any audio file, obtained without decoding its samples. The
/// Symphonia analogue of [`WavInfo`], so the CLI `info` verb can describe FLAC/MP3/OGG too.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AudioInfo {
    /// Sample rate in Hz (`0` if the container does not declare it).
    pub fs: u32,
    /// Channel count (`0` if the container does not declare it).
    pub channels: u16,
    /// Total frames (samples per channel) if the container declares it in its header, else `None` —
    /// some streamed formats (e.g. a live OGG) don't store a length up front.
    pub frames: Option<u64>,
    /// Short codec name, e.g. `"flac"`, `"mp3"`, `"vorbis"`, `"pcm_s16le"`.
    pub codec: String,
}

impl AudioInfo {
    /// Duration in seconds, or `None` when the frame count or sample rate is unknown.
    pub fn seconds(&self) -> Option<f64> {
        match self.frames {
            Some(n) if self.fs > 0 => Some(n as f64 / self.fs as f64),
            _ => None,
        }
    }
}

/// Probe any Symphonia-supported audio file (FLAC, MP3, OGG/Vorbis, AAC, WAV, …) for header
/// metadata — sample rate, channels, frame count, codec — **without decoding its samples**.
///
/// Uses the same format-probe machinery as [`decode`], then reads the track's declared parameters;
/// no packets are decoded. Prefer [`probe_wav`] for WAV when you only need hound (no Symphonia).
pub fn probe(path: impl AsRef<Path>) -> Result<AudioInfo, SymphoniaError> {
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

    let track = probed
        .format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or(SymphoniaError::Unsupported("no decodable audio track"))?;
    let cp = &track.codec_params;

    Ok(AudioInfo {
        fs: cp.sample_rate.unwrap_or(0),
        channels: cp.channels.map_or(0, |c| c.count() as u16),
        frames: cp.n_frames,
        codec: codec_name(cp.codec),
    })
}

/// Human-readable short codec name from Symphonia's registry, or `"unknown"` if unregistered.
fn codec_name(codec: CodecType) -> String {
    symphonia::default::get_codecs()
        .get_codec(codec)
        .map_or_else(|| "unknown".to_string(), |d| d.short_name.to_string())
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

// --- streaming / chunked readers (plan task H5) ----------------------------------------------
//
// `read_wav` / `decode` buffer the whole file into one `Signal` — a 1-hour stereo file at 48 kHz
// is ~1.4 GB in RAM. The block readers below yield fixed-size `Signal` chunks so peak memory is
// bounded by `block_frames` regardless of file length (dataset preprocessing, the goal-3
// augmentation path). Each yields the file's frames in order; the final block may be shorter.

/// De-interleave a frame-major slice into planar channels (the last frame may be partial).
fn planar_from_interleaved(interleaved: &[f32], channels: usize) -> Vec<Vec<f32>> {
    let n = channels.max(1);
    let mut planar = vec![Vec::with_capacity(interleaved.len() / n + 1); n];
    for (i, &s) in interleaved.iter().enumerate() {
        planar[i % n].push(s);
    }
    planar
}

/// A bounded-memory iterator over a WAV file, yielding [`Signal`] blocks of `block_frames` frames
/// (the last block may be shorter). See [`read_wav_blocks`].
pub struct WavBlocks<R: Read> {
    reader: WavReader<R>,
    fs: u32,
    channels: usize,
    is_float: bool,
    int_scale: f32,
    block_samples: usize,
}

impl<R: Read> Iterator for WavBlocks<R> {
    type Item = Result<Signal, hound::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        // hound reads sequentially, so each `samples()` call continues from the current position.
        let taken: Result<Vec<f32>, hound::Error> = if self.is_float {
            self.reader
                .samples::<f32>()
                .take(self.block_samples)
                .collect()
        } else {
            let scale = self.int_scale;
            self.reader
                .samples::<i32>()
                .take(self.block_samples)
                .map(|r| r.map(|s| s as f32 / scale))
                .collect()
        };
        match taken {
            Ok(v) if v.is_empty() => None, // EOF
            Ok(v) => Some(Ok(Signal::new(
                self.fs,
                planar_from_interleaved(&v, self.channels),
            ))),
            Err(e) => Some(Err(e)),
        }
    }
}

/// Stream a WAV file as [`Signal`] blocks of `block_frames` frames each (bounded memory).
///
/// The returned iterator yields the file in order; the last block may be shorter. `block_frames`
/// is clamped to at least 1.
///
/// ```no_run
/// # fn main() -> Result<(), hound::Error> {
/// for block in fluxion_io::read_wav_blocks("big.wav", 65_536)? {
///     let sig = block?; // at most 65_536 frames in memory at once
///     // …process sig…
/// }
/// # Ok(()) }
/// ```
pub fn read_wav_blocks(
    path: impl AsRef<Path>,
    block_frames: usize,
) -> Result<WavBlocks<std::io::BufReader<std::fs::File>>, hound::Error> {
    read_wav_blocks_from(
        std::io::BufReader::new(std::fs::File::open(path)?),
        block_frames,
    )
}

/// Stream WAV blocks from any reader (e.g. stdin). See [`read_wav_blocks`].
pub fn read_wav_blocks_from<R: Read>(
    reader: R,
    block_frames: usize,
) -> Result<WavBlocks<R>, hound::Error> {
    let reader = WavReader::new(reader)?;
    let spec = reader.spec();
    let channels = (spec.channels as usize).max(1);
    let bits = spec.bits_per_sample.clamp(1, 64);
    Ok(WavBlocks {
        fs: spec.sample_rate,
        channels,
        is_float: spec.sample_format == SampleFormat::Float,
        int_scale: (1u64 << (bits - 1)) as f32,
        block_samples: block_frames.max(1) * channels,
        reader,
    })
}

/// A bounded-memory iterator over any Symphonia-decodable file (FLAC/MP3/OGG/AAC/WAV/…), yielding
/// [`Signal`] blocks of `block_frames` frames. See [`decode_blocks`].
pub struct AudioBlocks {
    format: Box<dyn symphonia::core::formats::FormatReader>,
    decoder: Box<dyn symphonia::core::codecs::Decoder>,
    track_id: u32,
    fs: u32,
    channels: usize,
    block_frames: usize,
    pending: Vec<Vec<f32>>, // planar accumulator, drained in block_frames-sized chunks
    sbuf: Option<SampleBuffer<f32>>,
    done: bool,
}

impl AudioBlocks {
    /// Frames currently buffered (the shortest channel bounds a whole-frame count).
    fn pending_frames(&self) -> usize {
        self.pending.iter().map(Vec::len).min().unwrap_or(0)
    }

    /// Drain the first `n` frames off the planar accumulator into a [`Signal`].
    fn take_block(&mut self, n: usize) -> Signal {
        let channels: Vec<Vec<f32>> = self
            .pending
            .iter_mut()
            .map(|ch| ch.drain(..n.min(ch.len())).collect())
            .collect();
        Signal::new(self.fs, channels)
    }
}

impl Iterator for AudioBlocks {
    type Item = Result<Signal, SymphoniaError>;

    fn next(&mut self) -> Option<Self::Item> {
        while !self.done && self.pending_frames() < self.block_frames {
            let packet = match self.format.next_packet() {
                Ok(p) => p,
                Err(SymphoniaError::IoError(e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    self.done = true;
                    break;
                }
                Err(SymphoniaError::ResetRequired) => {
                    self.done = true;
                    break;
                }
                Err(e) => {
                    self.done = true;
                    return Some(Err(e));
                }
            };
            if packet.track_id() != self.track_id {
                continue;
            }
            match self.decoder.decode(&packet) {
                Ok(decoded) => {
                    let spec = *decoded.spec();
                    let ch = spec.channels.count().max(1);
                    if self.fs == 0 {
                        self.fs = spec.rate;
                    }
                    if self.pending.is_empty() {
                        self.pending = vec![Vec::new(); ch];
                        self.channels = ch;
                    }
                    if self.sbuf.is_none() {
                        self.sbuf = Some(SampleBuffer::<f32>::new(decoded.capacity() as u64, spec));
                    }
                    let sb = self.sbuf.as_mut().unwrap();
                    sb.copy_interleaved_ref(decoded);
                    for (i, &s) in sb.samples().iter().enumerate() {
                        self.pending[i % ch].push(s);
                    }
                }
                Err(SymphoniaError::DecodeError(_)) => continue, // skip a corrupt packet
                Err(e) => {
                    self.done = true;
                    return Some(Err(e));
                }
            }
        }

        let avail = self.pending_frames();
        if avail == 0 {
            return None; // fully drained
        }
        let take = avail.min(self.block_frames);
        Some(Ok(self.take_block(take)))
    }
}

/// Stream any Symphonia-decodable file as [`Signal`] blocks of `block_frames` frames (bounded
/// memory) — the streaming counterpart to [`decode`]. `block_frames` is clamped to at least 1.
pub fn decode_blocks(
    path: impl AsRef<Path>,
    block_frames: usize,
) -> Result<AudioBlocks, SymphoniaError> {
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
    let format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or(SymphoniaError::Unsupported("no decodable audio track"))?;
    let track_id = track.id;
    let fs = track.codec_params.sample_rate.unwrap_or(0);
    let decoder =
        symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;

    Ok(AudioBlocks {
        format,
        decoder,
        track_id,
        fs,
        channels: 0,
        block_frames: block_frames.max(1),
        pending: Vec::new(),
        sbuf: None,
        done: false,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        AudioInfo, WavEncoding, decode, decode_blocks, probe, probe_wav, read_wav, read_wav_blocks,
        read_wav_from, write_wav, write_wav_encoded, write_wav_to,
    };
    use fluxion_core::Signal;
    use std::io::Cursor;

    /// Per-test temp path, unique across tests and processes so the default parallel test runner
    /// can't collide on a shared file.
    fn tmp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("fluxion_io_{}_{name}.wav", std::process::id()))
    }

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

    #[test]
    fn pcm16_roundtrip_within_1_lsb() {
        // Dither OFF so quantization is deterministic and error is bounded by 0.5 LSB.
        let enc = WavEncoding {
            bits: 16,
            float: false,
            dither: false,
        };
        // Values kept off the ±1.0 clip edge so the only error is quantization.
        let original = Signal::new(
            24_000,
            vec![vec![0.0, 0.5, -0.5, 0.25, -0.25, 0.123, -0.777]],
        );
        let path = tmp_path("pcm16");
        write_wav_encoded(&path, &original, enc).expect("write");
        let read_back = read_wav(&path).expect("read");
        let _ = std::fs::remove_file(&path);

        assert_eq!(read_back.fs, 24_000);
        assert_eq!(read_back.channel_count(), 1);
        let lsb = 1.0 / 32_768.0;
        for (o, r) in original.channels[0].iter().zip(&read_back.channels[0]) {
            assert!(
                (o - r).abs() <= lsb,
                "16-bit sample {o} -> {r} exceeds 1 LSB"
            );
        }
    }

    #[test]
    fn pcm24_roundtrip_within_2_lsb() {
        let enc = WavEncoding {
            bits: 24,
            float: false,
            dither: false,
        };
        let original = Signal::new(
            48_000,
            vec![
                vec![0.0, 0.5, -0.5, 0.333_33, -0.9, 0.123_456],
                vec![0.1, -0.1, 0.875, -0.333_33, 0.2, -0.6],
            ],
        );
        let path = tmp_path("pcm24");
        write_wav_encoded(&path, &original, enc).expect("write");
        let read_back = read_wav(&path).expect("read");
        let _ = std::fs::remove_file(&path);

        assert_eq!(read_back.channel_count(), 2);
        // 0.5 LSB quantization + f32 round-trip representation error near full-scale; 2 LSB covers it.
        let tol = 2.0 / 8_388_608.0;
        for ch in 0..2 {
            for (o, r) in original.channels[ch].iter().zip(&read_back.channels[ch]) {
                assert!(
                    (o - r).abs() <= tol,
                    "24-bit sample {o} -> {r} exceeds 2 LSB"
                );
            }
        }
    }

    #[test]
    fn dither_spreads_a_constant_signal() {
        // A constant off-grid mid-scale value quantizes to a single code without dither, but TPDF
        // dither must scatter it across neighboring codes (that's the whole point — decorrelation).
        let constant = Signal::new(16_000, vec![vec![0.3; 512]]);

        let no_dither = WavEncoding {
            bits: 16,
            float: false,
            dither: false,
        };
        let with_dither = WavEncoding {
            bits: 16,
            float: false,
            dither: true,
        };

        let p_off = tmp_path("dither_off");
        let p_on = tmp_path("dither_on");
        write_wav_encoded(&p_off, &constant, no_dither).expect("write off");
        write_wav_encoded(&p_on, &constant, with_dither).expect("write on");
        let off = read_wav(&p_off).expect("read off");
        let on = read_wav(&p_on).expect("read on");
        let _ = std::fs::remove_file(&p_off);
        let _ = std::fs::remove_file(&p_on);

        let distinct = |ch: &[f32]| {
            let mut codes: Vec<i32> = ch.iter().map(|&s| (s * 32_768.0).round() as i32).collect();
            codes.sort_unstable();
            codes.dedup();
            codes.len()
        };
        assert_eq!(
            distinct(&off.channels[0]),
            1,
            "no-dither must be a single code"
        );
        assert!(
            distinct(&on.channels[0]) > 1,
            "TPDF dither must produce more than one code"
        );
    }

    #[test]
    fn stdio_memory_roundtrip() {
        // write_wav_to (stdout path) -> read_wav_from (stdin path), fully in-memory, f32 exact.
        let original = Signal::new(
            44_100,
            vec![vec![0.0, 0.5, -0.5, 1.0], vec![0.2, -0.2, 0.6, 0.0]],
        );
        let mut buf = Cursor::new(Vec::new());
        write_wav_to(&mut buf, &original).expect("write");
        let bytes = buf.into_inner();
        let read_back = read_wav_from(Cursor::new(bytes)).expect("read");

        assert_eq!(read_back.fs, original.fs);
        assert_eq!(read_back.channels, original.channels);
    }

    #[test]
    fn probe_wav_reports_header() {
        let enc = WavEncoding {
            bits: 24,
            float: false,
            dither: false,
        };
        let sig = Signal::new(32_000, vec![vec![0.0; 800], vec![0.0; 800]]);
        let path = tmp_path("probe");
        write_wav_encoded(&path, &sig, enc).expect("write");
        let info = probe_wav(&path).expect("probe");
        let _ = std::fs::remove_file(&path);

        assert_eq!(info.fs, 32_000);
        assert_eq!(info.channels, 2);
        assert_eq!(info.bits, 24);
        assert!(!info.float);
        assert_eq!(info.frames, 800);
        assert!((info.seconds() - 0.025).abs() < 1e-9);
    }

    #[test]
    fn malformed_input_is_a_clean_error() {
        // Garbage bytes must return Err, never panic, on every reader entry point.
        let garbage = [
            0xDE_u8, 0xAD, 0xBE, 0xEF, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05,
        ];
        assert!(read_wav_from(Cursor::new(garbage.to_vec())).is_err());

        let path = tmp_path("garbage");
        std::fs::write(&path, garbage).unwrap();
        assert!(read_wav(&path).is_err());
        assert!(decode(&path).is_err());
        assert!(probe(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn invalid_encoding_is_rejected() {
        let sig = Signal::new(8_000, vec![vec![0.0, 0.1]]);
        // 20-bit int and 16-bit float are unsupported; both must Err without writing a bad file.
        let bad_int = WavEncoding {
            bits: 20,
            float: false,
            dither: false,
        };
        let bad_float = WavEncoding {
            bits: 16,
            float: true,
            dither: false,
        };
        let mut sink = Cursor::new(Vec::new());
        assert!(super::write_wav_encoded_to(&mut sink, &sig, bad_int).is_err());
        assert!(super::write_wav_encoded_to(&mut sink, &sig, bad_float).is_err());
    }

    #[test]
    fn decode_flac_fixture_matches_known_content() {
        // Fixture: 1 kHz sine, 8 kHz fs, mono, 240 frames, peak amplitude 0.125, period 8 samples.
        // Generated once with ffmpeg (see tests/data), padding stripped so it's ~230 bytes.
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/sine_1k_8000_mono.flac"
        );
        let sig = decode(path).expect("decode flac");

        assert_eq!(sig.fs, 8_000);
        assert_eq!(sig.channel_count(), 1);
        assert_eq!(sig.frames(), 240);
        // sin(2*pi*1000*k/8000) = sin(pi*k/4): frame 0 is the zero crossing, frame 2 the +peak.
        assert!(sig.channels[0][0].abs() < 1e-6, "sine must start at 0");
        assert!(
            (sig.channels[0][2] - 0.125).abs() < 1e-4,
            "frame 2 must be +peak"
        );
        assert!(
            (sig.channels[0][6] + 0.125).abs() < 1e-4,
            "frame 6 must be -peak"
        );
        let peak = sig.channels[0].iter().fold(0.0_f32, |m, &s| m.max(s.abs()));
        assert!(
            (0.12..0.13).contains(&peak),
            "peak {peak} out of expected band"
        );
    }

    #[test]
    fn probe_flac_fixture_reports_metadata() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/sine_1k_8000_mono.flac"
        );
        let info: AudioInfo = probe(path).expect("probe flac");

        assert_eq!(info.fs, 8_000);
        assert_eq!(info.channels, 1);
        assert_eq!(info.codec, "flac");
        assert_eq!(info.frames, Some(240));
        assert!((info.seconds().unwrap() - 0.03).abs() < 1e-9);
    }

    #[test]
    fn wav_blocks_reassemble_the_whole_signal() {
        // Streaming in 100-frame blocks must reconstruct exactly what read_wav returns whole.
        let original = Signal::new(
            16_000,
            vec![
                (0..250).map(|i| (i as f32 / 250.0) - 0.5).collect(),
                (0..250).map(|i| 0.4 * (i as f32 * 0.05).sin()).collect(),
            ],
        );
        let path = tmp_path("wav_blocks");
        write_wav_encoded(
            &path,
            &original,
            WavEncoding {
                bits: 24,
                float: false,
                dither: false,
            },
        )
        .expect("write");

        let whole = read_wav(&path).expect("read whole");
        let mut blocks = Vec::new();
        for block in read_wav_blocks(&path, 100).expect("open blocks") {
            blocks.push(block.expect("block"));
        }
        let _ = std::fs::remove_file(&path);

        // 250 frames / 100 → blocks of 100, 100, 50.
        assert_eq!(
            blocks.iter().map(Signal::frames).collect::<Vec<_>>(),
            vec![100, 100, 50]
        );
        assert!(blocks.iter().all(|b| b.channel_count() == 2));

        let rebuilt = concat_blocks(&blocks);
        assert_eq!(rebuilt.frames(), 250);
        for ch in 0..2 {
            assert_eq!(rebuilt.channels[ch], whole.channels[ch]);
        }
    }

    #[test]
    fn block_writer_matches_whole_file_writer_bit_exact() {
        // Streamed writes (including the dither PRNG stream carried across blocks) must
        // produce byte-identical files to the whole-file writer.
        let sig = Signal::new(
            22_050,
            vec![
                (0..300).map(|i| ((i as f32) * 0.037).sin() * 0.8).collect(),
                (0..300).map(|i| ((i as f32) * 0.011).cos() * 0.6).collect(),
            ],
        );
        for enc in [WavEncoding::default(), WavEncoding::pcm(16), WavEncoding::pcm(24)] {
            let p_whole = tmp_path(&format!("whole_{}", enc.bits + enc.float as u16));
            let p_stream = tmp_path(&format!("stream_{}", enc.bits + enc.float as u16));
            write_wav_encoded(&p_whole, &sig, enc).expect("whole write");

            let mut w = super::WavBlockWriter::create(&p_stream, sig.fs, 2, enc).expect("create");
            for start in (0..300).step_by(128) {
                let end = (start + 128).min(300);
                let block: Vec<Vec<f32>> = sig
                    .channels
                    .iter()
                    .map(|ch| ch[start..end].to_vec())
                    .collect();
                w.write_block(&block).expect("block");
            }
            w.finalize().expect("finalize");

            let a = std::fs::read(&p_whole).unwrap();
            let b = std::fs::read(&p_stream).unwrap();
            let _ = std::fs::remove_file(&p_whole);
            let _ = std::fs::remove_file(&p_stream);
            assert_eq!(a, b, "bits={} float={}", enc.bits, enc.float);
        }
    }

    /// Concatenate streamed blocks back into one [`Signal`] (test helper).
    fn concat_blocks(blocks: &[Signal]) -> Signal {
        let fs = blocks.first().map(|b| b.fs).unwrap_or(0);
        let nch = blocks.first().map(Signal::channel_count).unwrap_or(0);
        let mut channels = vec![Vec::new(); nch];
        for b in blocks {
            for (c, ch) in b.channels.iter().enumerate() {
                channels[c].extend_from_slice(ch);
            }
        }
        Signal::new(fs, channels)
    }

    #[test]
    fn decode_blocks_reassemble_the_flac_fixture() {
        // The streaming decoder must yield the same 240 frames as decode(), in 64-frame blocks.
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/sine_1k_8000_mono.flac"
        );
        let whole = decode(path).expect("decode whole");

        let mut blocks = Vec::new();
        for block in decode_blocks(path, 64).expect("open blocks") {
            blocks.push(block.expect("block"));
        }
        let rebuilt = concat_blocks(&blocks);

        assert_eq!(rebuilt.fs, whole.fs);
        assert_eq!(rebuilt.frames(), whole.frames());
        assert!(
            blocks
                .iter()
                .take(blocks.len() - 1)
                .all(|b| b.frames() == 64)
        );
        for ch in 0..whole.channel_count() {
            assert_eq!(rebuilt.channels[ch], whole.channels[ch]);
        }
    }
}

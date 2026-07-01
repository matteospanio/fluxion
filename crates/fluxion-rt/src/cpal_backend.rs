//! CPAL cross-platform realtime audio backend (plan task G5, feature `cpal`).
//!
//! Opens the default output device and drives a render callback on the OS audio thread. That
//! callback must be allocation- and lock-free — feed it an [`RtEngine`](crate::RtEngine) or
//! [`RtGraph`](crate::RtGraph), whose `process` paths satisfy that (proven in `tests/rt_safety.rs`).
//! Gated behind the `cpal` feature so the default build needs no platform audio libraries.
//!
//! ```no_run
//! use fluxion_rt::{RtGraph, Biquad, cpal_backend};
//! # fn design(_: u32) -> Vec<Biquad> { vec![] }
//! let stream = cpal_backend::run_output(|buf, channels| {
//!     // Fill `buf` (interleaved f32) on the audio thread — drive your prepared RtGraph here.
//!     for frame in buf.chunks_mut(channels) {
//!         for s in frame.iter_mut() { *s = 0.0; }
//!     }
//! })
//! .expect("open output");
//! let _fs = stream.sample_rate; // design filters for this rate
//! // keep `stream` alive; drop it to stop audio.
//! ```

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

/// Failure opening or starting an audio stream.
#[derive(Debug)]
pub enum BackendError {
    /// No default output device is available.
    NoOutputDevice,
    /// No default input device is available.
    NoInputDevice,
    /// The device reported no usable default configuration.
    Config(cpal::DefaultStreamConfigError),
    /// The output stream could not be built (e.g. unsupported f32 format).
    Build(cpal::BuildStreamError),
    /// The stream could not be started.
    Play(cpal::PlayStreamError),
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendError::NoOutputDevice => write!(f, "no default output device"),
            BackendError::NoInputDevice => write!(f, "no default input device"),
            BackendError::Config(e) => write!(f, "default stream config: {e}"),
            BackendError::Build(e) => write!(f, "build stream: {e}"),
            BackendError::Play(e) => write!(f, "play stream: {e}"),
        }
    }
}

impl std::error::Error for BackendError {}

impl From<cpal::DefaultStreamConfigError> for BackendError {
    fn from(e: cpal::DefaultStreamConfigError) -> Self {
        BackendError::Config(e)
    }
}
impl From<cpal::BuildStreamError> for BackendError {
    fn from(e: cpal::BuildStreamError) -> Self {
        BackendError::Build(e)
    }
}
impl From<cpal::PlayStreamError> for BackendError {
    fn from(e: cpal::PlayStreamError) -> Self {
        BackendError::Play(e)
    }
}

/// A running output stream. Audio plays until this is dropped.
pub struct OutputStream {
    _stream: cpal::Stream,
    /// Sample rate (Hz) the device is running at — design filters for this rate.
    pub sample_rate: u32,
    /// Interleaved channel count of the render buffer.
    pub channels: usize,
}

/// Open the default output device and start streaming, invoking `render(buffer, channels)` on the
/// audio thread to fill each block of interleaved `f32` samples. Keep the returned [`OutputStream`]
/// alive; drop it to stop.
///
/// Assumes an `f32` output format (the default on modern devices). `render` must not allocate, lock,
/// or block — it runs under the audio deadline.
pub fn run_output<F>(mut render: F) -> Result<OutputStream, BackendError>
where
    F: FnMut(&mut [f32], usize) + Send + 'static,
{
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or(BackendError::NoOutputDevice)?;
    let supported = device.default_output_config()?;
    let sample_rate = supported.sample_rate().0;
    let channels = supported.channels() as usize;
    let config: cpal::StreamConfig = supported.config();

    let stream = device.build_output_stream(
        &config,
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| render(data, channels),
        |err| eprintln!("fluxion cpal stream error: {err}"),
        None,
    )?;
    stream.play()?;

    Ok(OutputStream {
        _stream: stream,
        sample_rate,
        channels,
    })
}

/// The default output device's sample rate, without opening a stream — so a file can be resampled to
/// the device rate before playback.
pub fn default_output_sample_rate() -> Result<u32, BackendError> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or(BackendError::NoOutputDevice)?;
    Ok(device.default_output_config()?.sample_rate().0)
}

/// The default input device's `(sample_rate, channels)`, without opening a stream — to size the
/// capture ring before recording.
pub fn default_input_config() -> Result<(u32, usize), BackendError> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or(BackendError::NoInputDevice)?;
    let cfg = device.default_input_config()?;
    Ok((cfg.sample_rate().0, cfg.channels() as usize))
}

/// A running input (capture) stream. Recording continues until this is dropped.
pub struct InputStream {
    _stream: cpal::Stream,
    /// Sample rate (Hz) the input device is running at.
    pub sample_rate: u32,
    /// Interleaved channel count of the capture buffer.
    pub channels: usize,
}

/// Open the default input device and start capturing, invoking `capture(buffer, channels)` on the
/// audio thread with each block of interleaved `f32` samples. Keep the returned [`InputStream`]
/// alive; drop it to stop. Like [`run_output`], `capture` runs under the audio deadline — it must not
/// allocate, lock, or block (push samples through a lock-free ring; see [`crate::ring`]).
///
/// Assumes an `f32` input format (the default on modern devices).
pub fn run_input<F>(mut capture: F) -> Result<InputStream, BackendError>
where
    F: FnMut(&[f32], usize) + Send + 'static,
{
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or(BackendError::NoInputDevice)?;
    let supported = device.default_input_config()?;
    let sample_rate = supported.sample_rate().0;
    let channels = supported.channels() as usize;
    let config: cpal::StreamConfig = supported.config();

    let stream = device.build_input_stream(
        &config,
        move |data: &[f32], _: &cpal::InputCallbackInfo| capture(data, channels),
        |err| eprintln!("fluxion cpal input error: {err}"),
        None,
    )?;
    stream.play()?;

    Ok(InputStream {
        _stream: stream,
        sample_rate,
        channels,
    })
}

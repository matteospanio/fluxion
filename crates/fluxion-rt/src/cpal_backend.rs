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
            BackendError::Config(e) => write!(f, "default stream config: {e}"),
            BackendError::Build(e) => write!(f, "build output stream: {e}"),
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

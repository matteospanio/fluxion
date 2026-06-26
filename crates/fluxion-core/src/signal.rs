//! The multichannel audio buffer that flows through the graph.

/// A block of audio in **planar, channel-first** layout: `channels[c]` holds every sample for
/// channel `c`, all channels the same length. This is the data container the graph operates on
/// (the analogue of torchfx's `Wave`), carrying its sample rate so filters can design coefficients.
#[derive(Clone, Debug, PartialEq)]
pub struct Signal {
    /// Sample rate in Hz.
    pub fs: u32,
    /// One sample vector per channel, normalized to roughly `[-1.0, 1.0]`.
    pub channels: Vec<Vec<f32>>,
}

impl Signal {
    /// Build a signal from channels and a sample rate.
    pub fn new(fs: u32, channels: Vec<Vec<f32>>) -> Self {
        Self { fs, channels }
    }

    /// Number of channels.
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    /// Number of frames (samples per channel); `0` if there are no channels.
    pub fn frames(&self) -> usize {
        self.channels.first().map_or(0, Vec::len)
    }
}

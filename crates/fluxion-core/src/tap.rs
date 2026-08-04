//! Observer taps — analysis that reads the chain and never touches it (ROADMAP A1).
//!
//! A host draws a spectrum while audio plays, and a script wants the peak of a stage without
//! running the whole thing twice. A tap sits in the chain like an op, measures whatever flows
//! through, and passes it on **unchanged**.
//!
//! Unchanged is structural rather than promised. A tap is not an op: it is a [`Graph::Tap`] node,
//! the executor hands the buffer to the backend to *borrow* for measurement, and the buffer that
//! carries on is the one that arrived — there is no code path by which a tap could return anything
//! else. The check `taps_do_not_touch_the_audio` compares a chain with taps to the same chain
//! without them, bit for bit, but the design is what makes it true.
//!
//! [`Graph::Tap`]: crate::Graph::Tap

use serde::{Deserialize, Serialize};

/// What a tap measures.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum TapKind {
    /// A windowed FFT for an analyser view (ROADMAP A2).
    Spectrum {
        /// FFT size in samples; rounded up to a power of two.
        size: usize,
        /// Fraction of a window each step advances by — 0.5 is 50 % overlap. Clamped to `0..0.95`.
        overlap: f32,
    },
    /// Peak, RMS and short-term loudness (ROADMAP A3).
    Meter,
}

impl TapKind {
    /// The name this tap is written with in chain text.
    pub fn name(self) -> &'static str {
        match self {
            TapKind::Spectrum { .. } => "spectrum",
            TapKind::Meter => "meter",
        }
    }
}

/// What one tap saw.
#[derive(Clone, Debug, PartialEq)]
pub struct TapReading {
    /// The label the tap was given (`analyser: spectrum(1024)`), or `None` if it had none.
    ///
    /// Readings come back in the order the chain reaches them, so an unlabelled tap is still
    /// identifiable by position; a label is for when position is not a nice way to say it.
    pub label: Option<String>,
    /// The measurement.
    pub data: TapData,
}

/// The measurement itself.
#[derive(Clone, Debug, PartialEq)]
pub enum TapData {
    /// A magnitude spectrum, averaged over every window the signal was long enough for.
    Spectrum {
        /// Hz per bin: bin `i` is centred at `i * bin_hz`.
        bin_hz: f32,
        /// Mean magnitude per bin, `size/2 + 1` entries (DC to Nyquist), linear.
        magnitude: Vec<f32>,
    },
    /// What a meter bridge shows. All three in decibels, because a meter is a decibel instrument
    /// and mixing units inside one reading is how a caller ends up drawing a linear number on a dB
    /// scale. Silence reads `-inf` in all three rather than 0, which on a dB scale is full scale.
    Meter {
        /// Largest absolute sample, dBFS.
        peak_db: f32,
        /// Root mean square over the whole signal, dBFS.
        rms_db: f32,
        /// The loudest 3-second window, LUFS (ITU-R BS.1770). `-inf` for material shorter than one
        /// window — a number that cannot be measured is not reported as a number.
        short_term_lufs: f32,
    },
}

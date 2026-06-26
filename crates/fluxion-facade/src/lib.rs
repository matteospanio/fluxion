//! `fluxion` — differentiable, cross-vendor, framework-agnostic audio DSP with a functional
//! graph API.
//!
//! This is the **facade** crate: it re-exports the public surface and a [`prelude`] of ergonomic
//! node constructors. See `PROJECT.md` for the design and `IMPLEMENTATION_PLAN.md` for the roadmap.
//!
//! ```
//! use fluxion::prelude::*;
//!
//! let chain = lowpass(800.0) | gain(0.5);        // `|` = series
//! let eq    = lowpass(800.0) + highpass(80.0);   // `+` = parallel (summed)
//! assert_eq!(chain.leaf_count(), 2);
//! assert_eq!(eq.leaf_count(), 2);
//! ```

pub use fluxion_backend::{
    Certificate, Verdict, certify_graph, graph_to_sos, process, process_batch, sos_filter_batch,
};
pub use fluxion_core::{Graph, Op, OpError, OpKind, ParamSpec, Signal, Unit, fxg};

/// Ergonomic node constructors plus the core types.
pub mod prelude {
    pub use fluxion_backend::process;
    pub use fluxion_core::{Graph, Op, OpKind, Signal};

    /// A linear gain node (`y = x * g`).
    pub fn gain(g: f32) -> Graph {
        Graph::op(OpKind::Gain, [g])
    }

    /// A 2nd-order Butterworth low-pass with the given cutoff in Hz.
    pub fn lowpass(cutoff_hz: f32) -> Graph {
        lowpass_n(cutoff_hz, 2)
    }

    /// A Butterworth low-pass of the given cutoff (Hz) and order.
    pub fn lowpass_n(cutoff_hz: f32, order: u32) -> Graph {
        Graph::op(OpKind::Lowpass, [cutoff_hz, order as f32])
    }

    /// A 2nd-order Butterworth high-pass with the given cutoff in Hz.
    pub fn highpass(cutoff_hz: f32) -> Graph {
        highpass_n(cutoff_hz, 2)
    }

    /// A Butterworth high-pass of the given cutoff (Hz) and order.
    pub fn highpass_n(cutoff_hz: f32, order: u32) -> Graph {
        Graph::op(OpKind::Highpass, [cutoff_hz, order as f32])
    }

    /// Peak-normalize to a target linear amplitude.
    pub fn normalize(peak: f32) -> Graph {
        Graph::op(OpKind::Normalize, [peak])
    }

    /// RBJ peaking EQ: `gain` dB around `frequency` Hz with bandwidth `q`.
    pub fn peaking(frequency: f32, gain_db: f32, q: f32) -> Graph {
        Graph::op(OpKind::Peaking, [frequency, gain_db, q])
    }

    /// RBJ low shelf: `gain` dB below `frequency` Hz (bandwidth `q`).
    pub fn lowshelf(frequency: f32, gain_db: f32, q: f32) -> Graph {
        Graph::op(OpKind::LowShelf, [frequency, gain_db, q])
    }

    /// RBJ high shelf: `gain` dB above `frequency` Hz (bandwidth `q`).
    pub fn highshelf(frequency: f32, gain_db: f32, q: f32) -> Graph {
        Graph::op(OpKind::HighShelf, [frequency, gain_db, q])
    }

    /// RBJ notch at `frequency` Hz with bandwidth `q`.
    pub fn notch(frequency: f32, q: f32) -> Graph {
        Graph::op(OpKind::Notch, [frequency, q])
    }

    /// RBJ band-pass (0 dB peak) at `frequency` Hz with bandwidth `q`.
    pub fn bandpass(frequency: f32, q: f32) -> Graph {
        Graph::op(OpKind::Bandpass, [frequency, q])
    }

    /// RBJ all-pass at `frequency` Hz with bandwidth `q`.
    pub fn allpass(frequency: f32, q: f32) -> Graph {
        Graph::op(OpKind::Allpass, [frequency, q])
    }

    /// Single delayed tap: `time` seconds, crossfaded by `mix` (0 = dry, 1 = fully delayed).
    pub fn delay(time: f32, mix: f32) -> Graph {
        Graph::op(OpKind::Delay, [time, mix])
    }

    /// Feedback echo: `time` seconds between repeats, `feedback` decay, `wet` echo level.
    pub fn echo(time: f32, feedback: f32, wet: f32) -> Graph {
        Graph::op(OpKind::Echo, [time, feedback, wet])
    }

    /// Chebyshev Type I low-pass: `cutoff` Hz, `order`, passband `ripple` dB.
    pub fn cheby1_lowpass(cutoff_hz: f32, order: u32, ripple_db: f32) -> Graph {
        Graph::op(OpKind::Cheby1Lowpass, [cutoff_hz, order as f32, ripple_db])
    }

    /// Chebyshev Type I high-pass: `cutoff` Hz, `order`, passband `ripple` dB.
    pub fn cheby1_highpass(cutoff_hz: f32, order: u32, ripple_db: f32) -> Graph {
        Graph::op(OpKind::Cheby1Highpass, [cutoff_hz, order as f32, ripple_db])
    }
}

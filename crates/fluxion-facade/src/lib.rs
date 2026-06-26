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

pub use fluxion_backend::process;
pub use fluxion_core::{Graph, Op, OpError, OpKind, ParamSpec, Signal, Unit};

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
}

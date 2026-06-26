//! `fluxion` — differentiable, cross-vendor, framework-agnostic audio DSP with a functional
//! graph API.
//!
//! This is the **facade** crate: it re-exports the public surface and a [`prelude`] of
//! ergonomic node constructors. See `PROJECT.md` at the repository root for the full design,
//! rationale, and roadmap. As the workspace fills in, this crate will also re-export the ops,
//! backend selection, autodiff adapters, and IO.
//!
//! ```
//! use fluxion::prelude::*;
//!
//! let chain = lowpass(800.0) | gain(-3.0);       // `|` = series
//! let eq    = lowpass(800.0) + highpass(80.0);   // `+` = parallel (summed)
//! assert_eq!(chain.leaf_count(), 2);
//! assert_eq!(eq.leaf_count(), 2);
//! ```

pub use fluxion_core::Graph;

/// Ergonomic node constructors plus the core [`Graph`] type.
pub mod prelude {
    pub use fluxion_core::Graph;

    /// A linear gain node (`y = x * g`).
    pub fn gain(g: f32) -> Graph {
        Graph::op("gain", [g])
    }

    /// A low-pass filter node with the given cutoff in Hz.
    pub fn lowpass(cutoff_hz: f32) -> Graph {
        Graph::op("lowpass", [cutoff_hz])
    }

    /// A high-pass filter node with the given cutoff in Hz.
    pub fn highpass(cutoff_hz: f32) -> Graph {
        Graph::op("highpass", [cutoff_hz])
    }
}

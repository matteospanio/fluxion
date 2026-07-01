//! `fluxion-rt` — the hard-real-time execution engine.
//!
//! Takes a graph plus its frozen (design-time) coefficients and runs them through a pre-allocated,
//! lock-free, allocation-free SIMD loop on a high-priority audio thread, fed by an SPSC ring
//! buffer. Inference-only and chunk-length-preserving by contract; never runs autograd or GPU
//! dispatch inside the audio callback (see `PROJECT.md` §5). Parameter automation arrives via a
//! lock-free command queue applied at block boundaries with ramping.
//!
//! So far:
//! - [`ring`] — lock-free SPSC ring buffer (G1).
//! - [`stream`] — allocation-free streaming SOS cascade, streaming == batch (G3 core).
//! - [`param`] — click-free [`SmoothedValue`] parameter ramping (G4).
//! - [`engine`] — [`RtEngine`]: cascade + smoothed gain + lock-free command queue (G3 + G4).
//! - [`graph`] — [`RtGraph`]: general series/parallel/filter/gain/delay/echo executor, alloc-free
//!   after `prepare` (G3); lower a `fluxion_core::Graph` to it via `fluxion_backend::to_rt_graph`.
//! - `cpal_backend` — CPAL output stream driving a render callback (G5, feature `cpal`).
//!
//! A frozen cascade plan comes from `fluxion-backend::freeze` (G2). Real-time-safety is enforced by
//! `tests/rt_safety.rs` (G6: no-alloc-in-callback + xrun stress). Next: duplex (live-input) CPAL,
//! fractional/interpolated delay, and the SIMD MAC loop.

#[cfg(feature = "cpal")]
pub mod cpal_backend;
pub mod engine;
pub mod graph;
pub mod param;
pub mod ring;
pub mod stream;

pub use engine::{Command, RtEngine};
pub use graph::{MAX_SETCOEFFS_SECTIONS, RtGraph, SetCoeffs};
pub use param::SmoothedValue;
pub use ring::{Consumer, Producer, channel};
pub use stream::SosStream;

// Re-exported so callers can build cascades (`RtEngine::new`, `SosStream::new`) without also
// depending on `fluxion-ops` directly.
pub use fluxion_ops::Biquad;

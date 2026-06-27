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
//! - [`graph`] — [`RtGraph`]: general series/parallel graph executor, alloc-free after `prepare` (G3).
//! - [`cpal_backend`] — CPAL output stream driving a render callback (G5, feature `cpal`).
//!
//! A frozen cascade plan comes from `fluxion-backend::freeze` (G2). Real-time-safety is enforced by
//! `tests/rt_safety.rs` (G6: no-alloc-in-callback + xrun stress). Next: lower an arbitrary
//! `fluxion_core::Graph` to an [`RtGraph`] (in `fluxion-backend`), realtime delay/echo nodes, and
//! duplex (live-input) CPAL.

#[cfg(feature = "cpal")]
pub mod cpal_backend;
pub mod engine;
pub mod graph;
pub mod param;
pub mod ring;
pub mod stream;

pub use engine::{Command, RtEngine};
pub use graph::RtGraph;
pub use param::SmoothedValue;
pub use ring::{Consumer, Producer, channel};
pub use stream::SosStream;

// Re-exported so callers can build cascades (`RtEngine::new`, `SosStream::new`) without also
// depending on `fluxion-ops` directly.
pub use fluxion_ops::Biquad;

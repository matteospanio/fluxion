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
//!
//! A frozen cascade plan comes from `fluxion-backend::freeze` (G2); build a stream from it with
//! [`SosStream::from_sections`]. Next: a CPAL audio backend (G5) and real-time-safety stress (G6).

pub mod param;
pub mod ring;
pub mod stream;

pub use param::SmoothedValue;
pub use ring::{Consumer, Producer, channel};
pub use stream::SosStream;

//! `fluxion-rt` — the hard-real-time execution engine.
//!
//! Takes a graph plus its frozen (design-time) coefficients and runs them through a pre-allocated,
//! lock-free, allocation-free SIMD loop on a high-priority audio thread, fed by an SPSC ring
//! buffer. Inference-only and chunk-length-preserving by contract; never runs autograd or GPU
//! dispatch inside the audio callback (see `PROJECT.md` §5). Parameter automation arrives via a
//! lock-free command queue applied at block boundaries with ramping.
//!
//! Empty scaffold for now.

//! `fluxion-ops` — DSP primitives: coefficient design and forward kernels.
//!
//! Today this crate provides Butterworth SOS design ([`butterworth_lowpass`] /
//! [`butterworth_highpass`]), the second-order-section cascade filter ([`sos_filter`]), a
//! frequency-response helper ([`sos_magnitude`]), and simple effects ([`gain`], [`normalize_peak`]).
//! Hand-derived analytic backward passes (VJPs) and the FIR/FFT-conv, delay, and reverb ops land in
//! later milestones (see `IMPLEMENTATION_PLAN.md` epics D/E).
//!
//! Kernels operate on plain `&[f32]` / `&mut [f32]` channels; the graph executor in
//! `fluxion-backend` applies them across a multichannel signal.

pub mod effect;
pub mod iir;

pub use effect::{gain, normalize_peak};
pub use iir::{Biquad, Sos, butterworth_highpass, butterworth_lowpass, sos_filter, sos_magnitude};

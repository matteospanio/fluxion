//! `fluxion-ops` — DSP primitives: coefficient design, forward kernels, and analytic backward (VJP).
//!
//! Today this crate provides Butterworth SOS design ([`butterworth_lowpass`] /
//! [`butterworth_highpass`]), the second-order-section cascade filter ([`sos_filter`]), a
//! frequency-response helper ([`sos_magnitude`]), a stability check ([`sos_is_stable`]), simple
//! effects ([`gain`], [`normalize_peak`]), and the **analytic backward passes** that make these ops
//! differentiable: [`biquad_vjp`] / [`sos_vjp`] for IIR filters and [`gain_vjp`] for gain.
//!
//! Owning these backward passes is the project's durable asset (see `PROJECT.md` §2, §4.3): the
//! framework integration crates (`fluxion-autodiff`, `fluxion-py`) *rent* a graph from Burn / torch
//! / jax and register these same forward+backward kernels. FIR/FFT-conv, delay, and reverb ops and
//! their VJPs land in later milestones.
//!
//! Kernels operate on plain `&[f32]` / `&mut [f32]` channels; the graph executor in
//! `fluxion-backend` applies them across a multichannel signal.

pub mod chebyshev;
pub mod delay;
pub mod effect;
pub mod fir;
pub mod iir;
pub mod rbj;
pub mod reverb;
pub mod stability;

pub use chebyshev::{chebyshev1_highpass, chebyshev1_lowpass};
pub use delay::{delay, delay_vjp, echo, echo_vjp};
pub use effect::{gain, gain_vjp, normalize_peak};
pub use fir::{fft_convolve, fir_filter, fir_vjp};
pub use iir::{
    Biquad, BiquadGrad, Sos, biquad_forward, biquad_vjp, butterworth_highpass, butterworth_lowpass,
    sos_filter, sos_filter_interleaved, sos_input_grad, sos_is_stable, sos_magnitude, sos_vjp,
};
pub use rbj::{allpass, bandpass, high_shelf, low_shelf, notch, peaking};
pub use reverb::reverb;
pub use stability::{Certificate, Verdict, certify_biquad, certify_sos, small_gain_certify};

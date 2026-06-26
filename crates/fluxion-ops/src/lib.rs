//! `fluxion-ops` — DSP primitives with hand-derived analytic forward **and** backward (VJP).
//!
//! Planned primitives, each as a `{ forward, backward }` pair over a backend tensor: SOS/biquad
//! cascade (with the analytic all-pole backward, not autograd-unrolling), FIR / FFT-conv,
//! fractional delay line, gain, reverb, echo, normalize, and masking. Owning these backward
//! passes is the project's durable asset (see `PROJECT.md` §2, §4.3).
//!
//! Empty scaffold for now.

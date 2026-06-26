//! `fluxion-autodiff` — make the graph trainable without owning an autograd engine.
//!
//! "Own the backward, rent the graph" (see `PROJECT.md` §2): the analytic VJPs now live in
//! `fluxion-ops` ([`fluxion_ops::biquad_vjp`], [`fluxion_ops::sos_vjp`], [`fluxion_ops::gain_vjp`])
//! and are verified by finite-difference gradcheck. This crate will register them with whatever
//! autograd is present — Burn's `Autodiff<B>` backend decorator for the Rust-native path, and
//! `torch.autograd.Function` / `jax.custom_vjp` adapters (in `fluxion-py`) when a host framework
//! drives the library.
//!
//! Not yet implemented: the Burn integration (plan task E6) pulls in the pre-1.0 Burn + CubeCL
//! stack and is the next milestone step; see `IMPLEMENTATION_PLAN.md` epics E/F.

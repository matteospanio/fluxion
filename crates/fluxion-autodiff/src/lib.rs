//! `fluxion-autodiff` — make the graph trainable without owning an autograd engine.
//!
//! "Own the backward, rent the graph" (see `PROJECT.md` §2): the analytic VJPs in `fluxion-ops`
//! are registered with whatever autograd is present — Burn's `Autodiff<B>` backend decorator for
//! the Rust-native path, and `torch.autograd.Function` / `jax.custom_vjp` adapters (in
//! `fluxion-py`) when a host framework drives the library.
//!
//! Empty scaffold for now.

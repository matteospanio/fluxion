//! `fluxion-autodiff` — make the graph trainable without owning an autograd engine.
//!
//! "Own the backward, rent the graph" (see `PROJECT.md` §2): the analytic VJPs live in `fluxion-ops`
//! ([`fluxion_ops::biquad_vjp`], [`fluxion_ops::sos_vjp`], [`fluxion_ops::gain_vjp`]), gradcheck-
//! verified. This crate registers them with a host framework's autograd. The Burn integration
//! ([`burn_backend`], feature `burn`) wraps a biquad as a Burn custom op whose backward is the
//! analytic adjoint — differentiable on `Autodiff<NdArray>` (CPU) or `Autodiff<Cuda>` (GPU) without
//! unrolling the recursion. `torch.autograd.Function` / `jax.custom_vjp` adapters (in `fluxion-py`)
//! follow the same pattern.

#[cfg(feature = "burn")]
pub mod burn_backend;

/// Whole-graph differentiation (feature `burn`): [`graph::diff_process`] lowers a `fluxion_core::Graph`
/// onto Burn's autograd via a `fluxion_backend::Backend` impl, so an entire effect chain is
/// differentiable end-to-end through the shared graph walk (plan tasks E12 + C1).
#[cfg(feature = "burn")]
pub mod graph;

/// GPU-resident differentiable ops (feature `cuda`): the same analytic VJPs as [`burn_backend`], but
/// forward and backward launch CubeCL kernels directly on a resident Burn tensor — no host roundtrip,
/// so a training loop stays on the device. The host-roundtrip ops stay the backend-agnostic fallback.
#[cfg(feature = "cuda")]
pub mod cuda;

//! `fluxion-py` — Python bindings (PyO3 + maturin).
//!
//! Exposes the graph algebra to Python as a torchaudio-style transform and a differentiable op:
//! zero-copy tensor handoff via DLPack, with `torch.autograd.Function` and `jax.custom_vjp`
//! adapters wrapping the owned forward/backward so gradients flow into a host framework's graph
//! (see `PROJECT.md` §4.3, §8.3). The Python-facing layer conforms to the Array API.
//!
//! Empty scaffold for now.

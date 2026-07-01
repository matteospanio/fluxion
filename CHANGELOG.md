<!--
Release checklist (maintainers), before tagging vX.Y.Z:
  1. `cargo test --workspace` + `cargo test -p fluxion-autodiff --features burn` green.
  2. `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo deny check`.
  3. `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --lib`.
  4. Python wheel builds + `pytest` passes (CPU and, on a CUDA host, the GPU wheel).
  5. Move the `[Unreleased]` entries below under a new `## [X.Y.Z] - DATE` heading; bump the workspace
     and `fluxion-py` versions; update the link references at the bottom.
  6. Tag `vX.Y.Z`; publish crates.io (dependency order) + PyPI (needs `PYPI_TOKEN` / `CARGO_REGISTRY_TOKEN`).
-->

# Changelog

All notable changes to fluxion are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); from 1.0.0 the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Filters & effects** — Butterworth and Chebyshev I/II low/high-pass; RBJ biquads (peaking,
  low/high shelf, notch, band-pass, all-pass); FIR (plus FFT convolution); gain, normalize, delay,
  echo, and Schroeder–Moorer reverb — all as composable graph ops, designed from closed forms with no
  SciPy at runtime.
- **Functional graph algebra** — `|` (series) and `+` (parallel) composition, node identity
  (`Graph::Named`, addressable by name), and the `~` feedback operator (`Graph::feedback`).
- **Differentiable DSP** — hand-derived analytic VJPs for every op; whole-graph reverse-mode autodiff
  through Burn (`fluxion::diff_process`); trainable filter coefficients and *design parameters*
  ("learn a cutoff") and FIR taps; an in-loop Jury-triangle stability projection; and torch
  (`SosModule`, `torch.autograd.Function`) + `jax.custom_vjp` adapters.
- **GPU** — CubeCL SOS forward + backward kernels (validated on CUDA) and a split CPU/GPU Python wheel.
- **Real-time engine** — allocation-free, lock-free block executor (SOS cascade, general
  series/parallel graph, reverb, FIR, delay/echo, fractional delay); click-free parameter automation
  with an equal-power coefficient crossfade; a lock-free SPSC command queue; and a CPAL audio backend.
- **CLI (`fluxion`)** — SoX-style `in.wav <effect …> out.wav`; `info`/`soxi`, `compile` (→ `.fxg`),
  `batch`, and stdin/stdout (`-`) / null-sink (`-n`); realtime `play`/`record` (feature `realtime`).
- **Python bindings** — torchaudio-style eager `Chain` API accepting 1-D `(T,)` and 2-D `(C, T)`
  input, zero-copy DLPack interop with NumPy / PyTorch / JAX, and Array-API consumer conformance.
- **Serialization** — versioned `.fxg` graph and `FrozenSos` plan envelopes
  (`{version, kind, fs, payload}`), rejecting incompatible/old files with a clear error.
- **Stability certification** — a pole-based + small-gain verdict ladder over a graph's frozen
  coefficients, gating `.fxg` export / realtime freeze.

### Notes

- Pre-1.0: the public Rust/Python API and the `.fxg` on-disk format are not yet stable.

[Unreleased]: https://github.com/matteospanio/fluxion/commits/main

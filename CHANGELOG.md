<!--
Release checklist (maintainers), before tagging vX.Y.Z:
  1. `cargo test --workspace` + `cargo test -p fluxion-autodiff --features burn` green.
  2. `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo deny check`.
  3. `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --lib`.
  4. Python wheel builds + `pytest` passes (CPU and, on a CUDA host, the GPU wheel).
  5. Move the `[Unreleased]` entries below under a new `## [X.Y.Z] - DATE` heading; bump the workspace
     and `fluxion-py` versions; update the link references at the bottom.
  6. Tag `vX.Y.Z`; publish crates.io in dependency order (needs `CARGO_REGISTRY_TOKEN`); PyPI publishes
     automatically from the tag via the wheels workflow (Trusted Publishing — register the GitHub
     `pypi` environment on PyPI once).
-->

# Changelog

All notable changes to fluxion are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); from 1.0.0 the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **CPU batch kernel: set-spreading tile** — `sos_filter_batch`'s AVX2 8-row path now
  bounces each time block through a small padded-pitch scratch instead of loading the
  planar rows at their raw stride. Power-of-two row strides (e.g. the 64×524k paper
  workload's 2 MB) map all sixteen row streams to a single L1/L2 set on 8-way parts
  and cap the kernel at LLC speed; the tile restores cache-resident loads for two
  L2-resident copies. Measured on the i9-10900KF: 1.66 → 2.60 Gsamples/s multi-thread
  (+57%, past TorchFX's OpenMP kernel at 1.89) for ≈5% single-thread; per-row outputs
  stay bit-identical (asserted across tile boundaries and scalar tails).

### Added

- **Checkpoint import (goal #6 / J13, full slice)** — run DDSP filters trained in other
  frameworks: `fluxion import ckpt.safetensors model.fxg` (CLI) and
  `fluxion.interop.import_checkpoint(...)` (Python; also parses `.pt` and `.onnx` and torchfx
  compiled artifacts) replay the exact param→coefficient math of FLAMO
  (`SOSFilter`/`SVF` all filter types/`Biquad`, realised `b`/`a`, RBJ tables) and torchfx.ddsp
  (learnable lowpass/highpass/peaking/parametric-EQ) into raw `biquad` sections, **certify** them
  on the stability ladder (E8; `--project-stable` Jury-clamps unstable checkpoints), and write a
  standard `.fxg` that splices into any pipeline, plays realtime, and hot-swaps. Rust converter in
  `fluxion-io::checkpoint` (feature `checkpoint`, pure-Rust `safetensors` reader); golden-tested
  against 15 real FLAMO/torchfx checkpoints. SISO only; MIMO banks and FIR taps are rejected with
  clear errors.
- **Filters & effects** — Butterworth and Chebyshev I/II low/high-pass; RBJ biquads (peaking,
  low/high shelf, notch, band-pass, all-pass) and a raw-coefficient `biquad`; FIR (plus FFT
  convolution); gain, normalize, delay (integer + fractional), echo, and Schroeder–Moorer reverb;
  plus a SoX-parity effect batch — `fade`, `tremolo`, `overdrive`, `compand` (feed-forward
  compressor, realtime-playable), `reverse`, and the modulated `chorus` / `flanger` / `phaser` — all
  as composable graph ops, designed from closed forms with no SciPy at runtime.
- **Geometry transforms** — whole-`Signal` verbs that change frame/channel count or sample rate
  (deliberately outside the graph algebra): `trim`, `pad`, `repeat`, `silence_trim`, a real
  windowed-sinc `resample` (the SoX `rate` replacement, anti-aliased) and `speed`, `remix` /
  `channels` (energy-preserving), and the `concat` / `mix` multi-input primitives.
- **Functional graph algebra** — `|` (series) and `+` (parallel) composition, node identity
  (`Graph::Named`, addressable by name), and the `~` feedback operator (`Graph::feedback`).
- **Differentiable DSP** — hand-derived analytic VJPs for every op; whole-graph reverse-mode autodiff
  through Burn (`fluxion::diff_process`); trainable filter coefficients and *design parameters*
  ("learn a cutoff") and FIR taps; an in-loop Jury-triangle stability projection; and torch
  (`SosModule`, `torch.autograd.Function`) + `jax.custom_vjp` adapters.
- **GPU** — CubeCL SOS forward + backward kernels (validated on CUDA) and a split CPU/GPU Python wheel.
- **Real-time engine** — allocation-free, lock-free block executor (SOS cascade, general
  series/parallel graph, reverb, FIR, delay/echo, fractional delay, compand); click-free parameter
  automation with an equal-power coefficient crossfade; a lock-free SPSC command queue; and a CPAL
  audio backend. Reachable from the `fluxion` facade via its `realtime` feature (re-exporting
  `RtGraph` / `RtEngine` / `SosStream` / `SmoothedValue`, `freeze` / `to_rt_graph`, and `FrozenSos`).
- **CLI (`fluxion`)** — a SoX substitute with named effects and long flags: a stage pipeline mixing
  filter passes with geometry stages (`trim`, `pad`, `rate`, `speed`, `repeat`, `silence`,
  `channels`, `remix`); multi-input concatenation and `--mix`; `--db` and SI-suffix (`1k`) parsing;
  output encoding control (`--bits 16|24|32`, `--float`, `--no-dither`); verbs `info`/`soxi` (all
  formats via Symphonia probe), `stat`, `effects` (self-describing op catalog), `synth`, `compile`
  (→ `.fxg`), `batch`, stdin/stdout (`-`) / null-sink (`-n`); realtime `play`/`record`
  (feature `realtime`).
- **Audio IO** — WAV read/write via hound with output encoding options (16/24/32-bit integer PCM
  with TPDF dither on by default, or 32-bit float) and decode + header-only `probe` of
  FLAC/MP3/OGG/AAC/… via Symphonia (pure Rust, no libsndfile/ffmpeg). Bounded-memory streaming
  readers (`read_wav_blocks` / `decode_blocks`) yield fixed-size `Signal` chunks for large files,
  and columnar dataset IO (`Signal` ↔ Arrow `RecordBatch` ↔ Parquet) sits behind an optional
  `parquet` feature for the augmentation workflow.
- **Python bindings** — torchaudio-style eager `Chain` API accepting 1-D `(T,)` and 2-D `(C, T)`
  input plus a batched `Chain.process_batch((B, T))`, zero-copy DLPack interop with NumPy /
  PyTorch / JAX, Array-API consumer conformance, `fluxion.augment` (`Compose`, `RandomChain`)
  for stochastic data augmentation, `fluxion.dataset` (Parquet audio-dataset IO — the same schema
  as the Rust side, streaming both ways; extra `fluxion[dataset]`), and
  `fluxion.interop.load_flamo_sos` for importing FLAMO-style SISO biquad checkpoints
  (`safetensors`).
- **C ABI (`fluxion-ffi`)** — a minimal panic-safe C surface (`fx_graph_load_fxg`, `fx_process`
  interleaved in-place, `fx_last_error`) with a checked-in `include/fluxion.h` and a C smoke test.
- **Quality gates** — SciPy/RBJ golden-vector oracle tests pinning every filter design's impulse
  response (32 cases, no runtime SciPy); Criterion benchmarks (`cargo bench`); CI jobs for
  benches, the C ABI, and a CUDA compile check; PyPI wheels for Linux x86_64/aarch64,
  macOS Intel/Apple-Silicon, and Windows, published on tag via Trusted Publishing.
- **Serialization** — versioned `.fxg` graph and `FrozenSos` plan envelopes
  (`{version, kind, fs, payload}`), rejecting incompatible/old files with a clear error.
- **Stability certification** — a pole-based + small-gain verdict ladder over a graph's frozen
  coefficients, gating `.fxg` export / realtime freeze.

### Notes

- Pre-1.0: the public Rust/Python API and the `.fxg` on-disk format are not yet stable.

[Unreleased]: https://github.com/matteospanio/fluxion/commits/main

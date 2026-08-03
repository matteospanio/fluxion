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

- **Breaking: the Python API is now torchfx-shaped (Epic I / I3)** — `fluxion.Wave` carries the
  sample rate, so `fs` leaves user code entirely: `wave | fx.filter.Highpass(80, order=4) |
  fx.effect.Gain(fx.db(-3))`, then `.save()`. `|` is series and `+` is parallel, the same algebra
  as the Rust library and the CLI. Piping is deferred — `w | a | b | c` accumulates one chain and
  runs it in a single fused pass rather than three. `Wave.from_file` / `.save` go through
  `fluxion-io`, so the wheel gained file I/O with no new Python dependency. Arrays still work
  everywhere a `Wave` does, and a chain is now directly callable: `chain(x, fs=48_000)`.
  The eighteen hand-written per-op functions are gone. Every op is a class in `fluxion.filter` or
  `fluxion.effect`, **generated** from the registry along with its typed stub and its docstring —
  coverage goes from 18 of 27 ops to all 27, and `fade`, `tremolo`, `overdrive`, `compand`,
  `reverse`, `biquad`, `chorus`, `flanger` and `phaser` reach Python for the first time. Names and
  parameter names are the registry's, so `low_shelf`/`high_shelf` become `LowShelf`/`HighShelf`,
  `delay(seconds=…)` becomes `Delay(time=…)` and `gain(value=…)` becomes `Gain(gain=…)`.
  `fluxion.chain("highpass(80, 4) | gain(-3dB)")` parses the shared text form and `str(chain)`
  prints it back. Errors name the class, the parameter and its range, and suggest a fix on a typo.
  A conformance test compares `fluxion.filter` / `fluxion.effect` against `fluxion.ops_table()` in
  both directions, so neither a missing class nor a stale generated file can survive CI. New
  helper: `fluxion.db(-3)` for the linear ratio the gain-like ops take. CI now builds the wheel
  and runs the ten-line quickstart from a *fresh* virtualenv on Linux, macOS and Windows on every
  pull request — previously the Python job was Linux-only and wheels were built on tags alone,
  so "pip install works" was never actually tested before a release.
- **Breaking: one op name on every interface (Epic I / I2)** — the four Chebyshev ops were spelled
  `cheby1low` / `cheby1high` / `cheby2low` / `cheby2high` in the CLI and chain text but
  `cheby1_lowpass` / … in Rust and Python. The registry name is now the long form everywhere, which
  is also what CONTRIBUTING's naming rule asks for. `.fxg` files are unaffected — they key on the
  Rust variant name, not this one. In the Rust prelude, `lowpass`/`highpass` now take
  `(cutoff, order)` like every other builder, and the `lowpass_n`/`highpass_n` synonyms are gone;
  the three ops the prelude was silently missing (`reverb`, `cheby2_lowpass`, `cheby2_highpass`)
  are added. A conformance test in each of `fluxion` and `fluxion-cli` now fails the build if the
  prelude or the `effects` listing drifts from the registry, which is how those three gaps were
  found.
- **`Graph`'s text rendering is now canonical (Epic I / I2)** — `Display` used to print
  `Series(Series(a,b),c)` and `Series(a,Series(b,c))` identically, and a `Named` node wrapping a
  composite lost track of where the label ended. Both are now bracketed: a same-kind child is
  parenthesized when it is the *right* operand (the operators parse left-associative, so only that
  side is ambiguous), and `name:` brackets a composite child. Mixed nesting is unchanged, and so is
  every string the crate printed before — this only adds parens where two different graphs used to
  render alike. That makes the rendering a canonical form, which is what lets the chain-text parser
  be its exact inverse.
- **One declarative op table (Epic I / I2)** — the op catalog in `fluxion-core` is now declared
  once, in a single `ops!` table: one row per op carrying its doc comment, Rust variant, stable
  text name, catalogue group and parameter schema. `OpKind::name`, `group`, `params` and `all` are
  generated from that row, replacing four hand-kept parallel lists and twenty `static X_PARAMS`
  tables that nothing kept in sync. Adding an op is one edit. `OpKind` stays a real
  `#[non_exhaustive]` enum, so the backend's exhaustive matching and the serde representation are
  unchanged. New: `OpKind::group() -> Group` (`filter` / `effect`), the catalogue split that drives
  `fluxion.filter` vs `fluxion.effect` in Python and the sections of the generated op docs.
  `.fxg` keys on the Rust variant identifier, not the text name, so a committed
  `tests/fixtures/all_ops_v1.fxg` holding one op of every kind now guards the format against a
  variant rename — the one refactor that would silently orphan saved graphs. New registry
  invariants are asserted too: op names are unique and identifier-shaped, parameter names are
  unique per op and identifier-shaped, and every default sits inside its own bounds.
- **CPU batch kernel: set-spreading tile** — `sos_filter_batch`'s AVX2 8-row path now
  bounces each time block through a small padded-pitch scratch instead of loading the
  planar rows at their raw stride. Power-of-two row strides (e.g. the 64×524k paper
  workload's 2 MB) map all sixteen row streams to a single L1/L2 set on 8-way parts
  and cap the kernel at LLC speed; the tile restores cache-resident loads for two
  L2-resident copies. Measured on the i9-10900KF: 1.66 → 2.60 Gsamples/s multi-thread
  (+57%, past TorchFX's OpenMP kernel at 1.89) for ≈5% single-thread; per-row outputs
  stay bit-identical (asserted across tile boundaries and scalar tails).

### Added

- **The interface contract, in `docs/` (Epic I / I1)** — a new `docs/` directory carrying the
  agreement between Fluxion's five doors. [`docs/interfaces.md`](docs/interfaces.md) states the
  rule the epic exists to enforce — *an op ships on every interface, or this document says why
  not* — plus who each interface is for, the definition of done a PR is held to, the ten-line
  quickstart rule and how a line is counted, and an honest "Not yet" section for the gaps that
  remain (the AudioWorklet half of the JS package, the geometry stages, `.to(device)`, alias
  spellings) with the reason for each rather than just the absence.
  [`docs/chain-syntax.md`](docs/chain-syntax.md) documents the shared text form.
  [`docs/ops.md`](docs/ops.md) is **generated** — every op, what it does, its parameters, ranges
  and its name on each interface — by `scripts/gen_interfaces.py`, which reads the registry through
  the new `fluxion effects --json`. A `contract` CI job regenerates and runs `git diff
  --exit-code`, so a hand edit or an op added without regenerating fails the build. The registry
  gained `OpKind::variant()` (the Rust identifier, which is also the Python class name) and
  `OpKind::doc()` (the catalog's doc comment), so an op's documentation is written once and used
  three times: rustdoc, the generated table, and the Python docstring.
- **The chain text syntax (Epic I / I2)** — `"highpass(80, 4) | gain(-3dB)".parse::<Graph>()`. One
  grammar, in `fluxion-core`, that is the exact inverse of what `Graph` prints: whatever the library
  renders parses back to the identical graph, asserted over a corpus covering every op, both
  associativities of `|` and `+`, labels around every node kind, feedback nested either way, and the
  numeric edges (`inf`, signed zero, exponents). That is what lets the CLI's `--chain`, Python's
  `fluxion.chain()`, C's `fx_chain_from_text` and the browser describe a chain with one string
  instead of four dialects. `+` binds tighter than `|`, matching Rust and Python, so
  `a | b + c | d` needs no parentheses. Ergonomics on the way in, canonical form on the way out:
  trailing parameters fall back to their defaults (`highpass(80)`), arguments may be named
  (`compand(threshold=-24, ratio=8)`), there is a `name=v1,v2` shorthand, and the suffixes `k`
  (×1000) and `dB` are accepted — `gain(-3dB)` is an actual 3 dB cut, which `gain(-3)` is not, since
  that parameter is a linear ratio. A suffix a parameter cannot take is refused rather than
  misread. Errors carry a byte offset, a message and an optional fix, and render with a caret:
  `unknown op 'hipass'` under `^^^^^^ did you mean 'highpass'?`. The suggestion helper
  (`fluxion_core::suggest`) counts a transposition as one edit, so `gian` finds `gain`; it adds no
  dependency.
- **Realtime bindings (Python + C ABI)** — the `fluxion-rt` streaming engine is now reachable
  from both binding surfaces, closing the "batch-only bindings" gap. Python: `fluxion.RtChain`
  (`from_chain` / `from_sections`) lowers a `Chain` once at a fixed `fs`, certifies it on the
  stability ladder (an `unstable` verdict is refused), pre-sizes every scratch buffer, then
  `process(input, output)` filters caller-provided numpy blocks in place — allocation-free,
  filter state carried across calls (chunked streaming ≡ whole-signal, asserted), GIL released
  while the Rust kernel runs; `set_coeffs(node, sections, fade_samples)` swaps a filter live with
  the equal-power crossfade (incoming sections certified too), and `filter_count` / `fs` /
  `max_block` / `verdict` / `margin` expose the executor's state. C:
  `fx_rt_new` / `fx_rt_process` / `fx_rt_set_coeffs` / `fx_rt_reset` / `fx_rt_filter_count` /
  `fx_rt_free` plus the `FX_VERDICT_*` codes in the regenerated `include/fluxion.h` — the same
  certification gate, panic-safe at the boundary, with an allocation-free process path that is
  safe inside an audio callback (in/out aliasing and oversized blocks are rejected, not UB).
  This lets a C/C++ host stream a loaded `.fxg` block-by-block; a round-trip test provisions a
  FIR + gain graph to disk, loads it through the C ABI, and asserts the streamed output matches
  the batch path.
- **`.fxg` provisioning from Python** — `Chain.save_fxg(path)` serializes the whole chain
  (biquads, FIR taps, gain, delay) to a sample-rate-agnostic `.fxg` graph a C/C++ host loads with
  `fx_graph_load_fxg` (the pre-existing `save_biquad_fxg` only covered raw SOS sections), and
  `Chain.certify(fs)` returns the `(verdict, margin)` stability certificate for fail-fast
  per-channel checks in a provisioning script.
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

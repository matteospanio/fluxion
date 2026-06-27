# Fluxion — Implementation Plan (scaffold → v1.0.0)

Atomic, PR-sized tasks grouped into epics, with **priority**, **size**, **dependencies**, and a
**∥** flag marking tasks that can run in parallel with their epic siblings once their listed deps
are met. The design rationale for every choice is in [PROJECT.md](PROJECT.md).

## Legend

- **Priority** — `P0` critical path (the spine to a usable release) · `P1` required for 1.0 ·
  `P2` 1.0-optional, may slip to 1.x.
- **Size** — `S` ≈ a few hours · `M` ≈ 1–2 days · `L` ≈ multi-day (split further before starting).
- **∥** — `✓` no ordering constraint vs epic siblings (parallel-safe once deps met) · `—` must follow a sibling.
- **Deps** — task IDs (cross-epic allowed). “—” = none beyond the current scaffold.

## Milestones (the release ladder)

| # | Tag | Theme | Definition of done | Gates on |
|---|-----|-------|--------------------|----------|
| **M0** | — | Scaffold | Workspace builds, core algebra + tests. **DONE.** | — |
| **M1** | `0.1` | CPU batch | CLI filters a WAV on CPU through a parsed effect chain. | B1, C1–C2, D1/D4/D7/D11, H1, I1–I3 |
| **M2** | `0.2` | Differentiable | Analytic backward + Burn autodiff; train-a-filter example; Python eager + autograd via DLPack. | C4, E1/E4/E6/E9, J1–J4 |
| **M3** | `0.3` | Cross-vendor GPU | CubeCL backend validated on NVIDIA **and** Apple; benchmarks. **(CubeCL go/no-go at F0.)** | F0–F2, F5 |
| **M4** | `0.4` | Real-time | Freeze/export + alloc-free engine + CPAL `play`/`record`, xrun-free @128. | G1–G6, I7–I8 |
| **M5** | **`1.0.0`** | Release | SoX-compat CLI, split CPU/GPU wheels, docs, benchmarks, C-ABI, API frozen. | L1–L6 + all P0/P1 |

## Critical path (the spine)

```
B1 ─ C1 ─ C2 ─ D1 ─ D4 ─ D11 ─ I2 ─ I3        (M1: filters a file on CPU)
                          │
        C4(Burn) ─ E1 ─ E6 ─ E9                (M2: trainable)
        J2 ─ J4                                 (M2: torch autograd)
        F0 ─ F1 ─ F2                            (M3: cross-vendor GPU)
        G2 ─ G3 ─ G5 ─ I7                       (M4: realtime)
        L4 ─ L5                                 (M5: 1.0.0 release)
```

Everything not on the spine (infra, IO formats, extra filters/effects, FFI, docs, benchmarks) is
parallelizable around it.

---

## Epic A — Project infrastructure  ·  *parallel from day 1*

| ID | Task | P | Sz | Deps | ∥ |
|----|------|---|----|------|---|
| A1 | GitHub Actions CI: `cargo build/test` on stable (Linux+macOS). | P0 | S | — | ✓ |
| A2 | CI gates: `cargo fmt --check` + `cargo clippy -D warnings`. | P1 | S | A1 | ✓ |
| A3 | `cargo-deny` (licenses/advisories) + `deny.toml`. | P1 | S | — | ✓ |
| A4 | Criterion benchmark harness (`benches/`, `--bench`). | P1 | S | — | ✓ |
| A5 | Per-crate `#![warn(missing_docs)]` + `cargo doc` CI. | P1 | S | A1 | ✓ |
| A6 | `CHANGELOG.md` (Keep a Changelog) + release checklist. | P2 | S | — | ✓ |
| A7 | CONTRIBUTING (point to AGENTS.md) + issue/PR templates. | P2 | S | — | ✓ |

## Epic B — Graph IR & algebra  *(crate: `fluxion-core`)*

| ID | Task | P | Sz | Deps | ∥ |
|----|------|---|----|------|---|
| B1 | Replace `Op{name,params}` placeholder with a typed op model: `OpKind` + parameter descriptors (name, unit, range, default). | P0 | M | — | — |
| B2 | `fs` propagation through the graph + validation (channel/time invariants) with typed errors. | P0 | M | B1 | — |
| B3 | IR pass: SOS-cascade fusion (merge adjacent IIR sections into one fused node). | P1 | M | B1 | ✓ |
| B4 | IR pass: delay-line sharing + common-subexpression elimination. | P2 | M | B1 | ✓ |
| B5 | `.fxg` (de)serialization of a graph + frozen coeffs (serde). | P1 | M | B1 | ✓ |
| B6 | `Display`/DSL pretty-printer for graphs (CLI + debug). | P2 | S | B1 | ✓ |
| B7 | Property tests for algebra laws (series assoc., parallel sum commutes). | P1 | S | B1 | ✓ |

## Epic C — Tensor & backend abstraction  *(crate: `fluxion-backend`)*

| ID | Task | P | Sz | Deps | ∥ |
|----|------|---|----|------|---|
| C1 | Define the `Backend` trait the ops target: `Buf` assoc type + primitive kernels (`map`, `zip`, `conv1d`, `biquad_scan`, `gather`, `rfft`/`irfft`). | P0 | M | — | — |
| C2 | CPU backend: scalar-correct `Backend` impl over a channel×sample buffer. | P0 | M | C1 | — |
| C3 | SIMD-accelerate CPU hot kernels (`pulp`/`wide`, runtime ISA dispatch). | P1 | M | C2, A4 | ✓ |
| C4 | Burn backend: `Backend` impl over Burn tensors (unlocks autodiff + GPU). | P1 | L | C1 | ✓ |
| C5 | Backend/device selection + runtime dispatch (CPU ↔ Burn-CPU ↔ Burn-GPU). | P1 | M | C2, C4 | — |

## Epic D — DSP ops: forward + coefficient design  *(crate: `fluxion-ops`)*

| ID | Task | P | Sz | Deps | ∥ |
|----|------|---|----|------|---|
| D1 | Butterworth SOS design (Lo/Hi), closed-form (no SciPy at runtime). | P0 | M | B1 | — |
| D2 | Chebyshev I/II SOS design (Lo/Hi). | P1 | M | D1 | ✓ |
| D3 | RBJ biquads: peaking, low/high shelf, notch, allpass, bandpass. | P1 | M | D1 | ✓ |
| D4 | SOS/biquad cascade forward kernel (over `Backend`). **CPU SIMD batch variant (2026-06-26):** `sos_filter_interleaved` filters a channel-interleaved (frame-major) batch in place — the per-channel inner loop auto-vectorizes across the batch (an IIR can't vectorize over time). Single-core ~665–691 Msamples/s vs torchfx's fused C++ kernel ~465 (1.4–1.5×); the prior scalar per-row path was ~85 (5.5× slower). Planar (torch `(B,T)`) input needs a transpose (a blocked one ≈ matches torchfx). | P0 | M | C1, D1 | — |
| D5 | FIR + FFT-convolution forward. | P1 | M | C1 | ✓ |
| D6 | Fractional delay line forward. | P1 | M | C1 | ✓ |
| D7 | Gain, Normalize, sum/diff, DC/mask ops. | P0 | S | C1 | ✓ |
| D8 | Reverb forward (FDN or Schroeder). | P1 | M | D6 | ✓ |
| D9 | Echo forward. | P1 | S | D6 | ✓ |
| D10 | Filterbank (band split) forward. | P2 | M | D4 | ✓ |
| D11 | Op registry wiring: every op → `Graph` node + facade constructor, `Lo`/`Hi` naming. | P0 | M | B1, D1, D4, D7 | — |
| D12 | Golden-vector correctness tests vs SciPy/reference oracle (per op). | P0 | M | each op | ✓ |

## Epic E — Differentiability: analytic backward + autodiff  *(crates: `fluxion-ops`, `fluxion-autodiff`)*

| ID | Task | P | Sz | Deps | ∥ |
|----|------|---|----|------|---|
| E1 | Analytic VJP for the SOS cascade (all-pole reformulation, no recursion-unrolling). **Highest-leverage, hardest.** | P1 | L | D4 | — |
| E2 | VJP for FIR/FFT-conv. | P1 | M | D5 | ✓ |
| E3 | VJP for delay line. | P1 | M | D6 | ✓ |
| E4 | VJP for gain/normalize/sum/mask. | P1 | S | D7 | ✓ |
| E5 | VJP for reverb/echo. | P2 | M | D8, D9, E3 | ✓ |
| E6 | Burn `Autodiff` integration: register ops’ owned backward so `loss.backward()` flows. **DONE (2026-06-26):** `fluxion-autodiff` `burn` feature wraps a biquad as a Burn custom op whose backward is the analytic LTI adjoint — gradcheck passes through Burn's tape, backend-agnostic (`Autodiff<NdArray>` tested; `Autodiff<Cuda>` proven in the spike). Coefficient gradients DONE too — `sos_trainable` (binary custom op over input + a trainable coeff tensor, `sos_vjp` for `grad_coeffs`): gradcheck passes and a filter's b-coeffs fit a target through Burn ("learn a filter"). Default build pure-Rust/offline. Next: GPU-kernel forward/backward (wire `fluxion-backend::cuda` into the op). | P1 | M | C4, E1, E4 | — |
| E7 | Finite-difference gradcheck tests (per op). | P1 | M | E1–E4 | ✓ |
| E8 | Stability guard: verify designed/optimized SOS poles inside the unit circle before freeze. | P1 | S | D1, E1 | ✓ |
| E9 | End-to-end “fit a filter to a target” training example + docs. | P1 | M | E6 | — |

## Epic F — GPU backend (CubeCL)  ·  *gated on the F0 go/no-go*  *(crate: `fluxion-backend`)*

| ID | Task | P | Sz | Deps | ∥ |
|----|------|---|----|------|---|
| F0 | **SPIKE — ✅ GO (NVIDIA, 2026-06-26):** Burn 0.21 + CubeCL + CUDA forward **and** on-device autodiff confirmed on an RTX 3070 (see `spikes/f0-burn-cuda`). Apple Metal / AMD ROCm validation still pending. | P0\* | M | D4 | — |
| F1 | CubeCL backend: `Backend` impl (elementwise + conv). **Burn↔CubeCL bridge proven (2026-06-26):** the SOS kernel runs directly on a *resident* Burn `CubeBackend<R>` tensor (public `CubeTensor.client`/`.handle` + `new_contiguous`) — bit-accurate, ~20 ms/iter vs ~430 ms transfer-bound (the resident speedup lands), generic over the runtime → cross-vendor. See `spikes/burn-cubecl-bridge`. Next: F3 backward kernels + wire into `fluxion-autodiff`'s op. | P1 | L | F0, C1 | — |
| F2 | Fused SOS cascade GPU kernel (single dispatch). **DONE + integrated (2026-06-26):** the CubeCL batched-cascade kernel is bit-accurate vs CPU and ~59× on 67 Msamples (RTX 3070), and is now wired into `fluxion-backend` behind the `cuda` feature (`cuda::sos_filter_batch`), GPU-tested against `sos_filter` per row. Default build stays pure-Rust/offline. Spike: `spikes/c4-cubecl-biquad`. | P1 | M | F0, B3 | ✓ |
| F3 | GPU VJP kernels (port the analytic backward to device). **Input-gradient (adjoint) kernel DONE (2026-06-26):** the cascade adjoint = same recurrence backward-in-time, sections reversed — bit-identical to `sos_input_grad`; resident forward+backward ~40 ms/iter (`spikes/burn-cubecl-bridge`). **Coefficient-gradient kernel DONE (2026-06-26):** single-biquad `grad_coeffs` on device — one pass builds the all-pole intermediates inline + accumulates 5 per-coeff sums per row, cross-row reduction via the tiny `[batch,5]` host sum; matches `sos_vjp` (1.9e-4) and finite-diff (8.9e-3). All three kernels verified together in `spikes/burn-cubecl-bridge`. **Burn-autograd integration DONE (2026-06-26):** a custom op over `Autodiff<CubeBackend>` (single trainable biquad) launches the forward + both backward kernels on resident tensors — `loss.backward()` on a GPU tensor gradchecks vs finite-diff (coeff 1.0e-4, input 1.5e-4); only the `[5]` coeffs + `[batch,5]` reduction cross the host. **Workspace port DONE (2026-06-26):** `fluxion-autodiff/src/cuda.rs` (feature `cuda`) ships `sos_gpu` (fixed cascade, input grad) + `biquad_train_gpu` (single trainable biquad); two GPU gradcheck tests pass on the RTX 3070 (`cargo test -p fluxion-autodiff --features cuda`), default build unaffected. Remaining: batched (`[batch, frames]`) entry + cascade coeff-grad orchestration + a cross-vendor (generic-`R`) backend. | P1 | M | F1, E1 | ✓ |
| F4 | FFT-conv on GPU. | P2 | M | F1 | ✓ |
| F5 | Cross-vendor validation matrix on the cluster (NVIDIA, AMD if available, Apple). | P1 | M | F1 | — |
| F6 | Autotuning + perf benchmarks vs CPU and torchaudio. **Benchmarks vs torchfx DONE (2026-06-26):** `spikes/throughput-vs-torchfx`. CPU single-core SIMD ~1.4–1.9× torchfx; GPU resident kernel (RTX 3070) **1.9× torchfx** (2962 vs 1580 Msamples/s); one-shot transfer torchfx 2× — **diagnosed (2026-06-27):** the D2H download is fine (6.3 GB/s); the H2D upload is the bottleneck (~0.9 GB/s) and pinned memory does **not** help (tested) — it's CubeCL 0.10's `create`/upload path (per-call host realloc + slow H2D), an upstream limitation, not a quick fluxion fix. Resident regime unaffected. Autotuning not started. | P2 | M | F2 | ✓ |

\* P0 *for the GPU track*; the CPU release (M1–M2) does not depend on it. If F0 is **No-Go**, fall back to the C++/nanobind + hand-written Metal/CUDA/HIP plan (PROJECT.md §4.1) — the CPU/differentiable milestones are unaffected.

## Epic G — Real-time engine  *(crate: `fluxion-rt`)*

| ID | Task | P | Sz | Deps | ∥ |
|----|------|---|----|------|---|
| G1 | Lock-free SPSC ring buffer (acquire/release atomics, pow-2 mask) + tests. **DONE (2026-06-26):** `fluxion-rt::ring` — free-running head/tail counters (full capacity usable), `Producer`/`Consumer` split, `Copy`-only (alloc/drop-free); tests incl. a 1M-item two-thread SPSC roundtrip. | P1 | M | — | ✓ |
| G2 | Freeze/export: lower graph + designed coeffs to an alloc-free realtime plan (`.fxg`). **DONE (2026-06-26):** `fluxion-core::FrozenSos` (serde, save/load — stores designed coeffs, not just design params like `fxg`) + `fluxion-backend::freeze(graph, fs)` (reuses `graph_to_sos`) → `fluxion-rt::SosStream::from_sections`. Linear cascade only (same constraint as `graph_to_sos`); general-graph plan later. | P1 | M | B5, D1, D4 | — |
| G3 | Alloc-free block executor (pre-allocated state, SIMD MAC loop). **Core DONE (2026-06-26):** `fluxion-rt::stream::SosStream` runs a frozen SOS cascade block-by-block with persistent per-section DF2T state, alloc-free; test proves chunked streaming == `sos_filter` whole-signal. **Integrated executor DONE (2026-06-27):** `fluxion-rt::engine::RtEngine` ties cascade + smoothed gain + lock-free command queue into one alloc/lock/panic-free `process_block` (the G5 callback body). **General-graph executor DONE (2026-06-27):** `fluxion-rt::graph::RtGraph` runs the full series/parallel graph algebra (filter/gain leaves) block-by-block; `prepare(max_block)` sizes the internal scratch once, then `process` is alloc-free (proven in `rt_safety.rs`). Tests: series == concatenated cascade, `(lp+hp)|gain` == reference, nested parallel-in-series. **Delay/echo nodes + graph lowering DONE (2026-06-27):** `RtGraph` gained stateful `Delay`/`Echo` nodes (each owns its delay-line ring, alloc-free `process`), streaming == `fluxion_ops::delay`/`echo`; `fluxion_backend::to_rt_graph(graph, fs)` lowers an arbitrary `Graph` (filters + gain + delay + echo + series/parallel) to a prepared `RtGraph`, verified to stream like batch `process` (returns `None` for whole-signal ops like `Normalize`). **Fractional delay DONE (2026-06-27):** `RtGraph::delay_frac(delay, mix)` — linear-interpolated sub-sample/modulated delay; streaming == reference, integer case == the `Delay` node. (SIMD MAC: N/A for the mono realtime IIR — the recurrence is sequential in time; across-batch SIMD lives in `sos_filter_interleaved`.) | P1 | L | G2, C3 | — |
| G4 | Parameter command queue + `SmoothedValue` ramping (click-free). **DONE (2026-06-27):** `SmoothedValue` (linear ramp, exact landing, alloc-free) + the G1 ring as the command queue, both wired into `RtEngine` (`Command::SetGain` applied at block boundaries). Hardened by an adversarial multi-agent review of `fluxion-rt`: fixed a **critical** ring soundness hole (`push`/`pop` now take `&mut self` so single-producer/single-consumer is borrow-checked, not just documented) + capacity-overflow guard; `MaybeUninit` slots (any `Copy` type). | P1 | M | G3 | ✓ |
| G5 | CPAL audio I/O backend (cross-platform callback). **DONE (2026-06-27):** `fluxion-rt::cpal_backend` (feature `cpal`) — `run_output(render)` opens the default output device and drives an alloc-free render callback on the audio thread (returns `sample_rate`/`channels`); `BackendError` wraps the CPAL errors. Compiles + clippy + doctest clean locally (ALSA); default build pulls no audio libs. Remaining: duplex (live-input) + non-f32 formats. | P1 | M | G1, G3 | — |
| G6 | Real-time-safety tests (no-alloc-in-callback assertion + xrun stress @128/48k). **DONE (2026-06-27):** `fluxion-rt/tests/rt_safety.rs` — a tracking `#[global_allocator]` flags allocations made inside a (thread-local) real-time section; asserts `RtEngine::process_block` allocates zero across 1000+ blocks under concurrent command automation, plus a `meta_tracker` test proving the tracker actually catches allocations (false-negative guard). xrun stress: 5 s of audio @ 128/48 kHz processed ~770× real time, alloc-free. | P1 | M | G3 | ✓ |

## Epic H — Audio & batch IO  *(crate: `fluxion-io`)*

| ID | Task | P | Sz | Deps | ∥ |
|----|------|---|----|------|---|
| H1 | WAV read/write (`hound`). | P0 | S | — | ✓ |
| H2 | Symphonia decode (flac/mp3/ogg/aac → samples + fs). | P1 | M | — | ✓ |
| H3 | Encoders for output formats (WAV P0; others P2). | P1 | M | H1 | ✓ |
| H4 | Arrow/Parquet batch IO (dataset → record batches). | P2 | M | — | ✓ |
| H5 | Streaming/chunked reader for large files. | P1 | M | H2 | — |

## Epic I — CLI  *(crate: `fluxion-cli`)*

| ID | Task | P | Sz | Deps | ∥ |
|----|------|---|----|------|---|
| I1 | clap skeleton + global flags (`--device`, `--fs`, `-o`, verbosity). | P0 | S | — | — |
| I2 | Effect-chain parser: positional `effect --flag val …` → `Graph`. | P0 | M | B1, D11 | — |
| I3 | `process`: in → chain → out (file). | P0 | M | I2, H1, C2 | — |
| I4 | `info` (soxi-style metadata). | P1 | S | H2 | ✓ |
| I5 | stdin/stdout `-` filter mode. | P1 | S | I3 | ✓ |
| I6 | glob/batch `--each` over many files. | P1 | M | I3 | ✓ |
| I7 | `play` / `record` (realtime). | P1 | M | G5, I2 | ✓ |
| I8 | `compile` → `.fxg`. | P1 | S | B5, G2 | ✓ |
| I9 | SoX-compat shims (`soxi`, `-n` null sink) + help/man polish. | P2 | M | I3, I4 | ✓ |
| I10 | CLI integration tests (golden output). | P1 | M | I3 | ✓ |

## Epic J — Python bindings  *(crate: `fluxion-py`, `python/`)*

| ID | Task | P | Sz | Deps | ∥ |
|----|------|---|----|------|---|
| J1 | PyO3 module skeleton + maturin build (`crate-type=cdylib`). **DONE (2026-06-27):** `fluxion-py` is a PyO3 0.22 + numpy cdylib built by maturin (`import fluxion`), a standalone crate excluded from the cargo workspace so `extension-module` linking doesn't break `cargo test --workspace`. | P1 | S | — | — |
| J2 | DLPack producer/consumer (zero-copy ↔ torch/numpy/jax). Numpy interop via the `numpy` crate (copy-in/out) ships now; zero-copy DLPack for torch/jax tensors is the follow-up. | P1 | M | C4 | — |
| J3 | Eager transform API: `chain(x)` torchaudio-style. **DONE (2026-06-27):** `Chain` with `lowpass`/`highpass`/`peaking`/`gain`/`delay`/`echo`/… constructors, `|` (series) and `+` (parallel) operators, `.process(np_array, fs)` via `fluxion_backend::process`. Pytest: shapes/dtype, low-pass attenuation, gain/parallel exactness, invalid-param `ValueError`. | P1 | M | J2, D11 | — |
| J4 | `torch.autograd.Function` adapter (forward + owned backward). **DONE (2026-06-27):** Rust exposes `sos_forward`/`sos_backward` (the analytic VJPs — input grad + coeff grad); a Python `torch.autograd.Function` wraps them. Finite-difference gradcheck (numpy) passes, and the torch adapter gradchecks with real torch (verified against torchfx's torch env). | P1 | M | J2, E6 | ✓ |
| J5 | `jax.custom_vjp` adapter. | P2 | M | J2, E6 | ✓ |
| J6 | Array API conformance layer + `.pyi` type stubs. | P2 | M | J3 | ✓ |
| J7 | Python tests (parity vs torchaudio) + `pyproject` + cibuildwheel (CPU). | P1 | M | J3 | — |
| J8 | Split GPU wheels (cibuildwheel CUDA images). | P2 | L | J7, F1 | — |

## Epic K — FFI / C-ABI  *(crate: `fluxion-ffi`)*  ·  *parallel*

| ID | Task | P | Sz | Deps | ∥ |
|----|------|---|----|------|---|
| K1 | cbindgen config → generated `fluxion.h` in CI. | P2 | S | — | ✓ |
| K2 | C-ABI: graph build/parse, `fx_process_dlpack`, lifecycle (`free`). | P2 | M | B5, C2 | — |
| K3 | C example + smoke test linking the staticlib. | P2 | S | K2 | ✓ |
| — | *(WASM bindings — `fluxion-wasm` — deferred to 1.x; not a 1.0 gate.)* | — | — | — | — |

## Epic L — Quality, docs & release  ·  *the 1.0 gate*

| ID | Task | P | Sz | Deps | ∥ |
|----|------|---|----|------|---|
| L1 | Benchmark suite vs SciPy + torchaudio (filters, batch); publish results. | P1 | M | D, C3, A4 | ✓ |
| L2 | Coverage gate + golden-vector oracle covering all ops. | P1 | M | D12, E7 | ✓ |
| L3 | Docs site (rustdoc + mdBook guide: quickstart, CLI, training, realtime). | P1 | L | most | ✓ |
| L4 | API stabilization: review public surface, semver, `#[non_exhaustive]`, deny `missing_docs`. | P0 | M | all public crates | — |
| L5 | Finalize CHANGELOG, tag `v1.0.0`, publish to crates.io + PyPI. | P0 | S | everything | — |
| L6 | Cross-vendor + realtime sign-off (GPU NVIDIA+Apple, xrun-free @128/48k). | P1 | M | F5, G6 | — |

---

## Parallelization waves (suggested execution order)

Each wave assumes the previous one’s **P0** spine tasks are done; ✓-marked tasks within a wave run
concurrently (ideally one contributor or one worktree per lane).

- **Wave 0 (now):** A1–A3 ∥ B1 ∥ H1. *(infra + IR model + WAV IO, all independent)*
- **Wave 1 → M1:** C1 → C2 ; D1 → D4 ; D7 ; D11 ; I1 → I2 → I3. Parallel lanes: H2/H5, A4–A5, B5/B7, D2/D3.
- **Wave 2 → M2:** C4 (Burn) ‖ E2/E3/E4 ; then E1 → E6 → E9 ; J1 → J2 → J3 → J4 ; E7/E8 ∥.
- **Wave 3 → M3:** **F0 first (go/no-go)**, then F1 → F2 ‖ F3 ‖ F5 ; F4/F6 ∥. Independent: H4, D8/D9/D10, I4–I6.
- **Wave 4 → M4:** G1 ∥ early ; G2 → G3 → {G4, G5, G6} ; I7, I8 ; K1–K3 ∥.
- **Wave 5 → M5 (1.0):** L1, L2, L3 ∥ ; then L4 → L6 → L5. J8 ∥.

## Notes on sequencing

- **F0 is the single riskiest gate.** Schedule the CubeCL spike as early as Wave 2 (in parallel with
  the differentiable work) so a No-Go is known before committing to the GPU lane. The CPU +
  differentiable + Python milestones (M1–M2) are deliberately independent of it.
- **E1 (analytic SOS backward) is the hardest task and the project’s durable asset** — give it the
  strongest contributor and budget for L size. Its correctness is gated by E7 (gradcheck) and E8
  (stability).
- **Keep heavy deps per-crate.** Adding `burn`/`cubecl` (C4/F1), `clap` (I1), `symphonia` (H2),
  `pyo3` (J1) happens inside their own crates so the rest of the workspace keeps building offline.
- **A 1.0 without GPU is still shippable** as `1.0.0` (CPU + differentiable + CLI + realtime +
  Python), with GPU promoted from a feature flag once F5 signs off — decide at M3 based on F0.

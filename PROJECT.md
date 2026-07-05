# Fluxion — Project Proposal

> A differentiable, cross-vendor, framework-agnostic audio DSP library with a functional
> graph API, a modern SoX-replacement CLI, and a hard-real-time engine — written in Rust,
> bound to anything.

Status: **draft proposal**. This document is the strategic and technical spec, not yet code.

---

## 1. Name

**Project / library name (recommended): `fluxion`.**

A *fluxion* is Newton's own word for a **derivative** (the instantaneous rate of change of a
*fluent*). It is therefore the perfect name for a library whose two defining properties are:

- **flux** — it processes *signal flow* (a functional graph of audio streams), and
- **-ion / fluxion** — every node in that flow is **differentiable**.

One word that means both "signal flow" and "derivative". Ownable, meaningful, not cute.

### Alternative names (check crates.io / PyPI / npm availability before committing)

| Name | Rationale | Risk |
|------|-----------|------|
| **`fluxion`** ⭐ | Newton's term for *derivative* + signal *flux*. Differentiable signal flow in one word. | Minor: a small unrelated actor crate exists. |
| `phonon` | The quantum of sound/vibration — a particle that travels through *any medium* (cross-vendor metaphor). | KDE Phonon (different domain, multimedia API). |
| `cascade` | SOS cascades / signal cascades; instantly readable to DSP people. | Generic, likely taken. |
| `resound` | "to sound everywhere" — contains *sound*, evokes cross-platform reach. | Clean. |
| `tine` | The tine of a tuning fork / Rhodes piano. Short, audio-native, ownable. | Clean. |
| `sonance` | The state of sounding (root of *con-sonance* / *dis-sonance*). | Clean. |

> ⚠️ `fundsp` and `grafx` already exist (functional Rust audio DSP, and a PyTorch audio-graph
> library respectively) — deliberately stay clear of those names. See [§13 References](#13-references).

The rest of this document uses **`fluxion`** / `fx::` as the working name.

---

## 2. Problem statement & motivation

Existing differentiable-audio libraries (torchaudio, **torchfx**, GRAFX, NablAFx, FLAMO) are
excellent but **married to PyTorch**, which transitively binds them to one tensor runtime, one
autograd, and effectively the NVIDIA/CUDA happy-path. The thesis of `fluxion`:

> The DSP value-add — the filter/effect algebra, the analytic gradients, the realtime engine,
> the CLI — is **not** intrinsically tied to any one framework or GPU vendor. Separate the
> *value-add* from the *substrate* and you can serve PyTorch, JAX, NumPy, bare Rust, the browser,
> and a hard-real-time audio callback from a single codebase.

### Four hard requirements (from the brief)

1. **General-purpose, differentiable DSP.** IIR / FIR / SOS filters, effects (reverb, gain,
   echo, …), masking, sums/differences of signals. Every module is **both** a plain
   transform (torchaudio-style) **and** a trainable, differentiable node (DDSP-style).
2. **A modern CLI** that replaces SoX and is pleasant to script in bash.
3. **Minimal external dependencies**, ideally self-contained.
4. **Extremely efficient** for *both* low-latency realtime *and* large-batch DL.

Plus two cross-cutting constraints: **framework-agnostic** (integrable into "potentially any"
framework via a functional API) and **hardware-agnostic** (NVIDIA / AMD-ROCm / Apple Silicon /
CPU, ideally WASM/WebGPU).

### Two further goals (added 2026-07)

5. **ML data augmentation.** The batched CPU/GPU path doubles as an augmentation engine for
   dataset pipelines: stochastic effect chains (randomly sampled parameters, apply-probability)
   over `(B, T)` batches, exposed in Python (`fluxion.augment`) and from the CLI (`batch`).
6. **Import DDSP modules trained in other frameworks.** Fluxion is not only integrable *into*
   host frameworks — it also *consumes* their artifacts: a checkpoint trained elsewhere (named
   targets: **FLAMO** SISO `SOSFilter`/`SVF`/`Biquad` cascades and **torchfx.ddsp** learnable
   filters, via `safetensors` state-dicts — `.pt`/`.onnx` parse on the Python side) is replayed
   into SOS coefficients by a Rust converter (`fluxion-io::checkpoint`, feature `checkpoint`),
   **stability-certified** (E8 ladder, with an opt-in Jury projection for unconstrained
   checkpoints), and written as a standard raw-`biquad` `.fxg` that processes, plays, and
   hot-swaps like any native graph (`fluxion import` verb /
   `fluxion.interop.import_checkpoint`). MIMO banks and frequency-sampled FIRs stay out of
   scope; the importer rejects them with actionable errors.

### The central tension (and its resolution)

"Differentiable" is *exactly* the thing a framework gives you for free. Resolving "differentiable
**and** not tied to a framework" is the make-or-break design decision. The resolution
(detailed in [§5](#5-architecture)):

- **Own the backward math, rent the graph machinery.** The DSP op set is *narrow* (SOS cascade,
  FIR/FFT-conv, delay line, gain, a handful of effects). For recursive IIRs you *must*
  hand-derive the analytic backward anyway — the DDSP literature shows naive autograd-unrolling
  of the recursion is ~30× slower and memory-heavy (Differentiable All-Pole Filters; Yu & Fazekas;
  torchlpc). So owning a per-op VJP is **unavoidable and is the durable asset**.
- **Never build a general autograd *engine*.** Plug the owned VJPs into whatever autograd is
  present: Burn's `Autodiff` backend for a torch-free Rust path, `torch.autograd.Function` /
  `jax.custom_vjp` when a host framework is present. DLPack moves *values, not graphs*, so the
  backward must be supplied per-framework regardless — design around that.

---

## 3. Design principles

1. **One declarative graph, two execution backends.** A functional graph algebra is the single
   user-facing API; it *lowers* to (a) a batch/training executor (GPU+CPU, autograd, allocates
   freely) and (b) an allocation-free realtime CPU executor. **Never run autograd/GPU dispatch
   inside an audio callback.**
2. **Standards-first boundary.** The public surface speaks **C-ABI · DLPack · Python Array API ·
   Apache Arrow** so the core is framework- and language-agnostic by construction.
3. **Own the backward, rent the graph.** (See §2.) The kernels and their VJPs are framework-free;
   thin adapters register them with Burn / torch / JAX.
4. **One kernel, all vendors.** A single kernel definition lowers to CUDA / ROCm-HIP / Metal /
   Vulkan / WGSL / CPU-SIMD via CubeCL — no per-vendor kernel zoo.
5. **Self-contained artifact.** Pure-Rust codecs (Symphonia) and audio I/O (CPAL) so the shipped
   binary/wheel has no libsndfile/ffmpeg/PortAudio C dependency.
6. **Lazy is a sin here.** Design-time coefficient computation is an explicit *compile* stage, not
   lazy mutation inside `forward` (that idiom breaks AOT/scripting — cf. the nnAudio2 TorchScript
   breakage). The graph is reified and inspectable.

---

## 4. Technology & language decisions (with trade-offs)

### 4.1 Core language

**Decision: Rust.** Rationale and the honest counter-arguments:

| Language | Pros for *this* project | Cons / blockers |
|----------|-------------------------|-----------------|
| **Rust** ⭐ | Only ecosystem with a **single-source cross-vendor GPU** layer (CubeCL) **and** a backend-agnostic **autodiff** (Burn). Ownership/`Send`+`Sync` make the lock-free realtime ring buffer, GPU stream sync, and zero-copy FFI *compile-time* safe — exactly the 3am-bug hotspots. `extern "C"`+`#[repr(C)]` gives the **same stable C-ABI** as C. Cargo + PyO3/maturin + wasm-bindgen + cbindgen = near-zero binding/packaging boilerplate. | CubeCL/Burn are **pre-1.0 / alpha** (real risk). Stable-Rust SIMD *multiversioning* (runtime AVX-512 vs NEON dispatch) is still awkward (`std::simd` nightly; `wide`/`pulp` on stable). |
| **C** | Maximal self-containment & embeddability; mature SIMD; the reference projects (AudioNoise, ds4, SoX) prove self-contained multi-backend C is *possible* (ds4 = CPU ref + Metal + CUDA + ROCm). Trivial FFI substrate. | **No cross-vendor GPU library** → you hand-maintain CUDA+Metal+HIP kernel families (the "kernel zoo"). **No autograd ecosystem** (ggml has only partial/example backward) → you build the tape yourself. Data races / use-after-free in the realtime+GPU+FFI hotspots are *yours to find*. Manual tooling (autotools/CMake). |
| **C++** | Mature SIMD (Highway, function multiversioning); `nanobind` gives best-in-class zero-copy `ndarray` (DLPack + buffer protocol, ~10× lower call overhead than pybind11); the pragmatic choice **if evolving torchfx**. | No single-source cross-vendor abstraction outside **SYCL** (which *does not target Apple Metal*) → same kernel-zoo problem for Apple. No safety guarantees. |
| **Zig** | First-class `@Vector` SIMD; tiny, fast compiles. | 0.16 **beta**, pre-1.0; no audio/DSP or GPU-kernel ecosystem; Python bindings (Pydust) lag the compiler; smallest community. Watch, don't bet. |

**The framing:** in C you hand-write the *three hardest* things (cross-vendor GPU, autograd,
concurrent safety); the DSP math is the same amount of code either way. The C-ABI — C's one real
advantage for bindings — you get identically from Rust. Hence Rust. *If* alpha-dependency risk is
unacceptable, the fallback is **C++ + nanobind + hand-written Metal/CUDA/HIP**, accepting the
kernel zoo.

### 4.2 Cross-vendor GPU layer

**Decision: CubeCL**, with `wgpu` reserved for the WASM/WebGPU path only.

| Option | NVIDIA | AMD | Apple | CPU | Single source? | Notes |
|--------|:---:|:---:|:---:|:---:|:---:|-------|
| **CubeCL** ⭐ | ✅ CUDA | ✅ HIP | ✅ Metal | ✅ SIMD | **yes** | + Vulkan/WGSL. Autotuning, kernel fusion, comptime specialization. Drives Burn. **Alpha (v0.10), breaking changes.** |
| wgpu / WebGPU | ✅ | ✅ | ✅ | ✅ | yes | Portable but **dispatch 24–71 µs** → unfit for low-latency; fine for batch & browser. |
| SYCL (AdaptiveCpp) | ✅ | ✅ | ❌ | ✅ | partial | No Metal on Apple Silicon (OpenCL only on Intel Macs). C++-only. |
| Hand-written CUDA+HIP+Metal | ✅ | ✅ | ✅ | — | **no (3 codebases)** | Best latency, worst maintenance. The fallback. |
| candle backends | ✅ | ❌ | ✅ | ✅ | yes | No AMD. Inference-leaning. |

Low-latency small-kernel DSP (SOS biquads, delay lines) is *dispatch-overhead bound*, so the
layer **must fuse cascades into one launch**; CubeCL's fusion + autotuning does this. wgpu's
dispatch cost is why it's batch/browser-only.

### 4.3 Autodiff strategy

**Decision: own the per-op VJP; expose via Burn `Autodiff` (Rust-native) + per-framework adapters.**

| Strategy | Verdict |
|----------|---------|
| Write a self-contained general autograd engine | ❌ micrograd is 150 LOC but real cost is GPU perf + op coverage + numerical edge cases = re-implementing torch/JAX for a narrow op set. Not worth it. |
| **Hand-derived analytic backward per op** ⭐ | ✅ Mandatory for recursive IIR anyway (~30× vs unrolling). Small, engine-independent, the durable asset. |
| Burn `Autodiff<B>` backend decorator | ✅ Wraps any CubeCL/torch/ndarray backend; gives reverse-mode for free over the custom ops. The Rust-native trainable path. |
| Enzyme / Rust `std::autodiff` | ⏳ Nightly-only (mid-2026), macOS blocked, type-analysis compile tax. Watch as a future torch-free runtime, don't depend on it. |
| Rent host framework (torch `Function` / jax `custom_vjp`) over DLPack | ✅ How `import fluxion` integrates into existing PyTorch/JAX training. DLPack carries values not grads → must register backward per framework (which we own). |

### 4.4 Bindings & interop standards

The framework-agnostic contract is **four layers**, all language-independent:

- **Stable C-ABI** (`#[repr(C)]` + cbindgen) — the universal substrate for C / C++ / Swift / Kotlin / JS bindings.
- **DLPack** — zero-copy CPU+GPU tensor handoff to NumPy / PyTorch / JAX / CuPy. *Caveats:* same physical device + contiguous; honor the CUDA stream handshake.
- **Python Array API (2024.12 / 2025)** — so the Python orchestration layer is not torch-syntax-locked (`array-api-compat` shim).
- **Apache Arrow C Data Interface / nanoarrow + Parquet** — batch/file IO for the CLI (decode → record batches → Parquet/IPC), zero-copy, ABI-only (no link dependency).

Python bindings via **PyO3 + maturin**; JS/browser via **wasm-bindgen** (CPU + WebGPU).

---

## 5. Architecture

```
                         ┌───────────────────────────────────────────────┐
                         │   Functional graph algebra  (user-facing API)  │
                         │   nodes = filters/effects ; edges = signals    │
                         │   operators:  |  (series)   +  (parallel sum)  │
                         │               splits / merges / feedback (~)   │
                         └───────────────────────────────┬───────────────┘
                                                          │  reify → small graph IR
                                          ┌───────────────┴───────────────┐
                                          │   IR passes (engine-agnostic)  │
                                          │  • SOS-cascade fusion          │
                                          │  • delay-line sharing / CSE    │
                                          │  • vectorization knobs         │
                                          │  • design-time coeff compute   │  ◀── explicit "compile",
                                          └───────────────┬───────────────┘       NOT lazy-in-forward
                          lower ↙                                         ↘ lower (freeze coeffs)
        ┌───────────────────────────────────┐             ┌──────────────────────────────────────┐
        │      BATCH / TRAINING engine        │             │          REALTIME engine             │
        │  • CubeCL kernels (CUDA/HIP/Metal/  │             │  • allocation-free, lock-free        │
        │    Vulkan/WGSL) + CPU SIMD          │             │  • SPSC ring buffer, atomics         │
        │  • Burn Autodiff (reverse-mode)     │             │  • frozen coeffs, pure SIMD MAC loop │
        │  • allocates freely, fuses, batches │             │  • parameter command queue + ramps   │
        │  → training, DDSP, offline CLI      │             │  → audio callback, plugins, `play`   │
        └───────────────────────────────────┘             └──────────────────────────────────────┘
                          │                                                  ▲
                          └──────────────  freeze / export  ─────────────────┘
                                  (design coeffs once, optionally
                                   via differentiable optimization)
```

**Why two engines, never one:** hard-RT audio (no alloc, no lock, ~1.3–10 ms budget) and
large-batch DL have opposite physics. A GPU kernel launch costs 3–50 µs → on a single low-latency
stream the GPU is *strictly worse* than CPU-SIMD (naive per-callback GPU dispatch ≈ 1000× slower
than realtime). Every serious project (RTNeural, JUCE, NAM, DDSP) splits this way: train
differentiably on GPU in batches, *export frozen weights/coeffs*, run inference through a separate
allocation-free SIMD engine on the audio thread.

---

## 6. Repository / crate layout

A Cargo **workspace** of focused crates. Each is independently testable; the FFI/Python/WASM/CLI
crates are thin shells over the core.

```
fluxion/
├── Cargo.toml                      # [workspace]
├── PROJECT.md                      # this file
├── README.md
├── crates/
│   ├── fluxion-core/               # graph IR, algebra (| + ~), node traits, scheduling, IR passes
│   │   ├── src/graph.rs            #   reified DSP graph + lowering entry points
│   │   ├── src/algebra.rs          #   operator overloads: BitOr=series, Add=parallel
│   │   └── src/pass/               #   fusion, delay-sharing, CSE, coeff-design
│   ├── fluxion-ops/                # DSP primitives: forward + ANALYTIC backward (the owned VJPs)
│   │   ├── src/iir.rs              #   SOS/biquad cascade (+ Lo/Hi variants) + backward
│   │   ├── src/fir.rs              #   FIR / FFT-conv + backward
│   │   ├── src/delay.rs            #   fractional delay line + backward
│   │   ├── src/effect.rs           #   gain, reverb, echo, normalize, mask
│   │   └── src/coeffs.rs           #   design-time math (Butterworth/Chebyshev/RBJ, SOS)
│   ├── fluxion-backend/            # lowering targets
│   │   ├── src/cubecl/             #   GPU kernels (one source → CUDA/HIP/Metal/Vulkan/WGSL/CPU)
│   │   └── src/simd/               #   CPU SIMD reference (wide/pulp)
│   ├── fluxion-autodiff/           # Burn Autodiff integration + framework VJP adapters
│   ├── fluxion-rt/                 # realtime engine: SPSC ring buffer, alloc-free executor, param queue
│   ├── fluxion-io/                 # Symphonia decode + hound/wav + Arrow/Parquet batch IO
│   ├── fluxion-ffi/                # C-ABI (cbindgen header gen) + DLPack producer/consumer
│   ├── fluxion-py/                 # PyO3 bindings + torch.autograd.Function / jax custom_vjp adapters
│   ├── fluxion-wasm/               # wasm-bindgen (CPU + WebGPU path)
│   └── fluxion-cli/                # clap-based SoX-replacement CLI (`fluxion`)
├── python/
│   ├── fluxion/__init__.py         # pythonic wrapper, Array-API layer, type stubs
│   └── tests/
└── bindings/
    └── fluxion.h                   # generated C header (for C/C++/Swift/JS consumers)
```

---

## 7. Dependencies

Curated and minimal. The "risk" column flags pre-1.0 / alpha pins.

| Crate | Purpose | Pure Rust? | Risk |
|-------|---------|:---:|------|
| `burn`, `burn-autodiff` | tensor + reverse-mode autodiff, backend-agnostic | ✅ | pre-1.0 |
| `cubecl` | one kernel → CUDA/HIP/Metal/Vulkan/WGSL/CPU-SIMD | ✅ | **alpha** |
| `wgpu` | WebGPU path (WASM + batch fallback) | ✅ | stable-ish |
| `pyo3` + `maturin` | Python bindings + wheel build | ✅ | stable |
| `cbindgen` | generate `fluxion.h` from the C-ABI crate | ✅ | stable |
| `wasm-bindgen` | JS/browser bindings | ✅ | stable |
| `clap` | CLI argument parsing | ✅ | stable |
| `symphonia` | pure-Rust decode (wav/flac/mp3/ogg/aac) — replaces libsndfile/ffmpeg | ✅ | stable |
| `hound` | simple WAV read/write | ✅ | stable |
| `cpal` | cross-platform realtime audio I/O — replaces PortAudio/sounddevice | ✅ | stable |
| `rustfft` / `realfft` | FFT for FFT-convolution (or use CubeCL FFT on GPU) | ✅ | stable |
| `arrow` / `parquet` | batch/file IO, columnar | ✅ | stable |
| `wide` / `pulp` | stable-Rust CPU SIMD (multiversioning via `pulp`) | ✅ | stable |

**Self-containment payoff:** Symphonia + CPAL + Arrow + rustfft are *all pure Rust*, so the
shipped artifact carries **no C audio dependency** (no libsndfile, ffmpeg, or PortAudio) — directly
serving requirement #3. The only heavy, non-pure-Rust pieces are the GPU vendor SDKs pulled in by
CubeCL *at build time* for the GPU wheels.

**Known distribution pain (language-independent):** GPU wheels are large (CUDA matrix, 600–900 MB)
and PyPI has no CUDA metadata. Mitigation: ship **split wheels** — `fluxion` (CPU-only, pure Rust,
tiny) and `fluxion-cuda` / `fluxion-rocm` / `fluxion-metal` variants — built with `cibuildwheel`
CUDA images. Lean on CUDA Enhanced Compatibility; track the Wheel Variants proposal.

---

## 8. API examples

### 8.1 Rust — functional graph, as a plain transform

The algebra mirrors torchfx for continuity: `|` = series, `+` = parallel (summed) — implemented as
`BitOr` / `Add` operator overloads on graph nodes.

```rust
use fluxion::prelude::*;

// Build a graph declaratively. Coefficients are NOT computed yet (no fs known).
let chain = LoButterworth::cutoff(800.0).order(4)
          | Reverb::new(/*mix*/ 0.4, /*decay*/ 0.8)
          | Gain::db(-3.0);

// A nested topology: parallel low/high shelves summed, then a trim.
let eq = (LoShelf::cutoff(200.0).gain_db(6.0) + HiShelf::cutoff(4_000.0).gain_db(-3.0))
       | Gain::db(-1.0);

// Apply as a pure transform (inference, no grad). `fs` flows in from the Wave.
let wave = Wave::read("in.wav")?;             // (channels, samples) + fs
let out: Wave = chain.process(&wave);         // lowers → batch engine, picks the device
out.write("out.wav")?;
```

### 8.2 Rust — differentiable / trainable (Burn autodiff, any vendor)

```rust
use fluxion::prelude::*;
use burn::{backend::Autodiff, optim::AdamConfig};

type Gpu = fluxion::backend::Cube;        // CUDA / HIP / Metal — chosen at runtime
type B   = Autodiff<Gpu>;                 // reverse-mode autodiff over the GPU backend

// Learnable parameters: cutoff and gain are tracked tensors.
let mut graph = (LoButterworth::cutoff_learnable(800.0).order(4)
               | Gain::db_learnable(0.0)).init::<B>(&device);

let opt = AdamConfig::new().init();
for (x, target) in dataloader {
    let y    = graph.forward(x);                 // uses owned analytic VJP for the IIR
    let loss = mse(&y, &target);                 // multi-resolution STFT loss, etc.
    let grads = loss.backward();                  // Burn reverse-mode
    graph = opt.step(lr, graph, grads.into());    // updates cutoff & gain
}
```

### 8.3 Python — torchaudio-style transform + zero-copy torch interop

```python
import fluxion as fx
import torch

# 1) Plain transform — torchaudio style. Zero-copy via DLPack: cuda:0 in → cuda:0 out, no copy.
chain = fx.LoButterworth(cutoff=800, order=4) | fx.Reverb(mix=0.4, decay=0.8) | fx.Gain(db=-3)
x = torch.randn(2, 48_000, device="cuda")
y = chain(x)                       # torch.Tensor on cuda:0

# 2) Differentiable inside a PyTorch training loop.
#    chain.torch() wraps the owned forward+backward in a torch.autograd.Function.
eq = (fx.LoShelf(cutoff=200, gain_db=6) + fx.HiShelf(cutoff=4000, gain_db=-3)).torch()
params = eq.parameters()           # registers as torch nn.Parameter
y = eq(x); loss = mse(y, target); loss.backward()   # grads flow into params

# 3) JAX is symmetric: chain.jax() → jax.custom_vjp wrapper.
```

The Python layer conforms to the **Array API** (`array-api-compat`), so the same code runs over
NumPy / CuPy / PyTorch / JAX arrays; DLPack handles the device handoff.

### 8.4 CLI — the SoX substitute

SoX's *jobs*, not SoX's interface: named effects with long `--flags`, explicit units (Hz, seconds,
dB), SI suffixes (`1k`), and a self-describing catalog (`fluxion effects [name]` prints every
parameter, unit, and default). Adjacent effects fuse into one filter pass; geometry stages
(`trim`, `pad`, `rate`, `speed`, `repeat`, `silence`, `channels`, `remix`) change frame/channel
count or fs and run between passes.

```bash
# Effect chain on one file (SoX-style positional pipeline; --db converts at parse time).
fluxion in.wav  lowpass --cutoff 800 --order 4  reverb --mix 0.4  gain --db -3  out.wav

# The convert jobs: trim + resample + bit-depth reduction (TPDF-dithered by default).
fluxion --bits 16 in.flac  trim --start 0.25 --len 30  rate --fs 44100  out.wav

# Multiple inputs: concatenate (default) or sum with --mix.
fluxion --mix a.wav b.wav mixed.wav

# Inspect / analyze (WAV via hound; FLAC/MP3/OGG/… via Symphonia probe).
fluxion info in.mp3          # alias: soxi
fluxion stat in.wav          # peak/RMS dBFS, DC offset, crest factor, …

# Unix filter on stdin/stdout; -n null sink; glob batch; tone generator.
cat in.wav | fluxion - lowpass --cutoff 1k - > out.wav
fluxion batch out/ 'data/*.wav' highpass --cutoff 80
fluxion synth --wave sine --freq 440 --secs 1 tone.wav

# Realtime monitoring through the CPU realtime engine (feature `realtime`; frozen coeffs, no GPU).
fluxion play in.wav  lowpass --cutoff 800
fluxion record --secs 5  gain --db 6  rec.wav

# Export a graph to a portable artifact (frozen coeffs, stability-certified) and reuse it.
fluxion compile lowpass --cutoff 800 reverb --mix 0.4  chain.fxg
fluxion in.wav chain.fxg out.wav
```

Planned, not yet implemented: `--device cuda` batch dispatch (waits on backend device selection,
plan C5) and time-stretch/pitch-shift stages (`tempo`/`pitch`, plan roadmap).

### 8.5 Rust — realtime engine (allocation-free)

```rust
// Lower the SAME graph to the realtime engine: pre-allocates everything, freezes coeffs.
let rt = chain.compile_realtime(RtConfig { fs: 48_000, block: 128, channels: 2 })?;

// In the audio callback (high-priority thread): NO alloc, NO lock, bounded time.
audio_callback(move |input: &[f32], output: &mut [f32]| {
    rt.process_block(input, output);          // pure SIMD MAC loop over frozen coeffs
});

// Parameter automation from another thread — staged via a lock-free command queue,
// applied at block boundaries with ramping to avoid zipper noise.
rt.set_smoothed("lowpass.cutoff", 1_200.0, /*ramp_ms*/ 20.0);
```

### 8.6 C-ABI — for C/C++/Swift/JS consumers (generated `fluxion.h`)

```c
#include "fluxion.h"

fx_graph*  g = fx_parse("lowpass --cutoff 800 | reverb --mix 0.4");  // or build programmatically
fx_status  st = fx_set_fs(g, 48000);

// Process via DLPack — zero-copy, CPU or GPU, no fluxion-side allocation.
DLManagedTensor *in = /* your tensor */, *out = NULL;
st = fx_process_dlpack(g, in, &out);

fx_tensor_free(out);
fx_graph_free(g);
```

---

## 9. Interfaces summary

| Interface | Mechanism | Serves |
|-----------|-----------|--------|
| Native Rust API | `fluxion-core` traits + operator algebra | Rust users, the engine internals |
| Python | PyO3 + DLPack + `torch.autograd.Function` / `jax.custom_vjp` adapters + Array API | torchaudio/DDSP users, training |
| C / C++ / Swift / Kotlin | `#[repr(C)] extern "C"` + cbindgen header + DLPack | embedding in plugins, other languages |
| JS / browser | wasm-bindgen, CPU + WebGPU | web audio, demos |
| CLI | clap binary `fluxion` | bash scripting, SoX replacement |
| File / batch IO | Symphonia decode + Arrow/Parquet | datasets, offline processing |
| Realtime | `fluxion-rt`: SPSC ring buffer + CPAL | live audio, plugins, `play`/`record` |
| Graph artifact | `.fxg` frozen-coefficient export | realtime engine, deployment, plugins |

---

## 10. Roadmap (phased; validate risk early)

- **Phase 0 — Spike (de-risk the alpha deps).** One primitive end-to-end: **SOS/biquad cascade** in
  CubeCL with a hand-derived analytic backward, wrapped in Burn `Autodiff`. Prove it runs on NVIDIA
  *and* Apple Metal and that the gradient is correct (oracle: scipy / RBJ cookbook). **Go/No-Go on
  Rust+CubeCL+Burn here.**
- **Phase 1 — Core algebra + batch engine.** Reified graph IR, `|` / `+` operators, the IR passes
  (SOS fusion, coeff-design), the full filter set (Butterworth/Chebyshev/FIR), gain/normalize.
  Plain-transform Python API over DLPack.
- **Phase 2 — Differentiable surface.** `torch.autograd.Function` + `jax.custom_vjp` adapters;
  Burn-native training path; the effect set (reverb/echo/mask) with backward.
- **Phase 3 — CLI.** clap binary with SoX-compatible command surface; Symphonia + Arrow IO; `info`,
  pipeline, glob/batch, `compile`.
- **Phase 4 — Realtime engine.** `fluxion-rt`: SPSC ring buffer, allocation-free executor, frozen
  coeffs, parameter command queue + ramping; CPAL `play`/`record`.
- **Phase 5 — Reach.** WASM/WebGPU; C-ABI hardening; split GPU wheels + cibuildwheel; docs.

---

## 11. Risks & open questions

- **Alpha dependencies.** CubeCL (alpha) and Burn (pre-1.0) are the strategic bet *and* the biggest
  risk. Phase 0 is explicitly a go/no-go. Fallback: C++ + nanobind + hand-written Metal/CUDA/HIP.
- **GPU wheel distribution.** Size/metadata trap (see §7). Solvable but ongoing toil.
- **Stable-Rust CPU SIMD multiversioning** is weaker than C++ Highway; the realtime CPU path may
  need `pulp` or hand intrinsics for runtime ISA dispatch.
- **DDSP backward correctness & IIR stability.** Hand-derived VJPs must be validated against an
  independent oracle; coefficients optimized differentiably must be re-checked for stability before
  freezing into the realtime kernel.
- **DLPack stream/sync footguns.** Honor the CUDA stream handshake or get races; versioned vs
  unversioned DLPack must be handled.
- **Scope.** This is a multi-person, multi-quarter effort. The functional graph IR + lowering is the
  long pole; keep it *small* (do not build a Faust/MLIR-grade compiler).

## 12. Explicitly out of scope (anti-goals)

- ❌ A general-purpose autograd *engine* (own the narrow VJPs, rent the graph).
- ❌ A full Faust/MLIR-grade DSP compiler (reify a *small* IR, lower it).
- ❌ Per-vendor hand-written kernel families (one CubeCL kernel).
- ❌ GPU/autograd inside the audio callback (realtime = CPU-SIMD, frozen coeffs).
- ❌ Copying GPL code from the reference projects (AudioNoise, SoX, sox-extended are GPL — reuse
  *ideas and math*, e.g. the RBJ cookbook, never the source).

---

## 13. References

### Reference projects analysed
- **torvalds/AudioNoise** — toy C guitar-pedal effects (RBJ biquads, phaser, echo, pitch), GPL-2.0,
  single-sample/zero-latency. Use as a *coefficient-correctness oracle* and for its lock-free SPSC
  ring buffer + double-buffered parameter swap. https://github.com/torvalds/AudioNoise
  (sibling hardware: https://github.com/torvalds/GuitarPedal)
- **antirez/ds4 ("DwarfStar")** — MIT, *inference-only* LLM engine (no autograd). Template for a
  small, readable, self-contained C core with one CPU reference fanning out to Metal/CUDA/ROCm
  backends. https://github.com/antirez/ds4
- **matteospanio/sox-extended** — GPL fork of SoX adding `spectrogram -L/-R`. Lesson: mirror SoX's
  command surface, modernise the flags. https://github.com/matteospanio/sox-extended
  · upstream SoX: https://github.com/chirlu/sox
- **matteospanio/torchfx** — the PyTorch-based predecessor this project deliberately un-couples from
  a single framework. https://github.com/matteospanio/torchfx

### Cross-vendor GPU
- CubeCL (one kernel → CUDA/HIP/Metal/Vulkan/WGSL/CPU): https://github.com/tracel-ai/cubecl
- AdaptiveCpp (SYCL): https://github.com/AdaptiveCpp/AdaptiveCpp
- wgpu / WebGPU: https://wgpu.rs/
- NVIDIA cuda-oxide (Rust→PTX): https://nvlabs.github.io/cuda-oxide/appendix/ecosystem.html
- rust-gpu "Rust on every GPU": https://rust-gpu.github.io/blog/2025/07/25/rust-on-every-gpu/

### Differentiable DSP & autodiff
- Burn (backend-agnostic autodiff): https://github.com/tracel-ai/burn · https://crates.io/crates/burn-autodiff
- candle: https://github.com/huggingface/candle
- Enzyme (LLVM AD): https://github.com/EnzymeAD/Enzyme · Rust `std::autodiff` PR: https://github.com/rust-lang/rust/pull/129176
- Differentiable All-Pole Filters (DiffAPF): https://arxiv.org/pdf/2404.07970 · torchlpc: https://github.com/yoyololicon/torchlpc
- Yu & Fazekas, differentiable IIR / `torchaudio.lfilter`: https://arxiv.org/pdf/2308.15422 · https://docs.pytorch.org/audio/stable/generated/torchaudio.functional.lfilter.html
- Magenta DDSP: https://magenta.tensorflow.org/ddsp · NablAFx: https://arxiv.org/pdf/2502.11668
- micrograd (minimal autograd): https://github.com/karpathy/micrograd

### Interop standards
- DLPack via Array API: https://data-apis.org/array-api/2024.12/design_topics/data_interchange.html
- Python Array API 2024/2025: https://data-apis.org/blog/array_api_v2024_release/ · https://data-apis.org/blog/array_api_v2025_release/
- Apache Arrow C Data Interface: https://arrow.apache.org/blog/2020/05/03/introducing-arrow-c-data-interface/ · nanoarrow: https://github.com/apache/arrow-nanoarrow · Arrow DLPack: https://arrow.apache.org/docs/python/dlpack.html
- JAX FFI / custom_vjp: https://docs.jax.dev/en/latest/ffi.html

### Bindings & packaging
- nanobind (the C++ alternative): https://nanobind.readthedocs.io/en/latest/why.html · ndarray/DLPack: https://nanobind.readthedocs.io/en/latest/ndarray.html
- PyO3: https://github.com/pyo3/pyo3 · maturin: https://github.com/PyO3/maturin
- cbindgen: https://github.com/mozilla/cbindgen · UniFFI: https://github.com/mozilla/uniffi-rs
- GPU wheels problem: https://pypackaging-native.github.io/key-issues/gpus/ · cibuildwheel 4.0 CUDA: https://iscinumpy.dev/post/cibuildwheel-4-0-0/ · scikit-build-core: https://scikit-build-core.readthedocs.io/

### Functional DSP & graph-IR lowering
- Faust (functional DSP language, IR → many backends): https://faust.grame.fr/ · backends: https://github.com/grame-cncm/faust/wiki/backends · new IR paper: https://hal.science/hal-03124677v1
- faust-ddsp (forward-mode AD in Faust): https://github.com/hatchjaw/faust-ddsp · Faust→JAX: https://github.com/jax-ml/jax/discussions/13652
- Elementary Audio (functional JS DSP): https://www.elementary.audio/docs
- DSP-MLIR: https://arxiv.org/pdf/2408.11205 · DAC-JAX: https://arxiv.org/pdf/2405.11554
- **fundsp** (functional Rust audio DSP — closest existing prior art; CPU-only, non-differentiable): https://lib.rs/crates/fundsp
- Rust audio ecosystem overview: https://andrewodendaal.com/rust-audio-programming-ecosystem/

### Realtime audio
- Ross Bencina, "Time Waits for Nothing" (the audio-callback rules): http://www.rossbencina.com/code/real-time-audio-programming-101-time-waits-for-nothing
- RTNeural (allocation-free C++ inference): https://github.com/jatinchowdhury18/RTNeural · paper: https://arxiv.org/pdf/2106.03037
- Lock-free SPSC ring buffers: https://github.com/szanni/ringbuf · https://blog.paul.cx/post/a-wait-free-spsc-ringbuffer-for-the-web/
- Soundpipe: https://github.com/PaulBatchelor/Soundpipe · miniaudio: https://miniaud.io/docs/manual/index.html · JUCE AudioProcessor: https://docs.juce.com/master/classAudioProcessor.html
- CPAL (cross-platform Rust audio I/O): https://github.com/RustAudio/cpal · Symphonia (pure-Rust decode): https://github.com/pdeljanov/Symphonia

### Comparable libraries (in-repo references / prior art)
- Spotify pedalboard (C++/JUCE + pybind11, CPU-only): https://github.com/spotify/pedalboard
- GRAFX (PyTorch audio processing graphs): prior art to stay clear of by name.
- FLAMO (differentiable frequency-domain LTI): https://github.com/gdalsanto/flamo

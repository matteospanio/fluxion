# Fluxion

> Differentiable, cross-vendor, framework-agnostic audio DSP with a functional graph API,
> a modern SoX-substitute CLI, and a hard-real-time engine — written in Rust, bound to anything.

**Status: pre-1.0 (0.x).** The core is implemented and tested: the graph algebra, the DSP op set
with hand-derived analytic gradients, Burn-based whole-graph autodiff, CPU SIMD + CUDA batch
kernels, the allocation-free realtime engine, audio IO, the CLI, and the Python package. The API
and the `.fxg` on-disk format may still change before 1.0. Design rationale lives in
[`PROJECT.md`](PROJECT.md); the remaining road to 1.0 in [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md).

One codebase, four ways in:

- **Rust library** — `fluxion` on crates.io: compose effects with `|` (series) and `+`
  (parallel), run them batched on CPU/GPU, differentiate them, or freeze them for realtime.
- **CLI** — `fluxion`, a SoX substitute with named effects and long flags (not SoX's interface,
  most of SoX's jobs).
- **Python** — `fluxion` on PyPI: a torchaudio-style eager API, zero-copy DLPack interop,
  torch/JAX autograd adapters, batched data augmentation.
- **C ABI** — a small panic-safe surface (`fluxion.h`) for C/C++/Swift consumers.

## Install

```bash
# Rust library
cargo add fluxion                 # batch/CPU core (pure Rust, builds offline)
cargo add fluxion -F autodiff     # + whole-graph differentiation through Burn
cargo add fluxion -F realtime     # + the realtime engine re-exports

# CLI
cargo install fluxion-cli                          # file processing, no audio-device deps
cargo install fluxion-cli --features realtime      # + play/record via CPAL

# Python (wheels: Linux x86_64/aarch64, macOS Intel/AS, Windows; numpy is the only hard dep)
pip install fluxion               # extras: [torch] [jax] [interop] (safetensors) [dataset] (parquet)
```

Dependencies are deliberately thin: the default build is pure Rust and compiles offline; heavy
optional stacks (Burn/CubeCL for autodiff/GPU, CPAL for audio devices, PyO3 for Python) only enter
behind the feature flags that need them.

## The graph in 30 seconds

```rust
use fluxion::prelude::*;

// `|` = series, `+` = parallel (outputs summed) — the same algebra everywhere.
let chain = (lowpass(800.0) + highpass(4000.0)) | compand(0.01, 0.1, -20.0, 4.0, 6.0, 0.0) | gain(0.5);

let wet = process(&chain, &signal);              // batch: any channels, allocates freely
```

The same graph lowers to three executors, never mixing their rules:

```rust
// Differentiable (feature `autodiff`): loss.backward() flows through the whole chain —
// every op owns its analytic VJP; Burn provides the tape. Train coefficients or design
// parameters ("learn a cutoff") with stability guaranteed by construction.
let out = fluxion::diff_process::<B>(&chain, x, fs);

// Realtime (feature `realtime`): freeze designed coefficients, then run alloc-free,
// lock-free, bounded-time blocks in the audio callback — no autograd, no GPU, no locks.
let mut rt = fluxion::to_rt_graph(&chain, fs).expect("realtime-lowerable");
rt.prepare(128);
rt.process(&mut block);
```

Geometry operations that change length, channel count, or sample rate (trim, pad, resample,
remix, …) are deliberately *not* graph ops — they live in `fluxion::transform` and run between
graph passes.

## CLI — a SoX substitute

Same philosophy, modern interface: named effects, long `--flags`, explicit units (Hz, seconds,
dB), SI suffixes (`--cutoff 1k`), and a self-describing catalog (`fluxion effects`).

```bash
# Filter a file: adjacent effects fuse into one pass.
fluxion in.wav lowpass --cutoff 1k gain --db -6 out.wav

# The bread-and-butter SoX jobs: trim, resample, bit-depth conversion (TPDF-dithered).
fluxion --bits 16 in.flac trim --start 0.25 --len 30 rate --fs 44100 out.wav

# Multiple inputs: concatenate by default, sum with --mix.
fluxion --mix vocals.wav backing.wav mixed.wav

# Generate, analyze, inspect.
fluxion synth --wave sine --freq 440 --secs 1 fade --fadein 0.1 tone.wav
fluxion stat in.wav            # min/max, peak & RMS dBFS, DC offset, crest factor
fluxion info in.mp3            # metadata for WAV/FLAC/MP3/OGG/… (alias: soxi)

# Unix filter, null sink, dataset batch.
fluxion - reverse - < in.wav > out.wav
fluxion batch out/ 'data/*.wav' highpass --cutoff 80

# Freeze a chain to a portable artifact (stability-certified), play it live.
fluxion compile lowpass --cutoff 800 echo --time 0.3 chain.fxg
fluxion play in.wav chain.fxg          # --features realtime
fluxion record --secs 5 take.wav       # --features realtime
```

**Effects** (graph ops): `gain`, `lowpass`/`highpass` (Butterworth, any order),
`cheby1low`/`cheby1high`/`cheby2low`/`cheby2high`, RBJ `peaking`/`lowshelf`/`highshelf`/
`notch`/`bandpass`/`allpass`, raw `biquad`, `fir --taps …`, `normalize`, `delay`, `echo`,
`reverb`, `fade`, `tremolo`, `overdrive`, `compand`, `reverse`, `chorus`, `flanger`, `phaser`.
**Geometry stages**: `trim`, `pad`, `rate`, `speed`, `repeat`, `silence`, `channels`, `remix`.
Run `fluxion effects [name]` for every parameter, unit, and default.

Not ported from SoX (yet or ever): `tempo`/`pitch` (time-stretch — roadmap), `spectrogram`
(imaging dependency), noise reduction, and legacy niches (`oops`, `riaa`, `earwax`).

## Python

```python
import numpy as np, fluxion
from fluxion import Compose, RandomChain

chain = fluxion.lowpass(8000) | fluxion.gain(0.5)     # same algebra
y  = chain.process(x, fs=48_000)                      # (T,) or (C, T); numpy/torch/jax in via DLPack
ys = chain.process_batch(batch, fs=48_000)            # (B, T) batched

# Data augmentation: stochastic chains, seeded.
aug = Compose([RandomChain(fluxion.lowpass, cutoff=(2_000, 16_000), p=0.8)])
x_aug = aug(x, fs=48_000)

# Dataset IO (extra: fluxion[dataset]) — Parquet, same schema as the Rust side, streamed.
from fluxion.dataset import iter_parquet, write_parquet
write_parquet("out.parquet",                                   # augment a whole dataset, lazily
              ((aug(x, fs), fs) for x, fs in iter_parquet("in.parquet")))

# Training: coefficients as nn.Parameter, analytic backward under torch autograd.
from fluxion.torch import SosModule
mod = SosModule.from_chain(chain, fs=48_000)

# Import a FLAMO-trained SISO biquad cascade (extra: fluxion[interop]).
coeffs = fluxion.interop.load_flamo_sos("checkpoint.safetensors")
```

## Workspace

| Crate | Role | State |
|-------|------|-------|
| `fluxion` (`crates/fluxion-facade`) | Facade + `prelude`; the crate users depend on (features: `autodiff`, `realtime`) | **working** |
| `fluxion-core` | Graph algebra + IR (`\|` series, `+` parallel, `~` feedback), typed op catalog, versioned `.fxg` | **working + tested** |
| `fluxion-ops` | DSP kernels + analytic VJPs, coefficient design, geometry transforms | **working + tested** (SciPy golden-vector oracle) |
| `fluxion-backend` | CPU executor (SIMD batch path), graph lowering, stability certification, CUDA kernels (feature `cuda`) | **working + tested** |
| `fluxion-autodiff` | Burn `Autodiff` integration: whole-graph `diff_process`, trainable coeffs/design params | **working + tested** (feature-gated) |
| `fluxion-rt` | Realtime engine: lock-free SPSC ring, alloc-free executor, CPAL backend (feature `cpal`) | **working + tested** (alloc-asserted) |
| `fluxion-io` | WAV read/write (16/24/32-bit, TPDF dither) + Symphonia decode/probe (FLAC/MP3/OGG/AAC/…), bounded-memory streaming readers, Arrow/Parquet dataset IO (feature `parquet`) | **working + tested** |
| `fluxion-cli` | The `fluxion` binary (feature `realtime` for play/record) | **working + tested** |
| `fluxion-py` | PyO3/maturin package `fluxion` (abi3, numpy-only hard dep; extras: torch/jax/interop) | **working + tested** |
| `fluxion-ffi` | C ABI (`include/fluxion.h`, cbindgen), panic-safe | minimal, tested; `publish = false` |
| `fluxion-wasm` | wasm-bindgen (CPU + WebGPU) | stub, deferred to 1.x; `publish = false` |

GPU status: CUDA forward + backward kernels are implemented and validated on NVIDIA (RTX 3070,
~59× CPU on large batches; benchmarked 1.9× torchfx resident); Apple Metal / AMD ROCm validation
via CubeCL is pending, so GPU stays behind the `cuda` feature for now.

## Develop

```bash
cargo build && cargo test        # whole workspace (offline, pure Rust)
cargo clippy --all-targets       # lints are CI-gated (-D warnings), incl. missing_docs
cargo bench -p fluxion-ops       # Criterion benchmarks
cargo run -p fluxion-cli -- effects
cd crates/fluxion-py && maturin develop && pytest tests/   # Python package
```

See [CONTRIBUTING.md](CONTRIBUTING.md) (and [AGENTS.md](AGENTS.md) for the conventions that
CI enforces).

## License

[MIT](LICENSE). No GPL code from the reference projects (SoX, AudioNoise) — ideas and math only.

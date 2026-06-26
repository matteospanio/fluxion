# Fluxion

> Differentiable, cross-vendor, framework-agnostic audio DSP with a functional graph API,
> a modern SoX-replacement CLI, and a hard-real-time engine — written in Rust, bound to anything.

**Status: scaffold.** This repository currently contains the Cargo workspace backbone and a working
graph-algebra core. See [`PROJECT.md`](PROJECT.md) for the full design, rationale, and roadmap.

## Workspace layout

| Crate | Role | State |
|-------|------|-------|
| `fluxion` (`crates/fluxion-facade`) | Facade + `prelude`; the crate users depend on | skeleton |
| `fluxion-core` | Graph algebra + IR (`\|` series, `+` parallel) | **working + tested** |
| `fluxion-ops` | DSP primitives: analytic forward + backward (VJP) | placeholder |
| `fluxion-backend` | Lowering to CubeCL (CUDA/HIP/Metal/Vulkan/WGSL) + CPU SIMD | placeholder |
| `fluxion-autodiff` | Burn `Autodiff` integration + per-framework VJP adapters | placeholder |
| `fluxion-rt` | Realtime engine: lock-free ring buffer, alloc-free executor | placeholder |
| `fluxion-io` | Symphonia decode + Arrow/Parquet batch IO | placeholder |
| `fluxion-ffi` | Stable C ABI (cbindgen) + DLPack | smoke stub |
| `fluxion-py` | PyO3 bindings + torch/jax autograd adapters | placeholder |
| `fluxion-wasm` | wasm-bindgen (CPU + WebGPU) | placeholder |
| `fluxion-cli` | clap CLI, binary `fluxion` | placeholder |

Heavy / alpha dependencies (Burn, CubeCL, PyO3, clap, Symphonia) are intentionally **not** wired
into the backbone yet — each is added to its crate when that crate is implemented, so the workspace
builds offline today.

## Build

```bash
cargo build              # whole workspace
cargo test               # runs the fluxion-core algebra tests + doctests
cargo run -p fluxion-cli # the (scaffold) CLI, binary name: fluxion
```

## Example (today, in `fluxion-core` / the facade)

```rust
use fluxion::prelude::*;

let chain = lowpass(800.0) | gain(-3.0);          // `|` = series
let eq    = lowpass(800.0) + highpass(80.0);      // `+` = parallel (summed)
assert_eq!(chain.leaf_count(), 2);
```

## License

`MIT OR Apache-2.0` (license texts to be added before first release).

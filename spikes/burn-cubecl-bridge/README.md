# Burn ↔ CubeCL bridge spike

Proves we can run fluxion's CubeCL kernel **directly on a resident Burn tensor**, so the GPU compute
win lands in a training-loop workflow (data stays on the device) rather than being lost to
per-call host↔device transfer.

## Result ✅ (RTX 3070, CUDA 12.4, Burn 0.21 / CubeCL 0.10, 2026-06-26)

```
resident forward, max |GPU-CPU| = 5.960e-8     # bit-accurate
resident GPU forward: 20.16 ms/iter            # vs ~430 ms transfer-bound, ~300 ms CPU
```

So once `x` is resident, each forward is ~20 ms (output alloc + compute), **~21× the one-shot
transfer-bound path** and ~15× a CPU core — the speedup the C4/F2 kernel promised, now realized
because nothing is uploaded per call.

## How the bridge works

- `burn::backend::Cuda` is **fusion-wrapped** by default, which hides the CubeCL tensor. Use the raw
  `CubeBackend<CudaRuntime, f32, i32, u8>` instead — its float primitive is `CubeTensor<R>`.
- `CubeTensor<R>` exposes **public `client` + `handle`**, so the `#[cube]` kernel launches straight on
  the resident buffer; wrap the result back with
  `CubeTensor::new_contiguous(client, device, shape, handle, dtype)`.
- High-level ↔ primitive: `tensor.into_primitive().tensor()` and
  `Tensor::from_primitive(TensorPrimitive::Float(ct))`.
- Generic over the runtime `R`, so the same path lowers to CUDA / ROCm / Metal / Vulkan / WGSL.

## What's proven vs. next

- ✅ **The bridge + resident forward** — kernel on a resident Burn tensor, correct and fast.
- ◻️ **F3** — the analytic backward as resident GPU kernels (adjoint + coeff-grad), so a full
  training loop stays on the device (forward *and* backward). Then wire both into
  `fluxion-autodiff`'s differentiable op for a `CubeBackend<R>` fast path (host roundtrip stays as
  the backend-agnostic fallback).

## Running it

Needs an NVIDIA GPU + CUDA toolkit (`libnvrtc`). Standalone crate, excluded from the main build.

```bash
CUDA_PATH=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH cargo run --release
```

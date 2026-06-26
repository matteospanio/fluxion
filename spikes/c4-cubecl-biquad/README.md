# C4/F2 spike — batched biquad as a CubeCL kernel

Proves we can author the **cross-vendor IIR kernel** the GPU lane rests on: one `#[cube]` kernel,
one thread per batch row running the biquad recurrence over time, lowered by CubeCL to
CUDA / ROCm / Metal / Vulkan / WGSL.

## Result ✅ (RTX 3070, CUDA 12.4, CubeCL 0.10, 2026-06-26)

```
batch=16384 frames=4096  (67 Msamples)
max |GPU-CPU| = 5.960e-8          # bit-accurate vs the CPU kernel (f32 epsilon)
GPU 5.7 ms/iter (warm)   CPU(1 core) 335.7 ms   speedup ~59x
```

A whole DL batch filtered in a single launch, ~59× a single CPU core, correct to f32 precision.
(The first launch's NVRTC JIT compile is excluded by warming up — it dominated the naive timing.)

## What's proven vs. what's next

- ✅ **The kernel** — authoring + launching a CubeCL IIR kernel, correct and fast on NVIDIA.
- ◻️ **Cascade** — a full SOS cascade is this kernel looped over sections (trivial extension).
- ◻️ **C4/F1** — wire it behind a `fluxion-backend` Burn/CubeCL backend over Burn tensors.
- ◻️ **E6/F3** — the analytic backward (`fluxion_ops::sos_vjp`) as the custom gradient, so the
  GPU filter is differentiable without unrolling the recursion. (Burn's custom-backward API —
  the internal `Backward` trait + `OpsPrep` — has been located; see notes in the GPU-box memory.)

Together with `../f0-burn-cuda` (Burn autodiff on CUDA), both halves of the GPU lane are validated.

## Running it

Needs an NVIDIA GPU + CUDA toolkit (`libnvrtc`). Standalone crate, excluded from the main build.

```bash
CUDA_PATH=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH cargo run --release
```

# Burn ↔ CubeCL bridge spike

Proves we can run fluxion's CubeCL kernel **directly on a resident Burn tensor**, so the GPU compute
win lands in a training-loop workflow (data stays on the device) rather than being lost to
per-call host↔device transfer.

## Result ✅ (RTX 3070, CUDA 12.4, Burn 0.21 / CubeCL 0.10, 2026-06-26)

```
forward    max|GPU-CPU| = 5.960e-8   # bit-accurate
adjoint    max|GPU-CPU| = 5.960e-8   # input-gradient kernel == sos_input_grad
resident fwd+bwd: 40.5 ms/iter       # vs ~860 ms if each pass round-tripped the host
coeff-grad max|GPU-CPU| = 1.9e-4     # == sos_vjp grad_coeffs; rel vs finite-diff = 8.9e-3
Burn autograd on GPU: coeff 1.0e-4, input 1.5e-4   # loss.backward() runs the kernels
```

So once `x` is resident, a forward+backward iteration is ~40 ms (two kernel passes + allocs) —
~20× the host-roundtrip path. The compute win the C4/F2 kernel promised is realized because nothing
is uploaded/downloaded per pass.

- **adjoint** (cascade input gradient) = the same recurrence run backward in time, sections in
  reverse — bit-identical to `fluxion_ops::sos_input_grad`.
- **coefficient gradient** = one pass building the all-pole intermediates `w = x/A`, `v = y/A` inline
  and accumulating the five per-coeff sums per row; the cross-row reduction (each coeff's gradient is
  a sum over all rows) is the tiny `[batch,5]` host sum. Matches `sos_vjp` and a finite-difference
  check.

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
- ✅ **Input-gradient (adjoint) kernel** — resident, bit-identical to `sos_input_grad`; resident
  forward+backward ~40 ms/iter.
- ✅ **Coefficient-gradient kernel** (single biquad) — `sos_vjp`'s `grad_coeffs` on device, resident,
  matches CPU + finite-diff.
- ✅ **Burn-autograd integration** (single biquad) — a custom op over `Autodiff<CubeBackend>` whose
  forward + backward launch the kernels on resident tensors; `loss.backward()` on a GPU tensor
  gradchecks vs finite-difference (coeff 1.0e-4, input 1.5e-4). Only the `[5]` coeffs and the
  `[batch,5]` reduction cross the host.
- ✅ **Workspace port** — `fluxion-autodiff/src/cuda.rs` (feature `cuda`) ships `sos_gpu` (fixed
  cascade, input grad) + `biquad_train_gpu` (single trainable biquad); GPU gradcheck tests pass
  (`cargo test -p fluxion-autodiff --features cuda`). The host roundtrip stays the CPU fallback.
- ◻️ **Batched + cascade + cross-vendor** — a `[batch, frames]` entry (the kernels already handle
  `n_rows`), cascade coeff-grad orchestration (compose per-section, no new math), and a generic-`R`
  backend (the kernels are runtime-generic) for ROCm/Metal/WGSL.

## Running it

Needs an NVIDIA GPU + CUDA toolkit (`libnvrtc`). Standalone crate, excluded from the main build.

```bash
CUDA_PATH=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH cargo run --release
```

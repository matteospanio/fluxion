# F0 spike — Burn + CubeCL on CUDA

The **go/no-go gate** for the GPU lane (PROJECT.md §4, IMPLEMENTATION_PLAN epic F).

## Result: GO ✅

Validated 2026-06-26 on an **NVIDIA RTX 3070** (driver 580, CUDA 12.4 toolkit, Burn 0.21):

```
CUDA forward  x*2 = [2.0, 4.0, 6.0, 8.0]
step   0: loss=120.00000 w=0.00000
step  50: loss=0.00000 w=2.00000
...
fitted w = 2.00000 (target 2.0)
>>> GPU FORWARD + AUTODIFF OK <<<
```

So the cross-vendor differentiable substrate (Burn → CubeCL → CUDA/NVRTC) compiles and runs on
NVIDIA, and reverse-mode autodiff works on-device. This de-risks:

- **C4** — implement the `fluxion-backend` Burn backend (CUDA/ROCm/Metal/wgpu via CubeCL),
- **E6** — register the analytic VJPs (`fluxion-ops::biquad_vjp` etc.) as Burn custom backward.

Still pending for the full cross-vendor claim: validation on **Apple Metal** and **AMD ROCm**.

## Running it

Needs an NVIDIA GPU + the CUDA toolkit (`libnvrtc`). This is a standalone crate (its own
`[workspace]`), excluded from the main fluxion build so the workspace stays CPU/offline-buildable.

```bash
CUDA_PATH=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH cargo run
```

# Throughput: fluxion vs torchfx (SOS IIR)

Honest head-to-head on the same hardware, order-8 Butterworth low-pass (4 sections), f32. Filtering
throughput in Msamples/s (higher is better). torchfx = Spotify-grade PyTorch DSP with a fused
C++/CUDA SOS kernel — a strong reference.

## CPU — single core (64 × 48 000)

Measured 2026-06-26 (this dev box, 1 thread both sides, `-C target-cpu=native` for fluxion):

| path | Msamples/s | vs torchfx |
|---|--:|--:|
| **fluxion `sos_filter_interleaved`** (SIMD across batch, frame-major) | **665–884** | **1.4–1.9×** |
| torchfx (fused C++) | 465 | 1.0× |
| fluxion `sos_filter_interleaved` + blocked transpose (planar in/out) | ~485 | ~1.0× |
| fluxion `sos_filter` (scalar per-row, the old reference) | 85 | 0.18× |

An IIR can't vectorize over time (sequential), but channels are independent — so a frame-major
(channel-interleaved) layout lets the inner per-channel loop auto-vectorize across the batch. That's
the win. On torch-style **planar** `(B,T)` input fluxion needs a transpose; a cache-blocked one ≈
ties torchfx, a naïve strided one loses (~100).

## GPU — RTX 3070 (16384 × 4096 = 67 Msamples)

Measured 2026-06-26 on alienware (CUDA 12.4, CubeCL 0.10 / torch 2.10 cu128):

| regime | fluxion | torchfx | |
|---|--:|--:|---|
| **resident** (data on GPU — the training / DSP-pipeline case) | **2962** | 1580 | **fluxion 1.9×** |
| one-shot transfer (upload + download per call — "filter a file") | 187 | 367 | torchfx 2.0× |

- **Resident kernel: fluxion's CubeCL kernel is ~1.9× faster** than torchfx's CUDA kernel — the
  regime that matters for differentiable training and resident DSP pipelines.
- **One-shot transfer: torchfx is ~2× faster** — fluxion's H2D/D2H is naïve synchronous,
  unpinned; torch uses pinned-memory / overlapped transfers. A known, bounded optimization.

## Takeaways

- fluxion's *kernels* (CPU SIMD and GPU) are **faster** than torchfx's once the data is in the right
  layout / resident — which is fluxion's bet (efficient cross-vendor kernels + differentiability).
- fluxion's *data movement* (CPU planar transpose, GPU host↔device transfer) is **unoptimized** and
  is where torchfx wins one-shot. Both are scoped follow-ups (blocked transpose; pinned/async copies).

## Reproduce

```bash
# GPU (needs NVIDIA + CUDA toolkit):
CUDA_PATH=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH cargo run --release
python bench_torchfx.py     # in a CUDA-built torchfx venv

# CPU fluxion: a standalone crate depending on fluxion-ops, timing sos_filter_interleaved on a
# [frames*channels] buffer (built with RUSTFLAGS="-C target-cpu=native").
```

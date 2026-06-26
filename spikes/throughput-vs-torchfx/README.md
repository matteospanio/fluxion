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
- **One-shot transfer: torchfx is ~2× faster.** Diagnosed (2026-06-27, see below) — it is **not** a
  pinned-memory issue: it's CubeCL 0.10's host→device *upload* path.

### Transfer diagnosis (one-shot)

Split the 268 MB-each-way round trip:

| | bandwidth |
|---|--:|
| D2H `read_one` (download) | 6.3 GB/s — fine |
| H2D `create_from_slice` (upload) | **0.9 GB/s — the bottleneck** |
| H2D `create(Bytes::from_elems)` | 0.91 GB/s |
| H2D **pinned** (`staging` + `create`) | 0.84 GB/s — **no better** |

So pinning does **not** help (tested directly). Two real costs: a per-call fresh 268 MB host buffer
(`create_from_slice` → internal `to_vec`, first-touch page faults ≈ 83 ms) and CubeCL's actual H2D
copy (~1.3 GB/s after subtracting the alloc — vs 6.3 GB/s D2H). cubecl 0.10's public API exposes no
way to reuse an upload buffer or a faster H2D path, so this is an **upstream CubeCL limitation**, not
a quick fluxion fix. The download path is already fast. Until then, the **resident path (1.9×) is the
recommended GPU regime** — and it's the one differentiable training and DSP pipelines actually use.

## Takeaways

- fluxion's *kernels* (CPU SIMD and GPU) are **faster** than torchfx's once the data is in the right
  layout / resident — which is fluxion's bet (efficient cross-vendor kernels + differentiability).
- fluxion's *data movement* is where torchfx wins one-shot: the CPU planar transpose (a blocked one
  ≈ ties — a fluxion-side fix) and the GPU H2D upload (a **CubeCL upstream** limitation; pinned
  memory does not help — see the transfer diagnosis above). Neither affects the resident regime.

## Reproduce

```bash
# GPU (needs NVIDIA + CUDA toolkit):
CUDA_PATH=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH cargo run --release
python bench_torchfx.py     # in a CUDA-built torchfx venv

# CPU fluxion: a standalone crate depending on fluxion-ops, timing sos_filter_interleaved on a
# [frames*channels] buffer (built with RUSTFLAGS="-C target-cpu=native").
```

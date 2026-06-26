"""torchfx reference throughput (CPU + GPU), matched to the fluxion benches.

Run in a torchfx venv (torch + a CUDA-built torchfx for the GPU rows):
    python bench_torchfx.py
"""
import time
import torch
from torchfx.filter import LoButterworth


def bench(name, fn, msamp, k=50, sync=False):
    fn()  # warmup (filter design + JIT)
    if sync:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(k):
        fn()
    if sync:
        torch.cuda.synchronize()
    el = (time.perf_counter() - t0) / k
    print(f"{name:34}: {el*1000:6.3f} ms   {msamp/el:7.0f} Msamples/s")


# ---- CPU: single core, 64 x 48000, order 8 ----
torch.set_num_threads(1)
B, T = 64, 48_000
msamp = B * T / 1e6
x = torch.rand(B, T) * 2 - 1
f = LoButterworth(cutoff=4000, order=8, fs=48000)
bench("torchfx CPU 1-thread", lambda: f(x), msamp)

# ---- GPU: 16384 x 4096 = 67 Msamples, order 8 ----
if torch.cuda.is_available():
    B, T = 16384, 4096
    msamp = B * T / 1e6
    f = LoButterworth(cutoff=4000, order=8, fs=48000)
    xg = (torch.rand(B, T) * 2 - 1).cuda()
    bench("torchfx GPU resident", lambda: f(xg), msamp, sync=True)
    xc = torch.rand(B, T) * 2 - 1
    bench("torchfx GPU one-shot transfer", lambda: f(xc.cuda()).cpu(), msamp, sync=True)

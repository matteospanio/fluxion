# fluxion

GPU-accelerated, differentiable audio DSP — Python bindings for [Fluxion](https://github.com/matteospanio/fluxion).

Build effect chains out of `fluxion.filter` / `fluxion.effect` classes, compose them with `|`
(series) and `+` (parallel), and apply them to a `Wave` or to any NumPy / PyTorch / JAX CPU array
(zero-copy via DLPack when it's already `float32` + contiguous). The differentiable SOS primitives
expose fluxion's analytic gradients to `torch.autograd` and `jax.custom_vjp`, so filter
coefficients train inside your model.

Every op has the same name here, in the Rust library, in the CLI and in the browser — they come
from one registry. The chain can also be written as text and passed between them:
`fluxion.chain("highpass(80, 4) | gain(-3dB)")`.

## Install

```bash
pip install fluxion               # CPU wheel (numpy only)
pip install 'fluxion[torch]'      # + the torch.autograd adapter
pip install 'fluxion[jax]'        # + the jax.custom_vjp adapter
pip install 'fluxion[interop]'    # + safetensors, for load_flamo_sos
pip install 'fluxion[dataset]'    # + pyarrow, for the Parquet dataset IO
```

The default wheel is CPU-only. A CUDA build (the "GPU wheel", extra batch kernels behind
`fluxion.cuda_available()`) is published as a separate `+cu12` local-version artifact — it is not on
PyPI (local versions can't be uploaded); grab it from the project's GitHub releases.

## Quick start

**A file in, a file out** — `Wave` carries the sample rate, so it stays out of your code:

```python
import fluxion as fx

wave = fx.Wave.from_file("in.wav")
(wave | fx.filter.Highpass(80, order=4) | fx.effect.Gain(fx.db(-3))).save("out.wav")
```

**Arrays** — for notebooks and batch jobs, a chain is simply callable:

```python
import numpy as np
import fluxion as fx

x = np.random.default_rng(0).standard_normal(48_000).astype(np.float32)
chain = fx.filter.Lowpass(2000.0, order=4) | fx.effect.Gain(0.8)   # series
y = chain(x, fs=48_000)                                            # (T,) float32 out
batch = chain.process_batch(np.stack([x, x]), fs=48_000)           # (B, T) batched CPU path
```

**Trainable filter** — a designed cascade as a torch `nn.Module`:

```python
import torch
import fluxion as fx
import fluxion.torch as fxt

m = fxt.SosModule.from_chain(fx.filter.Lowpass(2000.0, 4), fs=48_000)  # seed from a design
y = m(torch.randn(512))            # differentiable; m.coeffs is an nn.Parameter
y.pow(2).sum().backward()          # gradients flow through fluxion's analytic IIR adjoint
```

**Augmentation** — stochastic, reproducible transforms:

```python
import numpy as np
import fluxion as fx
from fluxion.augment import Compose, RandomChain

aug = Compose([
    RandomChain(fx.filter.Lowpass, cutoff=(500.0, 8000.0), order=4, p=0.5, rng=0),
    fx.effect.Gain(0.9),
])
y = aug(np.random.default_rng(0).standard_normal(1000).astype(np.float32), fs=48_000)
```

**Dataset IO** — Parquet audio datasets (extra `fluxion[dataset]`), the same on-disk schema as the
Rust `fluxion_io::arrow` side, so files interoperate. Streams both ways, so augmenting a whole
dataset is bounded-memory:

```python
from fluxion.dataset import iter_parquet, write_parquet
write_parquet("out.parquet", ((aug(x, fs), fs) for x, fs in iter_parquet("in.parquet")))
```

Import a FLAMO/DDSP biquad checkpoint into fluxion's SOS path with
`fluxion.interop.load_flamo_sos(...)` (see its docstring for supported layouts).

## License

MIT — see [LICENSE](LICENSE).

# fluxion (browser)

Differentiable, cross-vendor audio DSP in the browser — the same engine as the
[Rust library](https://github.com/matteospanio/fluxion), the CLI and the Python package, compiled
to WebAssembly. Preview and export come from the same DSP, because it *is* the same DSP: the
output is checked against the native library sample for sample in CI.

```bash
npm install fluxion
```

```js
import init, { Chain } from "fluxion";

await init();

const chain = Chain.fromText("highpass(80, 4) | gain(-3dB)");
const out = chain.process(samples, 48000);   // Float32Array in, Float32Array out
```

`init()` loads the WebAssembly — in a browser it fetches the `.wasm` beside the module; in Node,
hand it the bytes:

```js
await init({ module_or_path: await readFile("node_modules/fluxion/pkg/fluxion_wasm_bg.wasm") });
```

## The chain syntax

One string describes a chain, and every fluxion interface reads it the same way — so a chain built
in a browser can be pasted into the CLI, into Python, or into a C host unchanged.

```
highpass(80, 4) | gain(-3dB)      // `|` series, `+` parallel
lowpass=1k                        // shorthand; `k` is x1000
compand(threshold=-24, ratio=8)   // named parameters, the rest take their defaults
```

Full grammar: [docs/chain-syntax.md](https://github.com/matteospanio/fluxion/blob/main/docs/chain-syntax.md).
`chain.toText()` prints the canonical form back, and it reparses.

## The op catalog

27 ops, with names, parameters, defaults and ranges, generated from the same registry the rest of
fluxion uses:

```js
import { OPS, OP_NAMES } from "fluxion/ops";

OPS.lowpass.params;   // [{ name: "cutoff", unit: "hz", default: 1000, min: 0, max: null }, ...]
```

TypeScript gets `OpName` as a literal union, so a misspelled op is a compile error rather than a
runtime throw. Reference: [docs/ops.md](https://github.com/matteospanio/fluxion/blob/main/docs/ops.md).

## Errors

A bad chain throws with the position marked and, where the mistake looks like a typo, the fix:

```
error: unknown op 'hipass'
  hipass(80)
  ^^^^^^ did you mean 'highpass'?
```

## Not yet

`Chain` renders **offline** — enough for waveforms, previews and bounces. Live playback through an
AudioWorklet is roadmap task W4; it extends this same class rather than replacing it. The WebGPU
path is W8.

## Building from source

```bash
./scripts/build-wasm.sh          # from the repository root
node crates/fluxion-wasm/js/quickstart.mjs
```

MIT.

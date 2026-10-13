# fluxion (browser)

Differentiable, cross-vendor audio DSP in the browser — the same engine as the
[Rust library](https://github.com/matteospanio/fluxion), the CLI and the Python package, compiled
to WebAssembly. Preview and export come from the same DSP, because it *is* the same DSP: the
output is checked against the native library sample for sample in CI.

It is on GitHub Packages, whose npm registry asks for a token even to install a public package —
a personal access token (classic) with `read:packages`. So the project's `.npmrc` points the scope
there first:

```ini
@matteospanio:registry=https://npm.pkg.github.com
//npm.pkg.github.com/:_authToken=${GITHUB_TOKEN}
```

```bash
npm install @matteospanio/fluxion-dsp
```

```js
import init, { Chain } from "@matteospanio/fluxion-dsp";

await init();

const chain = Chain.fromText("highpass(80, 4) | gain(-3dB)");
const out = chain.process(samples, 48000);   // Float32Array in, Float32Array out
```

`init()` loads the WebAssembly — in a browser it fetches the `.wasm` beside the module; in Node,
hand it the bytes:

```js
await init({ module_or_path: await readFile("node_modules/@matteospanio/fluxion-dsp/pkg/fluxion_wasm_bg.wasm") });
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

## Frequency response

An EQ editor draws the curve from fluxion's own filter design, not a JS copy of it.
`chain.sosCoeffs(fs)` returns the designed sections as one `Float32Array`, five numbers each —
`b0, b1, b2, a1, a2`, with `a0 = 1` — the same layout and values as Python's `sos_coeffs`:

```js
function magnitude(sos, f, fs) {            // |H(e^jw)| at f Hz
  const w = (2 * Math.PI * f) / fs, [c1, s1, c2, s2] = [Math.cos(w), Math.sin(w), Math.cos(2 * w), Math.sin(2 * w)];
  let mag = 1;
  for (let i = 0; i < sos.length; i += 5) {
    const [b0, b1, b2, a1, a2] = sos.subarray(i, i + 5);
    mag *= Math.hypot(b0 + b1 * c1 + b2 * c2, b1 * s1 + b2 * s2) / Math.hypot(1 + a1 * c1 + a2 * c2, a1 * s1 + a2 * s2);
  }
  return mag;
}
const db = 20 * Math.log10(magnitude(Chain.fromText("peaking(3000, 6, 1.5)").sosCoeffs(48000), 3000, 48000)); // 6.0
```

It returns `undefined` for a chain that is not one filter cascade — `gain` included, which is not
folded in — so draw one-op chains per band and add the gain yourself. `name:` labels are ignored.
`chain.certify(fs)` returns `{ verdict, margin }`, Python's `certify` as an object: refuse
`"unstable"` (a hand-typed `biquad` can be), or anything but `"certified-stable"` /
`"marginally-stable"` to be strict. CI checks both against the native library, and the curve
against the DFT of an impulse through `process`.

## Live playback

The same chain text, on the audio thread — playing audio the page hands it, or as an insert
effect in a Web Audio graph.

```js
import { attachWorklet } from "fluxion/worklet";

const ctx = new AudioContext();
const node = await attachWorklet(ctx, "highpass(80, 4) | gain(-3dB)");
node.connect(ctx.destination);

node.port.postMessage({ type: "audio", samples });          // feed it
node.port.postMessage({ type: "gain", value: 0.2, rampMs: 30 });  // ramped, not stepped
```

Audio moves through a lock-free ring, so the page and the audio thread never wait on each other.
Once started, the wasm side allocates nothing — asserted, not assumed: the module counts its own
allocations and the test requires the count not to move across five seconds of playback. A
parameter change is a ramp, because a gain that jumps between blocks is a click.

### As an insert effect

```js
const fx = await attachWorklet(ctx, "highpass(80) | reverb(0.5)", { input: true, channels: 2 });
source.connect(fx).connect(ctx.destination);
source.start(when);
```

`input: true` gives the node an input, and each 128-frame quantum is processed as it arrives: no
ring and **no added latency**, so tracks whose sources start at the same `when` stay
sample-aligned through their chains. `channels: N` runs N independent copies of the chain, one per
channel, as the offline engine does; whatever is connected is first mixed to N channels by the Web
Audio rules. (`channels` works in ring mode too: post one `Float32Array` per channel, or one for
all of them.) Build as many nodes on a context as there are tracks — the processor registers once
and they all share one wasm instance.

Change the chain while it plays:

```js
fx.port.postMessage({ type: "chain", chain: "lowpass(800) | reverb(0.5)", crossfadeMs: 20 });
fx.port.onmessage = ({ data }) => {
  if (data.type === "error") console.warn(data.chain, data.message); // the old chain plays on
};
```

The new chain starts clean and hears the same input as the old one while a linear crossfade moves
between them. Every chain gets one answer: `{ type: "chain", chain, ok: true }`, or
`{ type: "error", chain, message }` for one that cannot run — a typo, or `reverse`, which needs the
whole signal — and then the old chain keeps playing. A node whose *first* chain cannot run passes
audio through and says so the same way. An empty chain is a pass-through, so removing every effect
needs no rewiring. On stop or seek, `{ type: "reset" }` clears what the chains remember — filter
state, delay lines, reverb tails — so nothing from the old position leaks into the new one.

Types for the options and for every message, both ways, are in `worklet-loader.d.ts`.
`crates/fluxion-wasm/js/demo.html` is a working page.

Playing offline instead — for a waveform, or a bounce — is the `Chain` API above. The two produce
identical samples, which is checked in CI both in Node and inside a browser, in both modes.

## Not yet

Loading a frozen `.fxg` graph in the browser is roadmap task W3; the WebGPU path is W8.

## Building from source

```bash
./scripts/build-wasm.sh          # from the repository root
node crates/fluxion-wasm/js/quickstart.mjs
```

MIT.

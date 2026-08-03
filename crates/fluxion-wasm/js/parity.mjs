// wasm renders what native renders — the roadmap's W2 check.
//
// The fixture holds an input and the output native produced for it (written by
// `crates/fluxion-wasm/tests/parity.rs`, and re-verified against native on every `cargo test`).
// Here the same input goes through the built wasm module and the two are compared. Because the
// reference is committed, this side needs no Rust toolchain.
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

import init, { Chain, ops, version } from "./pkg/fluxion_wasm.js";

const url = (p) => fileURLToPath(new URL(p, import.meta.url));
await init({ module_or_path: await readFile(url("./pkg/fluxion_wasm_bg.wasm")) });

const fixture = JSON.parse(await readFile(url("./fixtures/parity.json"), "utf8"));
const { chain: text, fs, tolerance, input, expected } = fixture;

const chain = Chain.fromText(text);
const actual = chain.process(Float32Array.from(input), fs);

if (actual.length !== expected.length) {
  throw new Error(`length ${actual.length}, expected ${expected.length}`);
}

let worst = 0;
let worstAt = -1;
for (let i = 0; i < expected.length; i++) {
  const diff = Math.abs(actual[i] - expected[i]);
  if (diff > worst) {
    worst = diff;
    worstAt = i;
  }
}
if (!(worst <= tolerance)) {
  throw new Error(
    `wasm diverged from native at sample ${worstAt}: ` +
      `${actual[worstAt]} vs ${expected[worstAt]} (${worst} > ${tolerance})`,
  );
}

// The chain text is the same canonical form every other interface prints, and it reparses.
const canonical = chain.toText();
if (Chain.fromText(canonical).toText() !== canonical) {
  throw new Error(`chain text does not round-trip: ${canonical}`);
}
if (chain.opCount() !== 2) {
  throw new Error(`expected 2 ops, got ${chain.opCount()}`);
}

// Every op is reachable from its bare name, the same coverage check C makes.
for (const name of ops()) {
  const one = Chain.fromText(name);
  if (one.opCount() !== 1) throw new Error(`op '${name}' did not build a single-op chain`);
}

// A bad chain throws with the caret rendering rather than failing silently.
let threw = false;
try {
  Chain.fromText("hipass(80)");
} catch (e) {
  threw = /did you mean 'highpass'/.test(String(e.message ?? e));
}
if (!threw) throw new Error("a misspelled op should throw with a suggestion");

console.log(
  `fluxion wasm parity OK (v${version()}, ${ops().length} ops, ` +
    `worst diff ${worst.toExponential(2)} <= ${tolerance})`,
);

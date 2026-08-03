// wasm renders what native renders, for every op — the roadmap's W6, and the last piece of F-M1.
//
// `crates/fluxion-wasm/tests/parity.rs` owns the case list and renders each one natively into
// reference.json; this feeds the identical input through the built wasm module and compares. Run
// both, in order:
//
//   cargo test -p fluxion-wasm --test parity -- --ignored write_reference
//   node crates/fluxion-wasm/js/parity.mjs
//
// Every case is compared before anything is reported, so one run tells you every op that diverged
// rather than the first.
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

import init, { Chain, ops, version } from "./index.mjs";

const url = (p) => fileURLToPath(new URL(p, import.meta.url));

let reference;
try {
  reference = JSON.parse(await readFile(url("./reference.json"), "utf8"));
} catch (e) {
  throw new Error(
    "reference.json is missing — generate it first:\n" +
      "  cargo test -p fluxion-wasm --test parity -- --ignored write_reference\n" +
      `(${e.message})`,
  );
}

await init({ module_or_path: await readFile(url("./pkg/fluxion_wasm_bg.wasm")) });

const { fs, frames, tolerance, cases } = reference;
const input = Float32Array.from(reference.input);

if (input.length !== frames) {
  throw new Error(`reference input is ${input.length} frames, header says ${frames}`);
}

/** Worst absolute difference, and where it is. */
function compare(actual, expected) {
  if (actual.length !== expected.length) {
    return { worst: Infinity, at: -1, note: `length ${actual.length} vs ${expected.length}` };
  }
  let worst = 0;
  let at = -1;
  for (let i = 0; i < expected.length; i++) {
    const diff = Math.abs(actual[i] - expected[i]);
    if (diff > worst) {
      worst = diff;
      at = i;
    }
  }
  return { worst, at };
}

const results = [];
for (const { name, chain, expected } of cases) {
  const actual = Chain.fromText(chain).process(input, fs);
  results.push({ name, chain, ...compare(actual, expected), actual, expected });
}

const failed = results.filter((r) => !(r.worst <= tolerance));
if (failed.length > 0) {
  console.error(`wasm diverged from native on ${failed.length} of ${results.length} cases:\n`);
  for (const r of failed) {
    const detail =
      r.note ?? `worst ${r.worst.toExponential(3)} at sample ${r.at} ` +
      `(wasm ${r.actual[r.at]}, native ${r.expected[r.at]})`;
    console.error(`  ${r.name.padEnd(18)} ${detail}\n    ${r.chain}`);
  }
  console.error(`\ntolerance is ${tolerance}`);
  process.exit(1);
}

// What it actually measures, reported rather than assumed — a suite that silently loosened would
// still print "ok".
const worst = results.reduce((m, r) => Math.max(m, r.worst), 0);
const exact = results.filter((r) => r.worst === 0).length;

// The coverage the other doors check too: every op reachable from its bare name.
for (const name of ops()) {
  if (Chain.fromText(name).opCount() !== 1) {
    throw new Error(`op '${name}' did not build a single-op chain`);
  }
}

// The chain text round-trips, and a typo is refused with a suggestion — same as everywhere else.
const sample = Chain.fromText(cases[0].chain);
if (Chain.fromText(sample.toText()).toText() !== sample.toText()) {
  throw new Error(`chain text does not round-trip: ${sample.toText()}`);
}
let threw = false;
try {
  Chain.fromText("hipass(80)");
} catch (e) {
  threw = /did you mean 'highpass'/.test(String(e.message ?? e));
}
if (!threw) throw new Error("a misspelled op should throw with a suggestion");

console.log(
  `fluxion wasm parity OK (v${version()}) — ${results.length} cases covering ${ops().length} ops, ` +
    `${frames} frames each\n` +
    `  ${exact}/${results.length} bit-identical to native, worst difference ` +
    `${worst.toExponential(2)} (tolerance ${tolerance})`,
);

// Name the ones that are not exact. They are the ops that call a transcendental per sample, where
// wasm's libm and the platform's differ in the last bit — see the tolerance table in parity.rs.
// Printing them keeps that claim checkable instead of a comment that drifts.
const inexact = results.filter((r) => r.worst > 0).sort((a, b) => b.worst - a.worst);
if (inexact.length > 0) {
  console.log(
    `  not bit-identical (per-sample libm): ` +
      inexact.map((r) => `${r.name} ${r.worst.toExponential(1)}`).join(", "),
  );
}

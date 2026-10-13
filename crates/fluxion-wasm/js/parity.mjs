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
for (const c of cases) {
  const actual = Chain.fromText(c.chain).process(input, fs);
  // A case may name its own bound; see `tolerance_for` in parity.rs for the one that does.
  const bound = c.tolerance ?? tolerance;
  results.push({ ...c, bound, ...compare(actual, c.expected), actual });
}

const failed = results.filter((r) => !(r.worst <= r.bound));
if (failed.length > 0) {
  console.error(`wasm diverged from native on ${failed.length} of ${results.length} cases:\n`);
  for (const r of failed) {
    const detail =
      r.note ?? `worst ${r.worst.toExponential(3)} at sample ${r.at} ` +
      `(wasm ${r.actual[r.at]}, native ${r.expected[r.at]})`;
    console.error(`  ${r.name.padEnd(18)} ${detail}\n    ${r.chain} (tolerance ${r.bound})`);
  }
  process.exit(1);
}

// What it actually measures, reported rather than assumed — a suite that silently loosened would
// still print "ok".
// Reported as a fraction of each case's own bound, since one case sets its own — a raw worst next
// to the shared tolerance would read as a near-miss when it is not being measured against it.
const worst = results.reduce((m, r) => Math.max(m, r.worst / r.bound), 0);
const exact = results.filter((r) => r.worst === 0).length;
const custom = results.filter((r) => r.bound !== tolerance);

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
    `  ${exact}/${results.length} bit-identical to native, worst case uses ` +
    `${(worst * 100).toFixed(0)}% of its tolerance (shared bound ${tolerance}` +
    `${custom.map((r) => `, ${r.name} ${r.bound}`).join("")})`,
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

// --- Filter coefficients (roadmap W9) --------------------------------------------------------------
//
// An EQ editor draws a chain's response from `sosCoeffs` and refuses it on `certify`. Both have to
// be fluxion's own: a curve from a JS copy of the RBJ formulas is a curve of a filter fluxion does
// not render.

/** |H(e^jω)| of a flat `[b0, b1, b2, a1, a2]`-per-section cascade at `f` Hz — the README example. */
function sosMagnitude(sos, f, fs) {
  const w = (2 * Math.PI * f) / fs;
  const [c1, s1, c2, s2] = [Math.cos(w), Math.sin(w), Math.cos(2 * w), Math.sin(2 * w)];
  let mag = 1;
  for (let i = 0; i < sos.length; i += 5) {
    const [b0, b1, b2, a1, a2] = sos.subarray(i, i + 5);
    mag *= Math.hypot(b0 + b1 * c1 + b2 * c2, b1 * s1 + b2 * s2) /
      Math.hypot(1 + a1 * c1 + a2 * c2, a1 * s1 + a2 * s2);
  }
  return mag;
}

/** |DFT| of `h` at `f` Hz, straight from the sum — no FFT, so no bin to land a frequency on. */
function dftMagnitude(h, f, fs) {
  const w = (2 * Math.PI * f) / fs;
  let re = 0;
  let im = 0;
  for (let n = 0; n < h.length; n++) {
    re += h[n] * Math.cos(w * n);
    im -= h[n] * Math.sin(w * n);
  }
  return Math.hypot(re, im);
}

// Where the curve is drawn: across the audio band, and on each case's own corner or centre.
const RESPONSE_HZ = [50, 500, 1000, 2000, 3000, 5000, 8000, 12000, 20000];
// |H| from the sections against the DFT of `process` on a one-second impulse. The two differ by
// the f32 recursion `process` runs (the sections are evaluated here in f64); the tail cut off at
// one second is not a factor, the response being below 1e-44 after half of it. Measured: at most
// 6.1e-6 relative wherever |H| is above -60 dB (the Chebyshev's passband at 50 Hz, its poles the
// closest to the unit circle), and at most 1.3e-8 absolute in any stopband, where a ratio of two
// numbers near -140 dB is noise over noise. 1e-4 relative (0.001 dB, finer than an EQ display
// draws) plus a -120 dB floor leaves 8x headroom over the tightest point; the run prints how much
// it used.
const RESPONSE_REL = 1e-4;
const RESPONSE_ABS = 1e-6;

const impulse = new Float32Array(fs);
impulse[0] = 1;
let worstResponse = 0;
for (const c of reference.sos) {
  const chain = Chain.fromText(c.chain);
  const actual = chain.sosCoeffs(fs);
  if (!(actual instanceof Float32Array) || actual.length !== c.expected.length) {
    throw new Error(`${c.chain}: sosCoeffs gave ${actual?.length} values, native ${c.expected.length}`);
  }
  // Exact: the same design, the same f32 cast. fround brings the JSON's shortest decimal back to
  // the f32 it was printed from.
  const off = c.expected.findIndex((v, i) => actual[i] !== Math.fround(v));
  if (off !== -1) {
    throw new Error(
      `${c.chain}: sosCoeffs[${off}] is ${actual[off]}, native graph_to_sos gives ${c.expected[off]}`,
    );
  }
  // A `name:` label is transparent, as it is to graph_to_sos.
  const labeled = Chain.fromText(`band: ${c.chain}`).sosCoeffs(fs);
  if (!labeled || labeled.some((v, i) => v !== actual[i])) {
    throw new Error(`${c.chain}: a label changed the sections`);
  }
  const h = chain.process(impulse, fs);
  for (const f of RESPONSE_HZ) {
    const want = sosMagnitude(actual, f, fs);
    const got = dftMagnitude(h, f, fs);
    const err = Math.abs(got - want);
    if (!(err <= RESPONSE_REL * want + RESPONSE_ABS)) {
      throw new Error(`${c.chain} at ${f} Hz: |H| from sosCoeffs ${want}, impulse DFT ${got}`);
    }
    worstResponse = Math.max(worstResponse, err / (RESPONSE_REL * want + RESPONSE_ABS));
  }
  const cert = chain.certify(fs);
  if (cert.verdict !== "certified-stable" || !(cert.margin > 0)) {
    throw new Error(`${c.chain} should certify stable, got ${JSON.stringify(cert)}`);
  }
}

// Poles at radius sqrt(1.2): unstable, and the margin is 1 - that radius, as in Python.
const unstable = Chain.fromText("biquad(b0=1, a1=-2.1, a2=1.2)").certify(fs);
if (unstable.verdict !== "unstable" || !(Math.abs(unstable.margin - (1 - Math.sqrt(1.2))) < 1e-6)) {
  throw new Error(`biquad(b0=1, a1=-2.1, a2=1.2) should be unstable, got ${JSON.stringify(unstable)}`);
}
// Not one cascade: no sections. `gain` included — graph_to_sos does not fold it in.
for (const text of ["compand", "peaking(3000, 6, 1.5) | gain(-3dB)", "lowpass(500) + highpass(3000)"]) {
  const sections = Chain.fromText(text).sosCoeffs(fs);
  if (sections !== undefined) {
    throw new Error(`\`${text}\` is not a filter cascade, yet sosCoeffs gave ${sections}`);
  }
}
// certify still covers the whole chain: a compander has no poles and nothing that recirculates.
if (Chain.fromText("compand").certify(fs).verdict !== "certified-stable") {
  throw new Error("compand should certify stable");
}

console.log(
  `  sosCoeffs: ${reference.sos.length} chains exact to native graph_to_sos; |H| vs impulse DFT ` +
    `at ${RESPONSE_HZ.length} frequencies uses ${(worstResponse * 100).toFixed(0)}% of its ` +
    `tolerance (${RESPONSE_REL} relative + ${RESPONSE_ABS})`,
);

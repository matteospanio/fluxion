// Load the built module in Node and call into it — the check that the wasm build works at all
// (roadmap W1), run by the `wasm` CI job.
//
// `--target web` means `init()` wants the wasm bytes; in a browser it fetches them, here we read
// the file. One artifact, both environments — nothing browser-specific to maintain separately.
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

import init, { Chain, ensureFs, version } from "./pkg/fluxion_wasm.js";

const wasm = fileURLToPath(new URL("./pkg/fluxion_wasm_bg.wasm", import.meta.url));
await init({ module_or_path: await readFile(wasm) });

const v = version();
if (!/^\d+\.\d+\.\d+/.test(v)) {
  throw new Error(`version() returned ${JSON.stringify(v)}, which is not a version`);
}

// The project-rate helper (roadmap R2): a page decodes at the file's rate and plays at the
// AudioContext's. Frame count is the one thing a caller computes for itself, so it has to be exact.
const tone = Float32Array.from({ length: 48_000 }, (_, i) =>
  Math.sin((2 * Math.PI * 1000 * i) / 48_000),
);
const at44k = ensureFs(tone, 48_000, 44_100);
if (at44k.length !== 44_100) {
  throw new Error(`ensureFs gave ${at44k.length} frames at 44.1 kHz, expected 44100`);
}
// A 1 kHz tone is still a 1 kHz tone: compare against the sine the new rate implies, away from the
// ends where the filter tapers.
let worst = 0;
for (let i = 2000; i < at44k.length - 2000; i++) {
  worst = Math.max(worst, Math.abs(at44k[i] - Math.sin((2 * Math.PI * 1000 * i) / 44_100)));
}
if (!(worst < 1e-4)) {
  throw new Error(`ensureFs moved the tone: worst sample off by ${worst}`);
}
if (ensureFs(tone, 48_000, 48_000).length !== tone.length) {
  throw new Error("ensureFs at a matched rate should hand the samples back");
}

// Side inputs (roadmap S1): the same chain text as every other interface, and the key is what
// decides. A loud programme with a silent key has to come out shut.
const gated = Chain.fromText("gate(-40, 60, 0.001, 0, 0.005) < side(0)");
if (gated.sideInputs() !== 1) {
  throw new Error(`the chain reads ${gated.sideInputs()} side inputs, expected 1`);
}
const silentKey = new Float32Array(tone.length);
const shut = gated.processWith(tone, 48_000, [silentKey]);
const shutPeak = shut.slice(24_000).reduce((m, s) => Math.max(m, Math.abs(s)), 0);
if (!(shutPeak < 0.01)) {
  throw new Error(`a silent key should have closed the gate; peak ${shutPeak}`);
}
// The control is the same gate with no key *written* — that one listens to itself, and on this
// material stays wide open. (Keeping the `< side(0)` and simply not supplying the signal is a
// different thing: an unconnected side input is silence, so the gate shuts. That is deliberate —
// a gate whose key went missing should close, not fall back to opening itself.)
const openPeak = Chain.fromText("gate(-40, 60, 0.001, 0, 0.005)")
  .process(tone, 48_000)
  .slice(24_000)
  .reduce((m, s) => Math.max(m, Math.abs(s)), 0);
if (!(openPeak > 0.9)) {
  throw new Error(`an unkeyed gate on loud material closed; peak ${openPeak}`);
}

// Observer taps (roadmap A1-A3): the analyser path a page actually wants — render and measure in
// one pass, with the audio unchanged.
const analysed = Chain.fromText("meter | gain(0.5) | spectrum(2048, 0.5)").processTaps(tone, 48_000);
const plainGain = Chain.fromText("gain(0.5)").process(tone, 48_000);
if (analysed.audio.length !== plainGain.length) {
  throw new Error("processTaps changed the length");
}
for (let i = 0; i < plainGain.length; i++) {
  if (analysed.audio[i] !== plainGain[i]) {
    throw new Error(`a tap changed sample ${i}: ${analysed.audio[i]} vs ${plainGain[i]}`);
  }
}
if (analysed.taps.map((t) => t.kind).join(",") !== "meter,spectrum") {
  throw new Error(`taps reported ${analysed.taps.map((t) => t.kind)}`);
}
// The 1 kHz tone is full scale before the gain: 0 dBFS at the meter, and about 0.5 in its own bin
// after it. "About", because 1 kHz is bin 42.67 of a 2048-point FFT at 48 kHz — a tone between two
// bins loses up to 1.4 dB to the window, so the reading is 0.465 rather than 0.5. The exact-bin
// case is pinned against SciPy in crates/fluxion-ops/tests/spectrum_golden.rs.
const meterDb = analysed.taps[0].peakDb;
const spec = analysed.taps[1];
const peakBin = spec.magnitude[Math.round(1000 / spec.binHz)];
if (Math.abs(meterDb) > 0.1 || Math.abs(peakBin - 0.5) > 0.05) {
  throw new Error(`meter ${meterDb} dBFS, 1 kHz bin ${peakBin}`);
}

console.log(
  `fluxion wasm smoke OK (version ${v}) — ensureFs 48k->44.1k exact to ${at44k.length} frames, ` +
    `tone within ${worst.toExponential(1)}; keyed gate shut to ${shutPeak.toExponential(1)}, ` +
    `unkeyed ${openPeak.toFixed(2)}; taps read ${meterDb.toFixed(2)} dBFS and ` +
    `${peakBin.toFixed(3)} at 1 kHz, audio bit-identical`,
);

// Load the built module in Node and call into it — the check that the wasm build works at all
// (roadmap W1), run by the `wasm` CI job.
//
// `--target web` means `init()` wants the wasm bytes; in a browser it fetches them, here we read
// the file. One artifact, both environments — nothing browser-specific to maintain separately.
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

import init, { ensureFs, version } from "./pkg/fluxion_wasm.js";

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

console.log(
  `fluxion wasm smoke OK (version ${v}) — ensureFs 48k->44.1k exact to ${at44k.length} frames, ` +
    `tone within ${worst.toExponential(1)}`,
);

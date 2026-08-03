// Load the built module in Node and call into it — the check that the wasm build works at all
// (roadmap W1), run by the `wasm` CI job.
//
// `--target web` means `init()` wants the wasm bytes; in a browser it fetches them, here we read
// the file. One artifact, both environments — nothing browser-specific to maintain separately.
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

import init, { version } from "./pkg/fluxion_wasm.js";

const wasm = fileURLToPath(new URL("./pkg/fluxion_wasm_bg.wasm", import.meta.url));
await init({ module_or_path: await readFile(wasm) });

const v = version();
if (!/^\d+\.\d+\.\d+/.test(v)) {
  throw new Error(`version() returned ${JSON.stringify(v)}, which is not a version`);
}
console.log(`fluxion wasm smoke OK (version ${v})`);

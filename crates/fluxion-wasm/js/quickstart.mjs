// The JavaScript quickstart. CI runs it with Node; in a browser drop the two node: imports and
// call `await init()` with no argument (it fetches the .wasm itself).
//
// After `npm install fluxion` the import is `from "fluxion"`; in-repo it is the relative path.
// Ten code lines is the budget (docs/interfaces.md); comments and blanks do not count.
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

import init, { Chain } from "./index.mjs";

await init({ module_or_path: await readFile(fileURLToPath(new URL("./pkg/fluxion_wasm_bg.wasm", import.meta.url))) });

const chain = Chain.fromText("highpass(80, 4) | gain(-3dB)");
const samples = Float32Array.from({ length: 48000 }, (_, n) => Math.sin((n * 440 * 2 * Math.PI) / 48000));
const out = chain.process(samples, 48000);

console.log(`ok: ${chain.toText()} -> ${out.length} samples`);

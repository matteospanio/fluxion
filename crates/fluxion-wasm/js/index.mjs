// The fluxion browser package.
//
// Re-exports the wasm module's API plus the op catalog, so a consumer has one import:
//
//   import init, { Chain, OPS } from "fluxion";
//   await init();
//   const out = Chain.fromText("highpass(80, 4) | gain(-3dB)").process(samples, 48000);
//
// `init` loads the WebAssembly. In a browser it fetches the .wasm next to this file; in Node,
// hand it the bytes: `await init({ module_or_path: await readFile(url) })`.

export { default, default as init, Chain, ops, version } from "./pkg/fluxion_wasm.js";
export { OPS, OP_NAMES } from "./ops.js";

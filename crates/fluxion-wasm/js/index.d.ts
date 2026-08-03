// Types for the fluxion browser package. The per-op catalog types are generated (see ops.d.ts);
// these re-export the wasm module's own generated declarations alongside them.
export { default, default as init, Chain, ops, version } from "./pkg/fluxion_wasm.js";
export { OPS, OP_NAMES } from "./ops.js";
export type { OpName, OpGroup, OpSpec, OpParam, Unit } from "./ops.js";

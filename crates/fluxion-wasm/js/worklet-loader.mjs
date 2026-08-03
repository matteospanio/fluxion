// Attach fluxion to an AudioContext as an AudioWorklet.
//
// The awkward part of running wasm in an AudioWorklet is that the worklet scope has no module
// loader and no `fetch`: it cannot import the glue and it cannot go and get the `.wasm`. So the
// page fetches both, concatenates the classic-script glue with the processor, and registers the
// result — and passes the *compiled* module through `processorOptions`, which is structured-clone
// -able where a fetch is not available.
//
//   import { attachWorklet } from "fluxion/worklet";
//
//   const ctx = new AudioContext();
//   const node = await attachWorklet(ctx, "highpass(80, 4) | gain(0.8)");
//   node.connect(ctx.destination);
//   node.port.postMessage({ type: "audio", samples });      // feed it
//   node.port.postMessage({ type: "gain", value: 0.5 });    // ramped, not stepped

const HERE = import.meta.url;

/**
 * Build the worklet module source: the `no-modules` glue, then the processor that uses it.
 * @param {string} base where the package's files live
 */
async function moduleSource(base) {
  const [codec, glue, processor] = await Promise.all([
    // Must come first: the glue builds a TextDecoder at its top level, and the worklet scope has
    // none. See worklet-textcodec.js.
    fetch(new URL("./worklet-textcodec.js", base)).then((r) => r.text()),
    fetch(new URL("./pkg/no-modules/fluxion_wasm.js", base)).then((r) => r.text()),
    fetch(new URL("./fluxion-worklet.js", base)).then((r) => r.text()),
  ]);
  const source = [codec, glue, processor].join("\n;\n");
  return URL.createObjectURL(new Blob([source], { type: "application/javascript" }));
}

/**
 * Register the processor and return a node running `chain` at the context's sample rate.
 *
 * @param {BaseAudioContext} context
 * @param {string} chain in the shared chain syntax, e.g. `"highpass(80, 4) | gain(-3dB)"`
 * @param {{ ringFrames?: number, outputChannels?: number, base?: string,
 *           prime?: Float32Array }} [options]
 *   `prime` is audio placed in the ring before the first block — the only way to feed an
 *   OfflineAudioContext, which can finish rendering before a postMessage arrives.
 * @returns {Promise<AudioWorkletNode>}
 */
export async function attachWorklet(context, chain, options = {}) {
  const base = options.base ?? HERE;
  const url = await moduleSource(base);
  try {
    await context.audioWorklet.addModule(url);
  } finally {
    URL.revokeObjectURL(url);
  }

  // Compile once on the main thread; the worklet instantiates it synchronously, because
  // `process()` can be called before any promise it started would resolve.
  const module = await WebAssembly.compileStreaming(
    fetch(new URL("./pkg/fluxion_wasm_bg.wasm", base)),
  );

  return new AudioWorkletNode(context, "fluxion-processor", {
    numberOfInputs: 0,
    numberOfOutputs: 1,
    outputChannelCount: [options.outputChannels ?? 1],
    processorOptions: {
      module,
      chain,
      ringFrames: options.ringFrames ?? 128 * 64,
      prime: options.prime,
    },
  });
}

/**
 * Ask the worklet how it is doing: blocks rendered, blocks that ran short, frames buffered, and
 * the wasm-side allocation count.
 * @param {AudioWorkletNode} node
 * @returns {Promise<{rendered: number, dropped: number, buffered: number, allocations: number}>}
 */
export function stats(node) {
  return new Promise((resolve) => {
    const listener = (event) => {
      if (event.data?.type === "stats") {
        node.port.removeEventListener("message", listener);
        resolve(event.data);
      }
    };
    node.port.addEventListener("message", listener);
    node.port.start();
    node.port.postMessage({ type: "stats" });
  });
}

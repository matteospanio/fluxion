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
//
// Or as an insert effect on a track, with no added latency (roadmap L1–L4):
//
//   const fx = await attachWorklet(ctx, "lowpass(2000)", { input: true, channels: 2 });
//   source.connect(fx).connect(ctx.destination);
//   fx.port.postMessage({ type: "chain", chain: "lowpass(800)", crossfadeMs: 20 });

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

// One registration per context and one compile per wasm file, however many nodes are built.
// Registering the processor a second time on the same context throws — the name is taken — and a
// mixer builds a node per track, so the first `attachWorklet` does the work and the rest wait on
// it. Only the promise is kept; a failed attempt is forgotten, so a retry tries again.
const registrations = new WeakMap();
const modules = new Map();

function once(cache, key, make) {
  let pending = cache.get(key);
  if (!pending) {
    pending = make();
    cache.set(key, pending);
    pending.catch(() => cache.delete(key));
  }
  return pending;
}

async function register(context, base) {
  const url = await moduleSource(base);
  try {
    await context.audioWorklet.addModule(url);
  } finally {
    URL.revokeObjectURL(url);
  }
}

/**
 * Register the processor (once per context) and return a node running `chain` at the context's
 * sample rate. Types, and the messages both ways, are in worklet-loader.d.ts.
 *
 * @param {BaseAudioContext} context
 * @param {string} chain in the shared chain syntax, e.g. `"highpass(80, 4) | gain(-3dB)"`; empty
 *   means pass-through
 * @param {import("./worklet-loader").WorkletOptions} [options]
 *   `input: true` makes an insert node, which processes what the graph connects to it with no
 *   added latency; otherwise the node plays audio posted to it through a ring. `channels` is how
 *   many independent channels the chain runs on. `prime` is audio placed in the ring before the
 *   first block — the only way to feed an OfflineAudioContext in ring mode, which can finish
 *   rendering before a postMessage arrives.
 * @returns {Promise<AudioWorkletNode>}
 */
export async function attachWorklet(context, chain, options = {}) {
  const base = options.base ?? HERE;
  const wasm = new URL("./pkg/fluxion_wasm_bg.wasm", base).href;
  const [module] = await Promise.all([
    // Compiled on the main thread; the worklet instantiates it synchronously, because `process()`
    // can be called before any promise it started would resolve.
    once(modules, wasm, () => WebAssembly.compileStreaming(fetch(wasm))),
    once(registrations, context, () => register(context, base)),
  ]);

  const channels = options.channels ?? 1;
  const insert = options.input === true;
  return new AudioWorkletNode(context, "fluxion-processor", {
    numberOfInputs: insert ? 1 : 0,
    numberOfOutputs: 1,
    outputChannelCount: [insert ? channels : (options.outputChannels ?? channels)],
    // Exactly `channels` in, whatever is connected: a mono source is spread over a stereo node
    // (and stereo folded to mono) by the Web Audio rules, before the chain sees it.
    ...(insert && {
      channelCount: channels,
      channelCountMode: "explicit",
      channelInterpretation: "speakers",
    }),
    processorOptions: {
      module,
      chain,
      channels,
      input: insert,
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

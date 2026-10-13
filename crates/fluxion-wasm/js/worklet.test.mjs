// The AudioWorklet path, checked deterministically (roadmap W4, W5, L1–L4).
//
// W4's check is "5 s of playback, no dropped blocks, the wasm-side allocation counter stays at 0".
// Two of those three are not really about a browser: they are about what the code does per block,
// and driving `render()` directly gives a decision that is the same on every run. A browser adds
// scheduling noise and answers a different question — whether an AudioWorklet can host this — which
// `browser.test.mjs` covers separately.
//
//   ./scripts/build-wasm.sh && node crates/fluxion-wasm/js/worklet.test.mjs
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

import init, { Chain, Player, allocations, version } from "./index.mjs";

const url = (p) => fileURLToPath(new URL(p, import.meta.url));
const wasm = await init({ module_or_path: await readFile(url("./pkg/fluxion_wasm_bg.wasm")) });

const FS = 48_000;
const BLOCK = 128; // what an AudioWorklet hands you, every time
const CHAIN = "highpass(80, 4) | peaking(1000, 6, 1.5) | gain(0.8)";

const checks = [];
const check = (name, fn) => checks.push([name, fn]);

/** A Float32Array view straight onto the player's buffer — no copy in either direction. */
const view = (ptr, len) => new Float32Array(wasm.memory.buffer, ptr, len);

/** Deterministic input: an LCG, so a failure is reproducible. A different seed, a different signal. */
function source(frames, seed = 0x1234_5678) {
  let state = seed;
  return Float32Array.from({ length: frames }, () => {
    state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0;
    return ((state >>> 8) / 16_777_216) * 1.2 - 0.6;
  });
}

check("five seconds of playback allocates nothing and drops no block", () => {
  const player = new Player(CHAIN, FS, BLOCK, BLOCK * 64);
  const input = view(player.inputPtr(), BLOCK);
  const output = view(player.outputPtr(), BLOCK);

  const seconds = 5;
  const blocks = Math.floor((FS * seconds) / BLOCK);
  const audio = source(blocks * BLOCK);

  // Prime the ring the way a page would before starting the graph, then settle: wasm-bindgen's own
  // first calls allocate, and what is being measured is the steady state.
  for (let i = 0; i < 8; i++) {
    input.set(audio.subarray(i * BLOCK, (i + 1) * BLOCK));
    player.push(BLOCK);
  }
  player.render();
  player.reset();
  for (let i = 0; i < 8; i++) {
    input.set(audio.subarray(i * BLOCK, (i + 1) * BLOCK));
    player.push(BLOCK);
  }

  const before = allocations();
  let peak = 0;
  for (let b = 8; b < blocks; b++) {
    input.set(audio.subarray(b * BLOCK, (b + 1) * BLOCK));
    player.push(BLOCK);
    player.render();
    for (let i = 0; i < BLOCK; i++) peak = Math.max(peak, Math.abs(output[i]));
  }
  const allocated = allocations() - before;

  if (allocated !== 0) {
    throw new Error(`${allocated} allocations during ${seconds} s of playback; the budget is 0`);
  }
  if (player.blocksDropped() !== 0) {
    throw new Error(`${player.blocksDropped()} of ${player.blocksRendered()} blocks ran short`);
  }
  if (!(peak > 0.05)) {
    throw new Error(`the output is silent (peak ${peak}) — nothing was actually played`);
  }
  return `${player.blocksRendered()} blocks, 0 allocations, 0 dropped, peak ${peak.toFixed(3)}`;
});

check("block-by-block playback matches the offline render", () => {
  // The promise the browser story rests on: what you preview is what you export. The realtime path
  // and the batch path are different executors, so this is worth checking rather than assuming.
  const frames = BLOCK * 200;
  const audio = source(frames);

  const offline = Chain.fromText(CHAIN).process(audio, FS);

  const player = new Player(CHAIN, FS, BLOCK, BLOCK * 8);
  const input = view(player.inputPtr(), BLOCK);
  const output = view(player.outputPtr(), BLOCK);
  const streamed = new Float32Array(frames);
  for (let b = 0; b < frames / BLOCK; b++) {
    input.set(audio.subarray(b * BLOCK, (b + 1) * BLOCK));
    player.push(BLOCK);
    player.render();
    streamed.set(output, b * BLOCK);
  }

  let worst = 0;
  for (let i = 0; i < frames; i++) worst = Math.max(worst, Math.abs(streamed[i] - offline[i]));
  // Both run the same filters over the same samples; the executors differ, so this is a tolerance
  // rather than an equality, but it should be tiny.
  if (!(worst <= 1e-5)) {
    throw new Error(`streamed output differs from the offline render by ${worst.toExponential(2)}`);
  }
  return `worst difference ${worst.toExponential(2)}`;
});

check("every chunk-local op plays live, exactly as it renders, allocating nothing", () => {
  // ROADMAP L5. These six used to be refused by `new Player`. Each node runs the offline kernel's
  // own per-frame step, so inside one wasm build the stream and the render are the same numbers:
  // equality, not a tolerance.
  const chains = [
    "tremolo(5, 0.5)",
    "overdrive(20, 0.2)",
    "gate(-20, 24, 0.002, 0.02, 0.05)",
    "chorus(1.5, 0.002, 0.025, 0.5)",
    "flanger(0.5, 0.002, 0.001, 0.5, 0.5)",
    "phaser(0.5, 0.5, 0.5, 0.5)",
  ];
  const frames = BLOCK * 200;
  const audio = source(frames);
  for (const chain of chains) {
    const offline = Chain.fromText(chain).process(audio, FS);
    const player = new Player(chain, FS, BLOCK, BLOCK * 8);
    const input = view(player.inputPtr(), BLOCK);
    const output = view(player.outputPtr(), BLOCK);
    const streamed = new Float32Array(frames);
    const play = (b) => {
      input.set(audio.subarray(b * BLOCK, (b + 1) * BLOCK));
      player.push(BLOCK);
      player.render();
      streamed.set(output, b * BLOCK);
    };
    play(0); // wasm-bindgen's first calls allocate; what is measured is the steady state
    const before = allocations();
    for (let b = 1; b < frames / BLOCK; b++) play(b);
    const allocated = allocations() - before;
    if (allocated !== 0) throw new Error(`'${chain}' allocated ${allocated} times while playing`);
    const i = streamed.findIndex((v, k) => !Object.is(v, offline[k]));
    if (i !== -1) {
      throw new Error(`'${chain}' plays ${streamed[i]} at frame ${i} but renders ${offline[i]}`);
    }
  }
  return `${chains.length} ops, bit-identical to the offline render, 0 allocations`;
});

check("a starved ring is reported, not hidden", () => {
  // Underrun has to be visible. A player that silently emitted silence would look identical to one
  // that was working, which is the failure mode this counter exists to prevent.
  const player = new Player("gain(1)", FS, BLOCK, BLOCK * 4);
  player.render(); // nothing pushed at all
  if (player.blocksDropped() !== 1) {
    throw new Error(`an empty ring should count as one dropped block, got ${player.blocksDropped()}`);
  }
  return "underrun counted";
});

check("a chain that cannot run in a callback is refused up front", () => {
  for (const chain of ["reverse", "loudnorm(-14, -1)", "normalize(0.5)"]) {
    let threw = false;
    try {
      new Player(chain, FS, BLOCK, BLOCK * 4);
    } catch (e) {
      threw = /audio callback/.test(String(e.message ?? e));
    }
    if (!threw) throw new Error(`'${chain}' should have been refused`);
  }
  return "whole-signal ops refused with a reason";
});

// --- W5: live parameter changes ---------------------------------------------------------------

check("a 40 dB gain jump renders as a smooth ramp", () => {
  const player = new Player("gain(1)", FS, BLOCK, BLOCK * 64);
  const input = view(player.inputPtr(), BLOCK);
  const output = view(player.outputPtr(), BLOCK);

  // DC in, so the output *is* the gain curve and nothing has to be inferred from a waveform.
  const dc = new Float32Array(BLOCK).fill(1.0);

  const rampMs = 20;
  const blocks = 40;
  const curve = [];
  for (let b = 0; b < blocks; b++) {
    input.set(dc);
    player.push(BLOCK);
    // 40 dB down, between one block and the next: the jump W5 names.
    if (b === 10) player.setGain(0.01, rampMs);
    player.render();
    for (let i = 0; i < BLOCK; i++) curve.push(output[i]);
  }

  // No step: the largest change between adjacent samples must be far below the 0.99 a jump would
  // give. A 20 ms ramp over 40 dB moves about 1e-3 per sample.
  let biggestStep = 0;
  for (let i = 1; i < curve.length; i++) {
    biggestStep = Math.max(biggestStep, Math.abs(curve[i] - curve[i - 1]));
  }
  if (!(biggestStep < 0.01)) {
    throw new Error(`the gain stepped by ${biggestStep.toFixed(4)} — that is a click, not a ramp`);
  }

  // And it has to actually arrive: monotone down, landing on the target.
  const settled = curve[curve.length - 1];
  if (Math.abs(settled - 0.01) > 1e-4) {
    throw new Error(`the ramp settled at ${settled}, not at the 0.01 target`);
  }
  const rampSamples = Math.round((rampMs / 1000) * FS);
  for (let i = 10 * BLOCK + 1; i < 10 * BLOCK + rampSamples; i++) {
    if (curve[i] > curve[i - 1] + 1e-6) {
      throw new Error(`the ramp went back up at sample ${i}`);
    }
  }
  return `largest per-sample step ${biggestStep.toExponential(2)}, settled at ${settled}`;
});

check("setting a gain allocates nothing", () => {
  const player = new Player("gain(1)", FS, BLOCK, BLOCK * 8);
  player.setGain(0.5, 10);
  const before = allocations();
  for (let i = 0; i < 1000; i++) player.setGain(i % 2 ? 0.25 : 0.75, 5);
  const allocated = allocations() - before;
  if (allocated !== 0) throw new Error(`${allocated} allocations from 1000 parameter changes`);
  return "1000 parameter changes, 0 allocations";
});

// --- L1–L4: the processor a page actually runs ------------------------------------------------
//
// Everything above drives `Player` directly. A page runs fluxion-worklet.js on top of the
// no-modules glue, and insert mode, channels, the views onto wasm memory and chain swaps all live in
// that file — so it is loaded here the way `attachWorklet` loads it, into a sandbox shaped like
// AudioWorkletGlobalScope (no TextDecoder, no fetch, one wasm instance shared by every processor),
// and `process()` is called the way the audio thread calls it.

const registered = {};
const scope = vm.createContext({
  sampleRate: FS,
  console,
  registerProcessor: (name, processor) => {
    registered[name] = processor;
  },
  AudioWorkletProcessor: class {
    constructor() {
      // The page's end of the port: whatever the processor posts lands in `sent`.
      this.port = {
        sent: [],
        postMessage(message) {
          this.sent.push(message);
        },
      };
    }
  },
});
const shipped = ["./worklet-textcodec.js", "./pkg/no-modules/fluxion_wasm.js", "./fluxion-worklet.js"];
vm.runInContext(
  (await Promise.all(shipped.map((p) => readFile(url(p), "utf8")))).join("\n;\n"),
  scope,
);
// Compiled inside the sandbox: `initSync` checks `instanceof WebAssembly.Module`, which a module
// from another realm fails. In a browser it arrives by structured clone, already in the right one.
scope.bytes = await readFile(url("./pkg/fluxion_wasm_bg.wasm"));
const module = vm.runInContext("new WebAssembly.Module(bytes)", scope);
scope.module = module;
const Processor = registered["fluxion-processor"];
const worklet = vm.runInContext("wasm_bindgen.initSync({ module })", scope);
const workletAllocations = () => vm.runInContext("wasm_bindgen.allocations()", scope);

/** A processor built the way `attachWorklet` builds one. */
const node = (chain, options = {}) =>
  new Processor({ processorOptions: { module, chain, ...options } });
/** A message from the page. */
const tell = (p, message) => p.port.onmessage({ data: message });

/**
 * One render quantum. `input` is one Float32Array per channel, `[]` for an insert node with
 * nothing connected, and `undefined` for a ring-mode node, which has no input at all.
 */
function quantum(p, input, channels = 1) {
  const out = Array.from({ length: channels }, () => new Float32Array(BLOCK));
  p.process(input === undefined ? [] : [input], [out]);
  return out;
}

/** Stream one signal per channel through an insert-mode processor, quantum by quantum. */
function stream(p, signals, at = () => {}) {
  const out = signals.map((s) => new Float32Array(s.length));
  for (let b = 0; b * BLOCK < signals[0].length; b++) {
    at(b);
    const input = signals.map((s) => s.subarray(b * BLOCK, (b + 1) * BLOCK));
    quantum(p, input, signals.length).forEach((block, c) => out[c].set(block, b * BLOCK));
  }
  return out;
}

const offline = (chain, x) => Chain.fromText(chain).process(x, FS);
/** Largest |a − b|; 0 means bit-identical. */
const worst = (a, b) => a.reduce((w, x, i) => Math.max(w, Math.abs(x - b[i])), 0);
const same = (what, a, b) => {
  const d = worst(a, b);
  if (d !== 0) throw new Error(`${what} differs by ${d.toExponential(2)}; it should be bit-identical`);
};
const largestStep = (x) => x.reduce((w, v, i) => (i ? Math.max(w, Math.abs(v - x[i - 1])) : 0), 0);

check("insert mode: the output is Chain.process of the input, with no offset (L1)", () => {
  // Bit-identical at the same index is what "zero latency" means: one frame of offset would leave
  // a difference as large as the signal.
  const audio = source(BLOCK * 100);
  const p = node(CHAIN, { input: true });
  const [out] = stream(p, [audio]);
  same("the insert node's output", out, offline(CHAIN, audio));
  return "100 quanta bit-identical to the offline render, zero offset";
});

check("an unconnected input is silence, and the chain's tail still plays out", () => {
  // A disconnected source is `inputs[0] = []` in Web Audio, not an error.
  const chain = "echo(0.01, 0.5, 0.5)";
  const audio = source(BLOCK * 20);
  const p = node(chain, { input: true });
  const out = new Float32Array(BLOCK * 40);
  for (let b = 0; b < 40; b++) {
    const input = b < 20 ? [audio.subarray(b * BLOCK, (b + 1) * BLOCK)] : [];
    out.set(quantum(p, input)[0], b * BLOCK);
  }
  const padded = new Float32Array(out.length);
  padded.set(audio);
  same("the output", out, offline(chain, padded));
  const tail = out.subarray(20 * BLOCK).reduce((m, v) => Math.max(m, Math.abs(v)), 0);
  if (!(tail > 0.01)) throw new Error(`no echo tail after the input went away (peak ${tail})`);
  return `tail peak ${tail.toFixed(3)} after the input went away`;
});

check("stereo: each output channel is Chain.process of that channel alone (L2)", () => {
  const chain = "highpass(80, 4) | reverb(0.7, 0.4, 0.5)";
  const left = source(BLOCK * 100);
  const right = source(BLOCK * 100, 0x9e37_79b9);
  const p = node(chain, { input: true, channels: 2 });
  const [l, r] = stream(p, [left, right]);
  same("the left channel", l, offline(chain, left));
  same("the right channel", r, offline(chain, right));
  // A reverb shared between the channels would still pass if left and right were the same signal.
  if (worst(l, r) < 0.1) throw new Error("left and right came out alike — the check is blind");
  return "both channels bit-identical, each to its own offline render";
});

check("ring mode carries N channels too: one array per channel, or one for all", () => {
  const frames = BLOCK * 50;
  const left = source(frames);
  const right = source(frames, 0x9e37_79b9);
  const p = node(CHAIN, { channels: 2, ringFrames: frames + BLOCK * 4, prime: [left, right] });
  const out = [new Float32Array(frames), new Float32Array(frames)];
  for (let b = 0; b < 50; b++) quantum(p, undefined, 2).forEach((x, c) => out[c].set(x, b * BLOCK));
  same("the left channel", out[0], offline(CHAIN, left));
  same("the right channel", out[1], offline(CHAIN, right));

  const mono = node("gain(0.5)", { channels: 2, ringFrames: BLOCK * 8 });
  tell(mono, { type: "audio", samples: left.subarray(0, BLOCK) });
  const [a, b] = quantum(mono, undefined, 2);
  same("a mono buffer on the right channel", b, a);
  same("a mono buffer", a, offline("gain(0.5)", left.subarray(0, BLOCK)));
  return "stereo ring bit-identical per channel; a mono buffer feeds every channel";
});

check("nine nodes on one wasm instance: the ninth grows memory, no view goes stale (L3)", () => {
  const frames = BLOCK * 40;
  const chains = [
    CHAIN,
    "reverb(0.8, 0.3, 0.4)",
    "echo(0.05, 0.5, 0.5) | lowpass(3000, 2)",
    "gain(0.25)",
    "compand(0.01, 0.1, -30, 4, 6, 3) | highshelf(4000, -3, 0.7)",
    "delay(0.02, 0.5) + gain(0.5)",
    "bandpass(1000, 2) | gain(2)",
    "id",
  ];
  const inputs = chains.map((_, i) => source(frames, 0x1000 + i));
  // Half insert and half ring: both kinds hold views onto wasm memory, in different places.
  const insert = (i) => i % 2 === 1;
  const nodes = chains.map((chain, i) =>
    insert(i)
      ? node(chain, { input: true })
      : node(chain, { ringFrames: frames + BLOCK * 4, prime: inputs[i] }),
  );
  const outs = chains.map(() => new Float32Array(frames));
  const ninthInput = source(BLOCK * 20, 0x2000);
  const ninthOut = new Float32Array(ninthInput.length);
  let ninth = null;

  const play = (from, to) => {
    for (let b = from; b < to; b++) {
      nodes.forEach((p, i) => {
        const input = insert(i) ? [inputs[i].subarray(b * BLOCK, (b + 1) * BLOCK)] : undefined;
        outs[i].set(quantum(p, input)[0], b * BLOCK);
      });
      if (ninth) {
        const at = (b - 20) * BLOCK;
        ninthOut.set(quantum(ninth, [ninthInput.subarray(at, at + BLOCK)])[0], at);
      }
    }
  };

  play(0, 20);
  const before = worklet.memory.buffer;
  const size = before.byteLength; // read now: growth detaches `before`, and it reads 0 after
  // Thirty seconds of delay line is 5.8 MB, which the heap does not have lying around.
  ninth = node("delay(30, 0.5)", { input: true });
  if (worklet.memory.buffer === before) {
    throw new Error("wasm memory did not grow, so this check would prove nothing");
  }
  play(20, 40);

  chains.forEach((chain, i) => {
    same(`node ${i + 1} (${chain})`, outs[i], offline(chain, inputs[i]));
  });
  same("the ninth node", ninthOut, offline("delay(30, 0.5)", ninthInput));
  return `memory grew ${size} -> ${worklet.memory.buffer.byteLength} bytes mid-play; all nine bit-identical`;
});

check("swapping gain(1) for gain(0.5) crossfades: no step larger than the fade allows (L4)", () => {
  const p = node("gain(1)", { input: true });
  const dc = new Float32Array(BLOCK).fill(1); // DC in, so the output *is* the gain curve
  const curve = [];
  for (let b = 0; b < 30; b++) {
    if (b === 10) tell(p, { type: "chain", chain: "gain(0.5)", crossfadeMs: 20 });
    curve.push(...quantum(p, [dc])[0]);
  }
  const swap = 10 * BLOCK;
  const fade = Math.round(0.02 * FS);
  if (curve.slice(0, swap).some((v) => v !== 1)) throw new Error("the old chain changed before the swap");
  if (curve.slice(swap + fade).some((v) => v !== 0.5)) throw new Error("not the new chain after the fade");
  const step = largestStep(curve);
  // A linear fade of 0.5 over 960 samples moves 5.2e-4 per sample; a hard switch would move 0.5.
  if (!(step <= 0.5 / fade + 1e-6)) throw new Error(`a step of ${step.toExponential(2)} — that is a click`);
  const reply = p.port.sent.find((m) => m.type === "chain" && m.chain === "gain(0.5)");
  if (!reply?.ok) throw new Error(`the swap was not acknowledged: ${JSON.stringify(p.port.sent)}`);
  return `largest per-sample step ${step.toExponential(2)} (bound ${(0.5 / fade).toExponential(2)})`;
});

check("across a swap: the old chain before, the new chain after, both fed the same input", () => {
  const from = CHAIN;
  const to = "lowpass(2000, 2) | echo(0.01, 0.4, 0.5)";
  const audio = source(BLOCK * 60);
  const p = node(from, { input: true });
  const swapAt = 20;
  const [out] = stream(p, [audio], (b) => {
    if (b === swapAt) tell(p, { type: "chain", chain: to, crossfadeMs: 10 });
  });
  const swap = swapAt * BLOCK;
  const fade = Math.round(0.01 * FS);
  same("before the swap", out.subarray(0, swap), offline(from, audio).subarray(0, swap));
  // The new chain starts cold at the swap and hears everything from there on — during the fade too,
  // or its filters and echo would still be settling when it takes over.
  same(
    "after the fade",
    out.subarray(swap + fade),
    offline(to, audio.subarray(swap)).subarray(fade),
  );
  return "bit-identical to the old chain before and to the new one after";
});

check("a chain that cannot run is refused on the port, and the old one keeps playing", () => {
  const audio = source(BLOCK * 30);
  const p = node(CHAIN, { input: true });
  const [out] = stream(p, [audio], (b) => {
    if (b === 10) tell(p, { type: "chain", chain: "reverse" });
    if (b === 15) tell(p, { type: "chain", chain: "hipass(80)" });
  });
  same("the output", out, offline(CHAIN, audio));
  const errors = p.port.sent.filter((m) => m.type === "error");
  if (errors.length !== 2) throw new Error(`expected two refusals, got ${JSON.stringify(p.port.sent)}`);
  if (errors[0].chain !== "reverse" || !/audio callback/.test(errors[0].message)) {
    throw new Error(`the refusal of 'reverse' does not say why: ${errors[0].message}`);
  }
  if (errors[1].chain !== "hipass(80)" || !/highpass/.test(errors[1].message)) {
    throw new Error(`the refusal of a typo does not suggest the fix: ${errors[1].message}`);
  }
  return "both refused with a reason; output bit-identical to the old chain throughout";
});

check("an initial chain that cannot run passes audio through, and says why", () => {
  const audio = source(BLOCK * 10);
  const p = node("reverse", { input: true }); // must not throw: a thrown constructor is a dead node
  const [out] = stream(p, [audio]);
  same("the output", out, audio);
  const [error] = p.port.sent;
  if (error?.type !== "error" || error.chain !== "reverse" || !/audio callback/.test(error.message)) {
    throw new Error(`expected an error message first, got ${JSON.stringify(p.port.sent)}`);
  }
  return "pass-through, and an error on the port";
});

check("an empty chain is a pass-through, at construction and on a swap", () => {
  const audio = source(BLOCK * 30);
  const p = node("", { input: true });
  const [out] = stream(p, [audio], (b) => {
    if (b === 10) tell(p, { type: "chain", chain: "gain(0.5)", crossfadeMs: 0 });
    if (b === 20) tell(p, { type: "chain", chain: "  \n", crossfadeMs: 0 });
  });
  const cut = (x, b) => x.subarray(b * BLOCK, (b + 10) * BLOCK);
  same("the empty chain", cut(out, 0), cut(audio, 0));
  same("gain(0.5)", cut(out, 10), cut(offline("gain(0.5)", audio), 10));
  same("the whitespace chain", cut(out, 20), cut(audio, 20));
  const acks = p.port.sent.filter((m) => m.type === "chain" && m.ok).map((m) => m.chain);
  if (acks.join("|") !== "|gain(0.5)|  \n") throw new Error(`acknowledged ${JSON.stringify(acks)}`);
  return "identity, and a crossfadeMs of 0 switches at the quantum boundary";
});

check("the node's gain carries across a swap", () => {
  const p = node("gain(1)", { input: true });
  const dc = new Float32Array(BLOCK).fill(1);
  tell(p, { type: "gain", value: 0.5, rampMs: 0 });
  tell(p, { type: "chain", chain: "id", crossfadeMs: 0 });
  const [out] = quantum(p, [dc]);
  if (out.some((v) => v !== 0.5)) throw new Error(`gain reset by the swap: ${out[0]}`);
  return "a new chain starts at the gain the node already had";
});

check("a swap that arrives mid-fade waits for it, and the latest one wins", () => {
  const p = node("gain(1)", { input: true });
  const dc = new Float32Array(BLOCK).fill(1);
  const curve = [];
  for (let b = 0; b < 40; b++) {
    if (b === 10) tell(p, { type: "chain", chain: "gain(0.5)", crossfadeMs: 20 });
    if (b === 11) {
      tell(p, { type: "chain", chain: "gain(0.9)", crossfadeMs: 20 });
      tell(p, { type: "chain", chain: "gain(0.25)", crossfadeMs: 20 });
    }
    curve.push(...quantum(p, [dc])[0]);
  }
  const fade = Math.round(0.02 * FS);
  const step = largestStep(curve);
  if (!(step <= 0.5 / fade + 1e-6)) throw new Error(`a step of ${step.toExponential(2)}`);
  if (curve.at(-1) !== 0.25) throw new Error(`landed on ${curve.at(-1)}, not the latest chain`);
  if (curve[10 * BLOCK + fade - 1] !== 0.5) throw new Error("the first fade was cut short");
  if (curve.some((v) => v > 1 || v < 0.25)) throw new Error("the superseded chain was heard");
  return `two fades back to back, largest step ${step.toExponential(2)}`;
});

check("reset clears the tail, the ring and the counters", () => {
  const p = node("reverb(0.9, 0.2, 1)", { input: true });
  stream(p, [source(BLOCK * 20)]);
  tell(p, { type: "reset" });
  const [after] = quantum(p, [new Float32Array(BLOCK)]);
  if (after.some((v) => v !== 0)) throw new Error("the reverb tail survived the reset");

  const ring = node("gain(1)", { ringFrames: BLOCK * 16, prime: source(BLOCK * 8) });
  quantum(ring, undefined);
  tell(ring, { type: "reset" });
  tell(ring, { type: "stats" });
  const stats = ring.port.sent.at(-1);
  if (stats.type !== "stats" || stats.buffered !== 0 || stats.rendered !== 0) {
    throw new Error(`after a reset: ${JSON.stringify(stats)}`);
  }
  return "silence in, silence out; nothing buffered";
});

check("steady-state quanta allocate nothing, through a crossfade too", () => {
  const p = node(CHAIN, { input: true, channels: 2 });
  const input = [source(BLOCK), source(BLOCK, 0x9e37_79b9)];
  const out = [[new Float32Array(BLOCK), new Float32Array(BLOCK)]];
  const run = (n) => {
    for (let i = 0; i < n; i++) p.process([input], out);
  };
  run(4); // wasm-bindgen's own first calls allocate; what is measured is the steady state

  let before = workletAllocations();
  run(400);
  const steady = workletAllocations() - before;
  // The swap itself builds players, and may allocate. The quanta after it — the fade, and the
  // quantum that frees the old chain — may not.
  tell(p, { type: "chain", chain: "lowpass(5000, 2) | reverb(0.5, 0.5, 0.2)", crossfadeMs: 20 });
  before = workletAllocations();
  run(100);
  const fading = workletAllocations() - before;
  if (steady !== 0 || fading !== 0) {
    throw new Error(`${steady} allocations in 400 steady quanta, ${fading} across the fade`);
  }
  return "400 stereo quanta and a crossfade, 0 allocations";
});

// --- run --------------------------------------------------------------------------------------

let failed = 0;
console.log(`fluxion wasm worklet tests (v${version()})`);
for (const [name, fn] of checks) {
  try {
    console.log(`  ok   ${name}${(() => { const d = fn(); return d ? ` — ${d}` : ""; })()}`);
  } catch (e) {
    console.error(`  FAIL ${name}\n       ${e.message}`);
    failed++;
  }
}
if (failed > 0) {
  console.error(`\n${failed} of ${checks.length} failed`);
  process.exit(1);
}
console.log(`${checks.length} checks passed`);

// The AudioWorkletProcessor half of fluxion playback (roadmap W4, W5, L1–L3).
//
// This file is *concatenated* after the `no-modules` wasm-bindgen glue and registered with
// `audioWorklet.addModule()` — see `attachWorklet()` in worklet-loader.mjs. An AudioWorklet has no
// module loader and no `fetch`, so the glue cannot be imported from in here; it has to already be
// in scope, and the compiled wasm has to arrive as bytes through `processorOptions`.
//
// Two ways in. **Ring mode** (the default) plays audio the page posts, through a lock-free ring.
// **Insert mode** (`input: true`) processes whatever the Web Audio graph connects to the node, each
// 128-frame quantum as it arrives: no ring and no added latency, so tracks started at the same
// `when` stay sample-aligned through their chains. Either way the chain itself sees the same thing
// — a quantum in, a quantum out — so there is one path through `process()`.
//
// Everything below runs on the audio thread. Nothing here may allocate, block, or wait.

/* global wasm_bindgen, registerProcessor, AudioWorkletProcessor, sampleRate */

// 128 is what the Web Audio API hands every processor, every call — there is no other size.
const BLOCK = 128;

class FluxionProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const {
      module,
      chain,
      channels = 1,
      input = false,
      ringFrames = BLOCK * 64,
      prime,
    } = options.processorOptions;

    // Synchronous instantiation: `process()` may be called immediately after construction, and a
    // promise would not have resolved by then. Every node on a context shares this worklet scope,
    // and `initSync` hands the second node the instance the first one made — one linear memory for
    // all of them, which is why `views()` has to watch it grow.
    this.memory = wasm_bindgen.initSync({ module }).memory;
    this.channels = channels;
    this.rendered = 0;
    this.silence = new Float32Array(BLOCK);

    // Ring mode keeps its ring in a pass-through player per channel (`id` is the chain syntax's
    // pass-through), and the chain then runs exactly as it does in insert mode.
    this.feed = input ? null : this.voices("id", ringFrames);
    this.current = this.voices(chain);

    // Audio supplied up front. An OfflineAudioContext can finish rendering before a
    // `postMessage` is delivered, so anything that must be in the ring before the first block has
    // to arrive through `processorOptions`.
    if (prime) {
      this.push(prime);
    }

    this.running = true;
    this.port.onmessage = (event) => this.control(event.data);
  }

  /// One player per channel for `chain`: N independent states, the batch engine's per-channel
  /// semantics. Throws if the chain does not parse or cannot run in a callback.
  voices(chain, capacity = BLOCK * 4) {
    const voices = [];
    for (let c = 0; c < this.channels; c++) {
      const player = new wasm_bindgen.Player(chain, sampleRate, BLOCK, capacity);
      voices.push({ player, input: null, output: null });
    }
    // Building players can grow wasm memory. Forgetting the buffer the views were made on makes the
    // next `views()` remake all of them — these included, which have none yet.
    this.buffer = null;
    return voices;
  }

  /// Views onto wasm memory, remade only when the memory has grown.
  ///
  /// Growth replaces `memory.buffer` and detaches every view onto the old one, and any node on the
  /// context can cause it just by being built. Comparing identities costs nothing per block; making
  /// views allocates, so it happens only when the buffer actually changed.
  views() {
    const buffer = this.memory.buffer;
    if (buffer === this.buffer) {
      return;
    }
    this.buffer = buffer;
    for (const voices of [this.feed, this.current]) {
      for (const voice of voices ?? []) {
        voice.input = new Float32Array(buffer, voice.player.inputPtr(), BLOCK);
        voice.output = new Float32Array(buffer, voice.player.outputPtr(), BLOCK);
      }
    }
  }

  /// Messages from the page. Parameter changes are applied here, between blocks, and take effect
  /// as a ramp rather than a step — see `Player::set_gain`.
  control(message) {
    switch (message.type) {
      case "audio":
        // The page hands over decoded audio; it lands in the lock-free ring, which is what lets
        // the two sides run at their own rates.
        this.push(message.samples);
        break;
      case "gain":
        for (const voice of this.current) {
          voice.player.setGain(message.value, message.rampMs ?? 20);
        }
        break;
      case "reset":
        // Stop, seek: whatever the chain remembers — a reverb tail, a delay line, audio waiting in
        // the ring — belongs to the old position.
        for (const voice of [...(this.feed ?? []), ...this.current]) {
          voice.player.reset();
        }
        this.rendered = 0;
        break;
      case "stats":
        // ponytail: the ring figures are channel 0's. Every channel is fed by the same message, so
        // they move together unless a page sends channels of different lengths.
        this.port.postMessage({
          type: "stats",
          rendered: this.rendered,
          dropped: this.feed ? this.feed[0].player.blocksDropped() : 0,
          buffered: this.feed ? this.feed[0].player.buffered() : 0,
          allocations: wasm_bindgen.allocations(),
        });
        break;
      case "stop":
        this.running = false;
        break;
      default:
        break;
    }
  }

  /// Ring mode: `samples` is one Float32Array for every channel, or an array of one per channel.
  push(samples) {
    if (!this.feed) {
      return; // insert mode reads the graph, not the page
    }
    this.views();
    for (let c = 0; c < this.channels; c++) {
      const data = ArrayBuffer.isView(samples) ? samples : samples?.[c];
      const feed = this.feed[c];
      for (let offset = 0; data && offset < data.length; offset += BLOCK) {
        const chunk = data.subarray(offset, Math.min(offset + BLOCK, data.length));
        feed.input.set(chunk);
        if (feed.player.push(chunk.length) < chunk.length) {
          // The ring is full: the page is ahead of playback, which is the good direction.
          // Dropping the surplus beats growing a buffer on the audio thread.
          break;
        }
      }
    }
  }

  process(inputs, outputs) {
    this.views();
    // An insert node with nothing connected gets `inputs[0] = []`: silence, not an error.
    const input = inputs[0];
    for (let c = 0; c < this.channels; c++) {
      let source = this.silence;
      if (this.feed) {
        this.feed[c].player.render();
        source = this.feed[c].output;
      } else if (input && c < input.length) {
        source = input[c];
      }
      // Pushed and rendered in the same quantum: this is the zero added latency.
      play(this.current[c], source);
    }

    // Output channels past the node's own (ring mode's `outputChannels`) repeat its last one, so a
    // mono chain still fills a stereo output.
    const output = outputs[0];
    const last = this.channels - 1;
    for (let k = 0; k < output.length; k++) {
      output[k].set(this.current[k < last ? k : last].output);
    }

    this.rendered++;
    return this.running;
  }
}

/// Run one quantum of `source` through a voice.
function play(voice, source) {
  voice.input.set(source);
  voice.player.push(BLOCK);
  voice.player.render();
}

registerProcessor("fluxion-processor", FluxionProcessor);

// The AudioWorkletProcessor half of fluxion playback (roadmap W4, W5, L1–L4).
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
// — a quantum in, a quantum out — which is what lets one crossfade serve both.
//
// Everything below runs on the audio thread. `process()` must not allocate, block, or wait. The
// messages may allocate — a chain swap builds new players — because they arrive between quanta.

/* global wasm_bindgen, registerProcessor, AudioWorkletProcessor, sampleRate */

// 128 is what the Web Audio API hands every processor, every call — there is no other size.
const BLOCK = 128;

// What an empty chain means. `id` is the chain syntax's own pass-through and lowers to a gain of
// exactly 1: bit-identical to the input, through the same code path as every other chain.
const IDENTITY = "id";

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
    this.gain = 1;
    this.rendered = 0;
    this.silence = new Float32Array(BLOCK);

    // Ring mode keeps its ring in a pass-through player per channel, and the chain then runs exactly
    // as it does in insert mode. That is what makes a swap in ring mode the same operation as in
    // insert mode: the new chain has to hear the same input as the old one, and audio already
    // sitting in the old chain's own ring could not be handed over to it.
    this.feed = input ? null : this.voices(IDENTITY, ringFrames);

    // An initial chain that cannot run leaves the node passing audio through, and says why. A
    // constructor that threw would leave a node that outputs silence and a `processorerror` with
    // nothing in it a host could show to a user.
    try {
      this.current = this.voices(chain);
      this.port.postMessage({ type: "chain", chain, ok: true });
    } catch (e) {
      this.current = this.voices(IDENTITY);
      this.port.postMessage({ type: "error", chain, message: String(e?.message ?? e) });
    }

    // Swap state: `fading` is the chain on its way out, `pending` a swap that arrived during a fade
    // and waits for it to end.
    this.fading = null;
    this.pending = null;
    this.faded = 0;
    this.fadeLength = 0;

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
  /// semantics. Throws if the chain does not parse or cannot run in a callback — on the first
  /// channel, before anything is built, since every channel runs the same chain.
  voices(chain, capacity = BLOCK * 4) {
    const text = String(chain ?? "").trim() === "" ? IDENTITY : chain;
    const voices = [];
    for (let c = 0; c < this.channels; c++) {
      const player = new wasm_bindgen.Player(text, sampleRate, BLOCK, capacity);
      player.setGain(this.gain, 0);
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
    for (const voices of [this.feed, this.current, this.fading, this.pending?.voices]) {
      for (const voice of voices ?? []) {
        voice.input = new Float32Array(buffer, voice.player.inputPtr(), BLOCK);
        voice.output = new Float32Array(buffer, voice.player.outputPtr(), BLOCK);
      }
    }
  }

  /// Messages from the page, handled between quanta.
  control(message) {
    switch (message.type) {
      case "audio":
        // The page hands over decoded audio; it lands in the lock-free ring, which is what lets
        // the two sides run at their own rates.
        this.push(message.samples);
        break;
      case "gain":
        // A ramp rather than a step — see `Player::set_gain`. Remembered, so a chain swapped in
        // later starts at the node's level instead of jumping back to 1.
        this.gain = message.value;
        for (const voices of [this.current, this.fading, this.pending?.voices]) {
          for (const voice of voices ?? []) {
            voice.player.setGain(message.value, message.rampMs ?? 20);
          }
        }
        break;
      case "chain":
        this.swap(message.chain, message.crossfadeMs ?? 20);
        break;
      case "reset":
        // Stop, seek: whatever the chains remember — a reverb tail, a delay line, audio waiting in
        // the ring — belongs to the old position.
        for (const voices of [this.feed, this.current, this.fading, this.pending?.voices]) {
          for (const voice of voices ?? []) {
            voice.player.reset();
          }
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

  /// Replace the chain under a crossfade. Building the new players allocates, which is why this
  /// happens on a message and never inside `process()`.
  ///
  /// ponytail: the message is still handled on the audio thread, between two quanta. A stereo
  /// filter or reverb chain builds in about 20 µs, under 1% of a quantum at 48 kHz; a 30 s delay
  /// line takes about 0.7 ms, and more the first time, when memory has to grow. The upgrade is
  /// live parameters (L6), which change a chain without building one.
  swap(chain, crossfadeMs) {
    let voices;
    try {
      voices = this.voices(chain);
    } catch (e) {
      // Refused — and the old chain never stopped playing.
      this.port.postMessage({ type: "error", chain, message: String(e?.message ?? e) });
      return;
    }
    this.port.postMessage({ type: "chain", chain, ok: true });
    const length = Math.round((crossfadeMs / 1000) * sampleRate);
    if (this.fading) {
      // A fade is running. Cutting it short would leave a step, so this one waits for it, and only
      // the latest waiting chain is kept.
      // ponytail: a swap sent mid-fade starts up to one fade late. Fine for what a debounced knob
      // sends; a host that swaps faster than it fades wants a three-way mix instead.
      this.release(this.pending?.voices);
      this.pending = { voices, length };
      return;
    }
    this.begin(voices, length);
  }

  begin(voices, length) {
    if (!(length > 0)) {
      this.release(this.current);
      this.current = voices;
      return;
    }
    this.fading = this.current;
    this.current = voices;
    this.faded = 0;
    this.fadeLength = length;
  }

  release(voices) {
    for (const voice of voices ?? []) {
      voice.player.free();
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
      if (this.fading) {
        play(this.fading[c], source);
      }
    }

    // Output channels past the node's own (ring mode's `outputChannels`) repeat its last one, so a
    // mono chain still fills a stereo output.
    const output = outputs[0];
    const last = this.channels - 1;
    for (let k = 0; k < output.length; k++) {
      const c = k < last ? k : last;
      const now = this.current[c].output;
      if (!this.fading) {
        output[k].set(now);
        continue;
      }
      // Equal-gain (linear) crossfade. Both chains hear the same input, so their outputs are
      // correlated and add in amplitude: gains that sum to 1 hold the level, where equal-power
      // gains would bulge +3 dB in the middle of every swap.
      const old = this.fading[c].output;
      const out = output[k];
      for (let i = 0; i < BLOCK; i++) {
        const a = Math.min(1, (this.faded + i + 1) / this.fadeLength);
        out[i] = old[i] * (1 - a) + now[i] * a;
      }
    }

    if (this.fading) {
      this.faded += BLOCK;
      if (this.faded >= this.fadeLength) {
        // Freeing hands memory back and never allocates, so the quantum a fade ends in is as
        // allocation-free as any other.
        this.release(this.fading);
        this.fading = null;
        if (this.pending) {
          const { voices, length } = this.pending;
          this.pending = null;
          this.begin(voices, length);
        }
      }
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

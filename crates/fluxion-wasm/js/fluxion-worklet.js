// The AudioWorkletProcessor half of fluxion playback (roadmap W4, W5).
//
// This file is *concatenated* after the `no-modules` wasm-bindgen glue and registered with
// `audioWorklet.addModule()` — see `attachWorklet()` in worklet-loader.mjs. An AudioWorklet has no
// module loader and no `fetch`, so the glue cannot be imported from in here; it has to already be
// in scope, and the compiled wasm has to arrive as bytes through `processorOptions`.
//
// Everything below runs on the audio thread. Nothing here may allocate, block, or wait.

/* global wasm_bindgen, registerProcessor, AudioWorkletProcessor, sampleRate */

class FluxionProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const { module, chain, ringFrames = 128 * 64, prime } = options.processorOptions;

    // Synchronous instantiation: `process()` may be called immediately after construction, and a
    // promise would not have resolved by then. `initSync` hands back the instance exports, which
    // is where the linear memory lives.
    const wasm = wasm_bindgen.initSync({ module });

    // 128 is what the Web Audio API hands every processor, every call — there is no other size.
    this.block = 128;
    this.player = new wasm_bindgen.Player(chain, sampleRate, this.block, ringFrames);

    // Views onto wasm memory, made once and after the player exists. Rebuilding them per block
    // would allocate; taking them before the player was built would risk the buffer being detached
    // by memory growth during its construction.
    this.input = new Float32Array(wasm.memory.buffer, this.player.inputPtr(), this.block);
    this.output = new Float32Array(wasm.memory.buffer, this.player.outputPtr(), this.block);

    // Audio supplied up front. An OfflineAudioContext can finish rendering before a
    // `postMessage` is delivered, so anything that must be in the ring before the first block has
    // to arrive through `processorOptions`.
    if (prime) {
      this.push(prime);
    }

    this.running = true;
    this.port.onmessage = (event) => this.control(event.data);
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
        this.player.setGain(message.value, message.rampMs ?? 20);
        break;
      case "stats":
        this.port.postMessage({
          type: "stats",
          rendered: this.player.blocksRendered(),
          dropped: this.player.blocksDropped(),
          buffered: this.player.buffered(),
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

  push(samples) {
    for (let offset = 0; offset < samples.length; offset += this.block) {
      const chunk = samples.subarray(offset, Math.min(offset + this.block, samples.length));
      this.input.set(chunk);
      if (this.player.push(chunk.length) < chunk.length) {
        // The ring is full: the page is ahead of playback, which is the good direction. Dropping
        // the surplus beats growing a buffer on the audio thread.
        break;
      }
    }
  }

  process(_inputs, outputs) {
    this.player.render();
    const channels = outputs[0];
    for (const channel of channels) {
      channel.set(this.output);
    }
    return this.running;
  }
}

registerProcessor("fluxion-processor", FluxionProcessor);

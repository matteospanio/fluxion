//! `fluxion-wasm` — WebAssembly / browser bindings (wasm-bindgen).
//!
//! The same graph algebra in the browser: build a chain from the shared text syntax, render a
//! buffer, get the samples back. "Preview and export come from the same DSP" is the point — a web
//! host renders with the engine that will produce the final file, not an approximation of it.
//!
//! Build it with `scripts/build-wasm.sh`; the JavaScript package that wraps the output is in `js/`.
//!
//! Scope today is the offline path (roadmap W1–W2). AudioWorklet playback (W4), frozen `.fxg`
//! loading (W3) and the WebGPU lowering (W8) come later; the `Chain` class here is the one they
//! extend rather than something they replace.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use fluxion::{Graph, OpKind, Signal, SmoothedValue, process};
use fluxion_rt::ring::{Consumer, Producer, channel};
use wasm_bindgen::prelude::*;

// --- allocation counter ------------------------------------------------------------------------

/// Every allocation this module has made since it loaded.
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

/// A global allocator that does nothing but count.
///
/// "No memory allocation once started" is the one claim about a realtime path that cannot be
/// checked by looking at the code — a `Vec` that grows inside `process` looks like any other line.
/// Counting makes it a number a test can assert on, which is what roadmap W4 asks for.
struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// How many allocations this module has made since it loaded.
///
/// Snapshot it before playback and again after: the difference is what the audio path allocated,
/// and for a correct realtime path it is zero.
#[wasm_bindgen]
pub fn allocations() -> usize {
    ALLOCATIONS.load(Ordering::Relaxed)
}

/// The crate version, e.g. `"0.0.0"` — a cheap way for a page to confirm which build it loaded.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Every op name, for a page that wants to validate or autocomplete a chain string.
///
/// Names only, deliberately: parameters, units and ranges are in the generated TypeScript types
/// and in `docs/ops.md`, and shipping the whole catalog as data would grow the module for
/// something almost no page needs at run time.
#[wasm_bindgen]
pub fn ops() -> Vec<String> {
    OpKind::all().iter().map(|k| k.name().to_string()).collect()
}

/// An effect chain: build it from text, run it over a buffer.
///
/// The same graph the native library, the CLI, Python and C build — and the same text describes it
/// everywhere, so a chain previewed in a browser is the chain that renders the final file.
#[wasm_bindgen]
pub struct Chain {
    graph: Graph,
}

#[wasm_bindgen]
impl Chain {
    /// Parse a chain, e.g. `Chain.fromText("highpass(80, 4) | gain(-3dB)")`.
    ///
    /// Throws on a syntax or name error, with a message that carets the offending character and
    /// suggests a fix where the mistake looks like a typo. See `docs/chain-syntax.md`.
    #[wasm_bindgen(js_name = fromText)]
    pub fn from_text(text: &str) -> Result<Chain, JsError> {
        match fluxion::parse::chain(text) {
            Ok(graph) => Ok(Chain { graph }),
            Err(e) => Err(JsError::new(&e.render(text))),
        }
    }

    /// The canonical chain text. Round-trips: `Chain.fromText(c.toText())` is the same chain.
    #[wasm_bindgen(js_name = toText)]
    pub fn to_text(&self) -> String {
        self.graph.to_string()
    }

    /// Render one mono channel at `fs`, returning a new buffer of the same length.
    ///
    /// This is the offline path — it allocates, and it is what a page uses to draw a waveform or
    /// bounce a file. Block-by-block playback is the AudioWorklet (roadmap W4).
    pub fn process(&self, samples: &[f32], fs: u32) -> Vec<f32> {
        let out = process(&self.graph, &Signal::new(fs, vec![samples.to_vec()]));
        out.channels.into_iter().next().unwrap_or_default()
    }

    /// How many leaf ops the chain has — enough for a page to show what it built.
    #[wasm_bindgen(js_name = opCount)]
    pub fn op_count(&self) -> usize {
        self.graph.leaf_count()
    }
}

/// Route Rust panics to `console.error` with a readable message and a stack.
///
/// Runs automatically when the module is instantiated. Without it a panic surfaces in the browser
/// as `RuntimeError: unreachable executed`, which says nothing about what went wrong.
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

// --- AudioWorklet playback (roadmap W4, W5) ---------------------------------------------------

/// A chain running in an audio callback: audio in through a lock-free ring, 128-frame blocks out,
/// nothing allocated once it has started.
///
/// The page pushes decoded audio whenever it has some; the worklet pulls a block at a time on the
/// audio thread. The ring is what lets those two run at their own rates without a lock, which is
/// the only arrangement that works when one of them must never block.
///
/// Buffers live inside the wasm memory and are reached by pointer, so neither direction copies
/// through a JavaScript array — `&[f32]` across the wasm-bindgen boundary would allocate on every
/// call, which is exactly what this must not do.
#[wasm_bindgen]
pub struct Player {
    graph: fluxion::RtGraph,
    /// Post-chain gain, ramped rather than stepped. A parameter that jumps between blocks puts a
    /// step in the signal, and a step is a click.
    gain: SmoothedValue,
    /// Audio waiting to be played.
    ring_tx: Producer<f32>,
    ring_rx: Consumer<f32>,
    /// Where JavaScript writes audio to be pushed, and reads rendered audio from.
    input: Vec<f32>,
    output: Vec<f32>,
    /// Scratch for one block, so `process` needs no temporary.
    scratch: Vec<f32>,
    block: usize,
    fs: u32,
    blocks_rendered: usize,
    blocks_dropped: usize,
}

#[wasm_bindgen]
impl Player {
    /// Build a player for `chain` at `fs`, rendering `block` frames at a time and buffering up to
    /// `capacity` frames of input.
    ///
    /// Fails if the chain cannot run in an audio callback — `reverse` needs the whole signal,
    /// `loudnorm` has to measure it first. Saying so here beats discovering it mid-playback.
    #[wasm_bindgen(constructor)]
    pub fn new(chain: &str, fs: u32, block: usize, capacity: usize) -> Result<Player, JsError> {
        let graph: Graph =
            fluxion::parse::chain(chain).map_err(|e| JsError::new(&e.render(chain)))?;
        let mut rt = fluxion::to_rt_graph(&graph, fs).ok_or_else(|| {
            JsError::new(&format!(
                "`{chain}` cannot run in an audio callback: it contains an op that needs the whole \
                 signal (reverse, normalize, limiter, loudnorm) — render it offline with Chain instead"
            ))
        })?;
        // Every buffer the audio path will ever touch is sized now, while allocating is still
        // allowed.
        rt.prepare(block);
        let (ring_tx, ring_rx) = channel::<f32>(capacity.max(block * 4));

        Ok(Player {
            graph: rt,
            gain: SmoothedValue::new(1.0),
            ring_tx,
            ring_rx,
            input: vec![0.0; block],
            output: vec![0.0; block],
            scratch: vec![0.0; block],
            block,
            fs,
            blocks_rendered: 0,
            blocks_dropped: 0,
        })
    }

    /// Pointer to the input buffer: write `block` frames here, then call `push`.
    #[wasm_bindgen(js_name = inputPtr)]
    pub fn input_ptr(&self) -> *const f32 {
        self.input.as_ptr()
    }

    /// Pointer to the output buffer: after `render`, read `block` frames from here.
    #[wasm_bindgen(js_name = outputPtr)]
    pub fn output_ptr(&self) -> *const f32 {
        self.output.as_ptr()
    }

    /// Frames per block.
    #[wasm_bindgen(js_name = blockSize)]
    pub fn block_size(&self) -> usize {
        self.block
    }

    /// Move `frames` from the input buffer into the ring. Returns how many were accepted — fewer
    /// than asked means the ring is full and the producer should wait.
    pub fn push(&mut self, frames: usize) -> usize {
        let mut pushed = 0;
        for &sample in self.input.iter().take(frames.min(self.block)) {
            if self.ring_tx.push(sample).is_err() {
                break;
            }
            pushed += 1;
        }
        pushed
    }

    /// Frames currently buffered.
    pub fn buffered(&self) -> usize {
        self.ring_rx.len()
    }

    /// Render one block into the output buffer. Returns `false` if the ring did not have a full
    /// block — the block still renders, from silence, so playback continues rather than stopping,
    /// and the miss is counted.
    ///
    /// Allocation-free: everything it touches was sized in the constructor.
    pub fn render(&mut self) -> bool {
        let mut complete = true;
        for slot in self.scratch.iter_mut() {
            match self.ring_rx.pop() {
                Some(sample) => *slot = sample,
                None => {
                    *slot = 0.0;
                    complete = false;
                }
            }
        }

        self.graph.process(&self.scratch, &mut self.output);
        for sample in self.output.iter_mut() {
            *sample *= self.gain.tick();
        }

        self.blocks_rendered += 1;
        if !complete {
            self.blocks_dropped += 1;
        }
        complete
    }

    /// Ramp the output gain to `target` (linear) over `ramp_ms` milliseconds.
    ///
    /// The ramp is why this exists rather than a plain multiply: a 40 dB jump applied between one
    /// block and the next is a step, and a step is audible. Roadmap W5.
    #[wasm_bindgen(js_name = setGain)]
    pub fn set_gain(&mut self, target: f32, ramp_ms: f32) {
        let samples = (ramp_ms / 1000.0 * self.fs as f32).round().max(0.0) as u32;
        self.gain.set_target(target, samples);
    }

    /// The gain right now, part-way through any ramp.
    pub fn gain(&self) -> f32 {
        self.gain.value()
    }

    /// Blocks rendered since the player was built.
    #[wasm_bindgen(js_name = blocksRendered)]
    pub fn blocks_rendered(&self) -> usize {
        self.blocks_rendered
    }

    /// Blocks that ran short of input — the number that has to stay at zero for playback to be
    /// glitch-free.
    #[wasm_bindgen(js_name = blocksDropped)]
    pub fn blocks_dropped(&self) -> usize {
        self.blocks_dropped
    }

    /// Clear the filter state and the buffered audio, keeping the chain.
    pub fn reset(&mut self) {
        self.graph.reset();
        while self.ring_rx.pop().is_some() {}
        self.blocks_rendered = 0;
        self.blocks_dropped = 0;
    }
}

#[cfg(test)]
mod tests {
    //! These run natively — the crate is also an `rlib`, so whatever is not wasm-specific is
    //! covered by `cargo test --workspace` like everything else.
    use super::version;

    #[test]
    fn version_is_the_crate_version() {
        assert_eq!(version(), env!("CARGO_PKG_VERSION"));
    }
}

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

/// Resample one channel from `fromFs` to `toFs`, returning exactly
/// `round(input.length · toFs/fromFs)` samples (ROADMAP R2).
///
/// A page decodes at whatever rate the file happens to be and plays at whatever rate the
/// `AudioContext` picked; this is the step between. Input already at `toFs` is returned as it is.
/// The converter is the one the native library and the CLI use, so a preview in the browser and
/// the final render agree.
#[wasm_bindgen(js_name = ensureFs)]
pub fn ensure_fs(input: &[f32], from_fs: u32, to_fs: u32) -> Result<Vec<f32>, JsError> {
    if from_fs == 0 || to_fs == 0 {
        return Err(JsError::new(&format!(
            "sample rates must be positive (got {from_fs} -> {to_fs})"
        )));
    }
    Ok(fluxion::resample::convert(
        input,
        from_fs,
        to_fs,
        fluxion::resample::Quality::Hq,
    ))
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

    /// Render with **side inputs**: `sides[n]` is what `side(n)` in the chain reads (ROADMAP S1).
    ///
    /// A gate on one track opened by another, a bass ducked by a kick. Side signals are taken to be
    /// at `fs` like the input and read as silence past their end; a `side(n)` with nothing supplied
    /// is silence throughout, so the same chain text still runs through `process`.
    #[wasm_bindgen(js_name = processWith)]
    pub fn process_with(
        &self,
        samples: &[f32],
        fs: u32,
        sides: Vec<js_sys::Float32Array>,
    ) -> Vec<f32> {
        let sides: Vec<Signal> = sides
            .iter()
            .map(|s| Signal::new(fs, vec![s.to_vec()]))
            .collect();
        let refs: Vec<&Signal> = sides.iter().collect();
        let out =
            fluxion::process_with(&self.graph, &Signal::new(fs, vec![samples.to_vec()]), &refs);
        out.channels.into_iter().next().unwrap_or_default()
    }

    /// Render, and also return what the chain's observer taps saw (ROADMAP A1).
    ///
    /// This is the analyser path: `meter | gain(0.5) | spectrum(2048)` renders *and* measures in
    /// one pass, and the audio is bit-identical to [`Chain::process`]'s, because a tap reads the
    /// buffer and never writes to it. Returns `{ audio, taps }`, where each tap is
    /// `{ label, kind, ... }` — `binHz` and `magnitude` for a spectrum, `peakDb`, `rmsDb` and
    /// `shortTermLufs` for a meter — in the order the chain reaches them.
    #[wasm_bindgen(js_name = processTaps)]
    pub fn process_taps(&self, samples: &[f32], fs: u32) -> js_sys::Object {
        let (out, readings) =
            fluxion::process_taps(&self.graph, &Signal::new(fs, vec![samples.to_vec()]));

        let taps = js_sys::Array::new();
        for reading in readings {
            let entry = js_sys::Object::new();
            let set = |key: &str, value: JsValue| {
                // Both arguments are ours and the target is a fresh object, so this cannot fail.
                let _ = js_sys::Reflect::set(&entry, &JsValue::from_str(key), &value);
            };
            set(
                "label",
                match reading.label {
                    Some(l) => JsValue::from_str(&l),
                    None => JsValue::NULL,
                },
            );
            match reading.data {
                fluxion::TapData::Spectrum { bin_hz, magnitude } => {
                    set("kind", JsValue::from_str("spectrum"));
                    set("binHz", JsValue::from_f64(f64::from(bin_hz)));
                    set(
                        "magnitude",
                        js_sys::Float32Array::from(&magnitude[..]).into(),
                    );
                }
                fluxion::TapData::Meter {
                    peak_db,
                    rms_db,
                    short_term_lufs,
                } => {
                    set("kind", JsValue::from_str("meter"));
                    set("peakDb", JsValue::from_f64(f64::from(peak_db)));
                    set("rmsDb", JsValue::from_f64(f64::from(rms_db)));
                    set(
                        "shortTermLufs",
                        JsValue::from_f64(f64::from(short_term_lufs)),
                    );
                }
            }
            taps.push(&entry);
        }

        let audio = out.channels.into_iter().next().unwrap_or_default();
        let result = js_sys::Object::new();
        let _ = js_sys::Reflect::set(
            &result,
            &JsValue::from_str("audio"),
            &js_sys::Float32Array::from(&audio[..]).into(),
        );
        let _ = js_sys::Reflect::set(&result, &JsValue::from_str("taps"), &taps);
        result
    }

    /// Render frames `[from, to)` of the chain — a waveform tile, a loop preview, the bar someone
    /// just edited (ROADMAP D4).
    ///
    /// Bit-identical to the same window of a whole render, because it *is* that window: the chain
    /// runs from frame 0 and the rest is discarded. That makes it cost `to` frames of work, not
    /// `to - from` — see `framesToCompute`. Throws for a chain containing an op that needs the
    /// whole signal (`normalize`, `loudnorm`, `reverse`, `limiter`).
    #[wasm_bindgen(js_name = renderRegion)]
    pub fn render_region(
        &self,
        samples: &[f32],
        fs: u32,
        from: usize,
        to: usize,
    ) -> Result<Vec<f32>, JsError> {
        let input = Signal::new(fs, vec![samples.to_vec()]);
        match fluxion::render_region(&self.graph, &input, from, to) {
            Ok(out) => Ok(out.channels.into_iter().next().unwrap_or_default()),
            Err(e) => Err(JsError::new(&e.to_string())),
        }
    }

    /// Render with automation driving the chain's parameters (ROADMAP D2).
    ///
    /// Curves are read at absolute frames, so this and [`Chain::render_region_automated`] agree
    /// about what a parameter was doing at any point.
    #[wasm_bindgen(js_name = processAutomated)]
    pub fn process_automated(
        &self,
        samples: &[f32],
        fs: u32,
        automation: &Automation,
    ) -> Result<Vec<f32>, JsError> {
        let input = Signal::new(fs, vec![samples.to_vec()]);
        match fluxion::process_automated(&self.graph, &input, &automation.inner) {
            Ok(out) => Ok(out.channels.into_iter().next().unwrap_or_default()),
            Err(e) => Err(JsError::new(&e.to_string())),
        }
    }

    /// Both together: a window of an automated render.
    #[wasm_bindgen(js_name = renderRegionAutomated)]
    pub fn render_region_automated(
        &self,
        samples: &[f32],
        fs: u32,
        automation: &Automation,
        from: usize,
        to: usize,
    ) -> Result<Vec<f32>, JsError> {
        let input = Signal::new(fs, vec![samples.to_vec()]);
        match fluxion::render_region_automated(&self.graph, &input, &automation.inner, from, to) {
            Ok(out) => Ok(out.channels.into_iter().next().unwrap_or_default()),
            Err(e) => Err(JsError::new(&e.to_string())),
        }
    }

    /// How many side inputs this chain reads — 0 for an ordinary one-input chain.
    #[wasm_bindgen(js_name = sideInputs)]
    pub fn side_inputs(&self) -> usize {
        self.graph.side_inputs()
    }

    /// How many leaf ops the chain has — enough for a page to show what it built.
    #[wasm_bindgen(js_name = opCount)]
    pub fn op_count(&self) -> usize {
        self.graph.leaf_count()
    }
}

/// How many frames a `renderRegion(from, to)` has to compute — `to`, not `to - from`.
///
/// A window late in a timeline still costs the whole timeline, because every op before it carries
/// state. Exposed so a page can decide whether to ask rather than discovering it by timing.
#[wasm_bindgen(js_name = framesToCompute)]
pub fn frames_to_compute(from: usize, to: usize) -> usize {
    fluxion::region::frames_to_compute(from, to)
}

/// Parameter automation: curves driving named nodes of a chain (ROADMAP D2, S4).
///
/// Build it up with the lane methods, then hand it to `Chain.processAutomated`. A lane names a
/// node by the `name:` label it has in the chain text and a parameter by its registry name:
///
/// ```js
/// const chain = Chain.fromText("fade: gain(1) | lp: lowpass(8000, 4)");
/// const a = new Automation()
///   .dbRamp("fade", "gain", 1.0, 0.001, 1.0)   // a 60 dB fade over a second
///   .lfo("lp", "cutoff", 0.5, 400, 4000, 0);   // and a slow filter sweep under it
/// const out = chain.processAutomated(samples, 48000, a);
/// ```
#[wasm_bindgen]
#[derive(Default)]
pub struct Automation {
    inner: fluxion::automation::Automation,
}

#[wasm_bindgen]
impl Automation {
    /// An empty lane set.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Automation {
        Automation::default()
    }

    /// A straight line in the parameter's own units, from `from` to `to` over `seconds`.
    pub fn ramp(mut self, node: &str, param: &str, from: f32, to: f32, seconds: f64) -> Automation {
        self.inner = std::mem::take(&mut self.inner).with(fluxion::automation::Lane::new(
            node,
            param,
            fluxion::automation::Curve::ramp(from, to, seconds),
        ));
        self
    }

    /// A fade at a constant rate **in decibels** — what a gain lane dragged from 0 to -60 dB means.
    ///
    /// Half way through it is at -30 dB, where `ramp` between the same endpoints would be at -6.
    #[wasm_bindgen(js_name = dbRamp)]
    pub fn db_ramp(
        mut self,
        node: &str,
        param: &str,
        from: f32,
        to: f32,
        seconds: f64,
    ) -> Automation {
        self.inner = std::mem::take(&mut self.inner).with(fluxion::automation::Lane::new(
            node,
            param,
            fluxion::automation::Curve::db_ramp(from, to, seconds),
        ));
        self
    }

    /// An LFO sweeping between `low` and `high` at `rateHz`, starting `phase` (0..1) into its cycle.
    pub fn lfo(
        mut self,
        node: &str,
        param: &str,
        rate_hz: f32,
        low: f32,
        high: f32,
        phase: f32,
    ) -> Automation {
        self.inner = std::mem::take(&mut self.inner).with(fluxion::automation::Lane::new(
            node,
            param,
            fluxion::automation::Curve::lfo(rate_hz, low, high, phase),
        ));
        self
    }

    /// How many lanes this holds.
    #[wasm_bindgen(js_name = laneCount)]
    pub fn lane_count(&self) -> usize {
        self.inner.lanes().len()
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

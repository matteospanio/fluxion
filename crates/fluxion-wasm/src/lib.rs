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

use fluxion::{Graph, OpKind, Signal, process};
use wasm_bindgen::prelude::*;

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

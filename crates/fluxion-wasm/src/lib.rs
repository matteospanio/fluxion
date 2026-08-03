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

use wasm_bindgen::prelude::*;

/// The crate version, e.g. `"0.0.0"` — a cheap way for a page to confirm which build it loaded.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
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

//! `fluxion` — a modern, SoX-style audio DSP command-line interface.
//!
//! Backbone only. Argument parsing (clap), file IO (Symphonia via `fluxion-io`), and the
//! effect-chain parser are wired in as the library lands. For now this prints the planned
//! usage and a demo graph so the binary builds and runs.

use fluxion::prelude::*;

fn main() {
    // ponytail: placeholder entrypoint — real parse/dispatch arrives with fluxion-io + clap.
    let demo = lowpass(800.0) | gain(-3.0);

    eprintln!("fluxion {} — audio DSP CLI (scaffold)", env!("CARGO_PKG_VERSION"));
    eprintln!();
    eprintln!("planned commands:");
    eprintln!("  fluxion <in> <effects...> <out>   process a file (SoX-style pipeline)");
    eprintln!("  fluxion info <file>               print metadata (soxi-style)");
    eprintln!("  fluxion play <in> <effects...>    realtime monitoring (CPU engine)");
    eprintln!("  fluxion compile <chain> --fs ...  export a frozen-coefficient graph (.fxg)");
    eprintln!();
    eprintln!("demo graph: {demo:?}  ({} ops)", demo.leaf_count());
}

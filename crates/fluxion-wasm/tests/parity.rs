//! wasm renders what native renders (roadmap W2).
//!
//! The claim the whole browser story rests on is "preview and export come from the same DSP". This
//! is where that stops being a slogan: a fixture holds an input signal and the output native
//! produces for it, and `js/parity.mjs` feeds the same input through the built wasm module and
//! compares. Nothing here is wasm-specific, so it runs in `cargo test --workspace`; the Node side
//! then needs no Rust toolchain, which is what lets the `wasm` CI job be a plain `node` invocation.
//!
//! Regenerate the fixture only when the reference output legitimately changes:
//!
//! ```text
//! cargo test -p fluxion-wasm --test parity -- --ignored write_fixture
//! ```

use fluxion::{Graph, Signal, process};

/// The chain from the roadmap's own W2 check.
const CHAIN: &str = "highpass(80, 4) | gain(-3dB)";
const FS: u32 = 48_000;
const FRAMES: usize = 1024;

/// The tolerance W2 asks for, used by `js/parity.mjs` to compare wasm against this fixture. Both
/// run the identical Rust code on the identical input, and it measures 0.0 — so anything
/// approaching this bound is a real divergence, not noise.
const TOLERANCE: f32 = 1e-6;

/// The tolerance for checking the fixture against *native*, which is a different question: this
/// test runs on Linux and macOS, and their libm implementations differ by an ULP or two in the
/// `sin`/`cos`/`sinh` the Butterworth design calls. Measured on this chain:
///
/// | perturbation                    | worst output difference |
/// |---------------------------------|-------------------------|
/// | cutoff by 8 ULP (a libm proxy)  | 3.6e-5                  |
/// | order 4 -> 3 (a real change)    | 8.2e-2                  |
///
/// 1e-3 sits in the gap: 27x above the platform noise, 80x below the smallest change that would
/// mean the DSP actually moved. Tightening it would make this test fail on macOS for reasons that
/// have nothing to do with fluxion.
const STALENESS_TOLERANCE: f32 = 1e-3;

/// A deterministic input — and deterministic **across platforms**, which rules out `sin`.
///
/// An earlier version summed three sines. Every value differed by an ULP between glibc and Apple's
/// libm, and a 4th-order high-pass at 80 Hz amplifies that by ~385x (6e-8 in, 2.3e-5 out), so the
/// fixture could not be shared between them. Integer arithmetic and power-of-two divisions are
/// exact in f32 everywhere, so this is bit-identical on every target — including wasm.
///
/// The content is broadband noise (which exercises a filter better than a few tones anyway) over a
/// slow triangle the high-pass has something to remove.
fn input() -> Vec<f32> {
    let mut state: u32 = 0x1234_5678;
    (0..FRAMES)
        .map(|n| {
            // Numerical Recipes LCG; the top 24 bits over 2^24 is exact in f32.
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = (state >> 8) as f32 / 16_777_216.0 * 2.0 - 1.0;
            // ~23 Hz triangle at 48 kHz; 1024 is a power of two, so the division is exact.
            let phase = (n % 2048) as i32 - 1024;
            let triangle = phase.abs() as f32 / 1024.0 * 2.0 - 1.0;
            noise * 0.3 + triangle * 0.5
        })
        .collect()
}

fn native() -> Vec<f32> {
    let graph: Graph = CHAIN.parse().expect("the fixture chain parses");
    let out = process(&graph, &Signal::new(FS, vec![input()]));
    out.channels.into_iter().next().unwrap()
}

/// The committed fixture, as `{"chain", "fs", "input": [...], "expected": [...]}`.
fn fixture() -> serde_json::Value {
    serde_json::from_str(include_str!("../js/fixtures/parity.json")).expect("the fixture is JSON")
}

fn floats(value: &serde_json::Value, key: &str) -> Vec<f32> {
    value[key]
        .as_array()
        .unwrap_or_else(|| panic!("fixture has no array '{key}'"))
        .iter()
        .map(|v| v.as_f64().expect("a number") as f32)
        .collect()
}

/// Pre-condition: the committed fixture was written from this chain on this input.
/// Post-condition: native still produces it. If this goes red the DSP changed, and the Node-side
/// comparison would be measuring against a stale reference rather than against native.
#[test]
fn the_fixture_still_matches_native() {
    let fixture = fixture();
    assert_eq!(fixture["chain"], CHAIN);
    assert_eq!(fixture["fs"], FS);
    assert_eq!(
        floats(&fixture, "input"),
        input(),
        "the fixture input drifted"
    );

    let expected = floats(&fixture, "expected");
    let actual = native();
    assert_eq!(expected.len(), actual.len());
    let worst = expected
        .iter()
        .zip(&actual)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst <= STALENESS_TOLERANCE,
        "the fixture no longer matches native: worst difference {worst:e} > \
         {STALENESS_TOLERANCE:e}. If the DSP changed on purpose, regenerate it (see the module \
         docs); a difference this large is not a platform libm artifact."
    );
}

/// The chain text survives the trip through the module's own accessor, which is what `parity.mjs`
/// checks on the wasm side.
#[test]
fn the_chain_text_is_canonical() {
    let graph: Graph = CHAIN.parse().unwrap();
    let rendered = graph.to_string();
    assert_eq!(rendered.parse::<Graph>().unwrap(), graph);
    assert_eq!(rendered, "highpass(80, 4) | gain(0.70794576)");
}

/// See the module docs. Writes `js/fixtures/parity.json`.
#[test]
#[ignore = "writes the committed fixture; run explicitly"]
fn write_fixture() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("js/fixtures/parity.json");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let json = serde_json::json!({
        "_comment": "Generated by `cargo test -p fluxion-wasm --test parity -- --ignored \
                     write_fixture`. Native reference output for the wasm parity check.",
        "chain": CHAIN,
        "fs": FS,
        // The tolerance `js/parity.mjs` uses for wasm-vs-native. The looser cross-platform one is
        // a Rust-side constant, since only this test needs it.
        "tolerance": TOLERANCE,
        "input": input(),
        "expected": native(),
    });
    std::fs::write(&path, serde_json::to_string_pretty(&json).unwrap()).unwrap();
    eprintln!("wrote {} frames to {}", FRAMES, path.display());
}

//! wasm renders what native renders — for **every op** (roadmap W6, the last piece of F-M1).
//!
//! "Preview and export come from the same DSP" is the reason to pick one engine instead of glueing
//! three libraries together, and it is only worth anything if it is checked. So: one bit-exact
//! input, every op in the catalog plus a set of composite chains, rendered natively here and in
//! wasm by `js/parity.mjs`, compared sample for sample.
//!
//! The native reference is **generated at test time**, not committed:
//!
//! ```text
//! cargo test -p fluxion-wasm --test parity -- --ignored write_reference
//! node crates/fluxion-wasm/js/parity.mjs
//! ```
//!
//! An earlier version committed the reference as a fixture. That needed a second, looser tolerance
//! to survive the difference between glibc's and Apple's `sin`, and it could go stale. Since the
//! wasm job has to build the module anyway, it has a Rust toolchain by definition — so comparing
//! against *current* native costs nothing and removes both problems.

use std::collections::BTreeSet;

use fluxion::{Graph, OpKind, Signal, process};

const FS: u32 = 48_000;

/// 85 ms. Long enough that the delay-based ops (delay, echo, reverb, chorus, flanger) have
/// something in their lines by the end — a shorter window would compare mostly dry signal and
/// call it a pass.
const FRAMES: usize = 4096;

/// The bound W6 asks us to write down, and what it is measured against.
///
/// 26 of the 32 cases are **bit-identical** to native. The six that are not are exactly the ops
/// that call a transcendental function *per sample*, where wasm uses the `libm` crate and native
/// uses the platform's:
///
/// | case           | worst difference | why                                  |
/// |----------------|------------------|--------------------------------------|
/// | `overdrive`    | 2.4e-7           | `tanh` soft-clipper                  |
/// | `phaser`       | 2.4e-7           | `sin` LFO, per sample                |
/// | `compand`      | 1.2e-7           | `exp`/`ln` in the envelope follower  |
/// | `fade`         | 6.0e-8           | quarter-sine curve                   |
/// | `tremolo`      | 4.5e-8           | `sin` LFO, per sample                |
/// | `chain:nested` | 4.5e-8           | contains `compand`                   |
///
/// f32's epsilon is 1.2e-7, so the worst of those is two ULP — the last bit of a `tanh`, not a
/// difference in the DSP. Every designed filter is bit-identical, which says the two libms agree
/// on the `sin`/`cos`/`sinh` the coefficient design calls; only the per-sample calls diverge.
///
/// 1e-6 leaves 4x headroom over the worst observed. Tightening it to sit just above 2.4e-7 would
/// make the suite fail the first time a platform's `tanh` rounds one bit differently, which is not
/// what this is watching for. `js/parity.mjs` prints the measured worst on every run, so a silent
/// drift toward the bound is visible rather than hidden behind a pass.
const TOLERANCE: f32 = 1e-6;

/// One comparison: an op (or a chain shape) and the text that builds it.
struct Case {
    /// What this covers — an op name from the registry, or `chain:<shape>` for a topology.
    name: &'static str,
    /// The chain, in the shared text syntax.
    chain: &'static str,
}

/// Every op, with parameters chosen to make it *do* something to the input — a `fade(0, 0, 1)` or
/// a `gain(1)` would pass this suite while proving nothing, so `every_case_changes_the_signal`
/// holds each of these to producing a different signal than it was given.
///
/// Frequencies sit inside the band the input occupies, and the delay-family times are short enough
/// to land inside `FRAMES`.
const CASES: &[Case] = &[
    // --- filters ---
    Case {
        name: "lowpass",
        chain: "lowpass(2000, 4)",
    },
    Case {
        name: "highpass",
        chain: "highpass(300, 4)",
    },
    Case {
        name: "peaking",
        chain: "peaking(1000, 9, 1.5)",
    },
    Case {
        name: "lowshelf",
        chain: "lowshelf(300, 6, 0.7)",
    },
    Case {
        name: "highshelf",
        chain: "highshelf(6000, -6, 0.7)",
    },
    Case {
        name: "notch",
        chain: "notch(1000, 8)",
    },
    Case {
        name: "bandpass",
        chain: "bandpass(1500, 2)",
    },
    Case {
        name: "allpass",
        chain: "allpass(1200, 0.7)",
    },
    Case {
        name: "cheby1_lowpass",
        chain: "cheby1_lowpass(3000, 4, 1)",
    },
    Case {
        name: "cheby1_highpass",
        chain: "cheby1_highpass(500, 4, 1)",
    },
    Case {
        name: "cheby2_lowpass",
        chain: "cheby2_lowpass(3000, 4, 40)",
    },
    Case {
        name: "cheby2_highpass",
        chain: "cheby2_highpass(500, 4, 40)",
    },
    Case {
        name: "biquad",
        chain: "biquad(0.5, -0.2, 0.1, -0.3, 0.05)",
    },
    Case {
        name: "fir",
        chain: "fir(0.5, 0.25, 0.125, -0.0625)",
    },
    // --- effects ---
    Case {
        name: "gain",
        chain: "gain(-6dB)",
    },
    Case {
        name: "normalize",
        chain: "normalize(0.5)",
    },
    Case {
        name: "delay",
        chain: "delay(0.002, 0.5)",
    },
    Case {
        name: "echo",
        chain: "echo(0.003, 0.5, 0.6)",
    },
    Case {
        name: "reverb",
        chain: "reverb(0.7, 0.3, 0.5)",
    },
    Case {
        name: "fade",
        chain: "fade(0.01, 0.02, 1)",
    },
    Case {
        name: "tremolo",
        chain: "tremolo(120, 0.8)",
    },
    Case {
        name: "overdrive",
        chain: "overdrive(18, 0.3)",
    },
    Case {
        name: "compand",
        chain: "compand(0.002, 0.02, -18, 6, 6, 3)",
    },
    Case {
        name: "reverse",
        chain: "reverse",
    },
    Case {
        name: "chorus",
        chain: "chorus(3, 0.002, 0.004, 0.5)",
    },
    Case {
        name: "flanger",
        chain: "flanger(2, 0.001, 0.001, 0.5, 0.5)",
    },
    Case {
        name: "phaser",
        chain: "phaser(3, 0.7, 0.5, 0.5)",
    },
    // --- topologies, not ops: the algebra has to survive the trip too ---
    Case {
        name: "chain:series",
        chain: "highpass(300, 4) | gain(-3dB)",
    },
    Case {
        name: "chain:parallel",
        chain: "lowpass(500, 2) + highpass(3000, 2)",
    },
    Case {
        name: "chain:nested",
        chain: "(lowpass(800, 2) + highpass(4000, 2)) | compand(0.01, 0.1, -20, 4, 6, 0) | gain(0.5)",
    },
    Case {
        name: "chain:labeled",
        chain: "eq: peaking(1000, 6, 1.5) | gain(0.8)",
    },
    // The one construct a series/parallel tree cannot encode, and the one that is sample-recursive
    // — so if any topology were going to drift between the two builds, it would be this.
    Case {
        name: "chain:feedback",
        chain: "(lowpass(2000, 2) ~ gain(0.5))",
    },
];

/// A deterministic input — and deterministic **across platforms**, which rules out `sin`.
///
/// An earlier version summed three sines. Every value differed by an ULP between glibc and Apple's
/// libm, and a 4th-order high-pass at 80 Hz amplifies that by ~385x (6e-8 in, 2.3e-5 out), so the
/// same numbers could not be shared between them. Integer arithmetic and power-of-two divisions
/// are exact in f32 everywhere, so this is bit-identical on every target — including wasm.
///
/// The content is broadband noise (which exercises a filter better than a few tones) over a slow
/// triangle, so the high-pass family has something to remove and the low-pass family has something
/// to keep.
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

fn render(chain: &str, x: &[f32]) -> Vec<f32> {
    let graph: Graph = chain
        .parse()
        .unwrap_or_else(|e| panic!("case chain `{chain}` does not parse: {e}"));
    process(&graph, &Signal::new(FS, vec![x.to_vec()]))
        .channels
        .into_iter()
        .next()
        .expect("one channel in, one channel out")
}

/// Pre-condition: `CASES` is meant to cover the whole catalog.
/// Post-condition: it does. Adding an op without adding a case turns this red — which is what
/// makes "every op exposed to wasm is compared" a fact rather than an intention.
#[test]
fn every_registry_op_has_a_case() {
    let covered: BTreeSet<&str> = CASES
        .iter()
        .map(|c| c.name)
        .filter(|n| !n.starts_with("chain:"))
        .collect();
    let registry: BTreeSet<&str> = OpKind::all().iter().map(|k| k.name()).collect();

    let missing: Vec<_> = registry.difference(&covered).collect();
    assert!(
        missing.is_empty(),
        "these ops are not compared against wasm: {missing:?} — add a case to CASES"
    );
    assert_eq!(
        covered, registry,
        "CASES names an op the registry does not have"
    );
}

/// Every case must be built from the op it claims to cover, so a case cannot drift into testing
/// something else after a rename.
#[test]
fn every_case_builds_the_op_it_names() {
    for case in CASES {
        if case.name.starts_with("chain:") {
            continue;
        }
        assert!(
            case.chain.starts_with(case.name),
            "case '{}' builds `{}`, which is not that op",
            case.name,
            case.chain
        );
    }
}

/// Pre-condition: each case is supposed to exercise its op.
/// Post-condition: each one changes the signal. A case at parameters that happen to be a no-op
/// (`gain(1)`, `fade(0, 0, 1)`) would pass the comparison suite while proving nothing.
#[test]
fn every_case_changes_the_signal() {
    let dry = input();
    for case in CASES {
        let wet = render(case.chain, &dry);
        assert_eq!(
            wet.len(),
            dry.len(),
            "case '{}' changed the length",
            case.name
        );
        let moved = dry
            .iter()
            .zip(&wet)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            moved > 1e-4,
            "case '{}' (`{}`) barely touches the signal (worst change {moved:e}) — \
             pick parameters that do something",
            case.name,
            case.chain
        );
    }
}

/// Write the native reference `js/parity.mjs` compares against. Not a fixture: it is regenerated
/// every run, and `js/reference.json` is git-ignored.
#[test]
#[ignore = "generates js/reference.json for the wasm comparison; run explicitly"]
fn write_reference() {
    let dry = input();
    let cases: Vec<serde_json::Value> = CASES
        .iter()
        .map(|c| {
            serde_json::json!({
                "name": c.name,
                "chain": c.chain,
                "expected": render(c.chain, &dry),
            })
        })
        .collect();

    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("js/reference.json");
    let json = serde_json::json!({
        "_comment": "Generated by `cargo test -p fluxion-wasm --test parity -- --ignored \
                     write_reference`. Native output for the wasm comparison suite (roadmap W6). \
                     Not committed — regenerate it, do not edit it.",
        "fs": FS,
        "frames": FRAMES,
        "tolerance": TOLERANCE,
        "input": dry,
        "cases": cases,
    });
    std::fs::write(&path, serde_json::to_string(&json).unwrap()).unwrap();
    eprintln!(
        "wrote {} cases x {FRAMES} frames to {}",
        CASES.len(),
        path.display()
    );
}

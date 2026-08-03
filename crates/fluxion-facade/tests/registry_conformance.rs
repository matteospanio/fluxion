//! The Rust prelude covers the op registry, exactly — no gaps, no synonyms.
//!
//! Rust cannot enumerate its own functions, so this list is written by hand. That is the point:
//! adding an op to the catalog without adding a builder here turns the build red, and the fix is
//! two lines. Before this check the prelude was quietly missing `reverb`, `cheby2_lowpass` and
//! `cheby2_highpass` — three ops the CLI and Python had and Rust did not.

use std::collections::BTreeSet;

use fluxion::prelude::*;
use fluxion::{Graph, OpKind};

/// One call per prelude builder, at values inside its bounds.
fn every_builder() -> Vec<Graph> {
    vec![
        gain(0.5),
        lowpass(800.0, 4),
        highpass(80.0, 2),
        normalize(1.0),
        peaking(1000.0, 6.0, 1.5),
        lowshelf(200.0, 6.0, 0.7),
        highshelf(8000.0, -3.0, 0.7),
        notch(50.0, 30.0),
        bandpass(1000.0, 1.0),
        allpass(1000.0, 0.707),
        delay(0.25, 0.5),
        echo(0.25, 0.3, 0.5),
        cheby1_lowpass(1000.0, 4, 1.0),
        cheby1_highpass(1000.0, 4, 1.0),
        cheby2_lowpass(1000.0, 4, 40.0),
        cheby2_highpass(1000.0, 4, 40.0),
        reverb(0.5, 0.3, 0.3),
        fir([0.5, 0.3, 0.2]),
        fade(0.1, 0.2, 1.0),
        tremolo(5.0, 0.5),
        overdrive(20.0, 0.2),
        compand(0.01, 0.1, -20.0, 4.0, 6.0, 0.0),
        reverse(),
        biquad(1.0, 0.0, 0.0, 0.0, 0.0),
        chorus(1.5, 0.002, 0.025, 0.5),
        flanger(0.5, 0.002, 0.001, 0.5, 0.5),
        phaser(0.5, 0.5, 0.5, 0.5),
    ]
}

/// Pre-condition: `every_builder` calls each `fluxion::prelude` constructor once.
/// Post-condition: the ops they build are exactly the registry, one builder per op.
#[test]
fn the_prelude_covers_the_registry_exactly() {
    let mut built = BTreeSet::new();
    for graph in every_builder() {
        match graph {
            Graph::Op(op) => assert!(
                built.insert(op.kind.name()),
                "two prelude builders return op '{}'",
                op.kind.name()
            ),
            other => panic!("a prelude builder returned a composite graph: {other}"),
        }
    }
    let registry: BTreeSet<&str> = OpKind::all().iter().map(|k| k.name()).collect();

    let missing: Vec<_> = registry.difference(&built).collect();
    assert!(
        missing.is_empty(),
        "the fluxion::prelude has no builder for: {missing:?}"
    );
    assert_eq!(built, registry, "the prelude drifted from the op registry");
}

/// The builder's name is the registry's name, so a user who learned the chain text, the CLI or
/// Python already knows what to type in Rust. Underscores are the only allowed difference — Rust
/// spells `cheby1_lowpass` the same way the registry now does.
#[test]
fn every_builder_is_reachable_by_the_registry_name() {
    for graph in every_builder() {
        let Graph::Op(op) = &graph else {
            unreachable!()
        };
        let text = graph.to_string();
        assert!(
            text.starts_with(op.kind.name()),
            "op '{}' renders as `{text}`, which does not start with its registry name",
            op.kind.name()
        );
        // And the text form of what the builder produced parses back to it.
        assert_eq!(text.parse::<Graph>().unwrap(), graph);
    }
}

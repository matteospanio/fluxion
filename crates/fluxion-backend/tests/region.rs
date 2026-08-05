//! Rendering part of a chain (ROADMAP D4).
//!
//! The check D4 names: "rendering `[0,N)` whole equals rendering it in random pieces, bit for bit".
//! Taken literally — `assert_eq!` on `f32`, over ragged pieces in a scrambled order, through a
//! chain of ops that all carry state.

use fluxion_backend::{
    RegionError, process, process_automated, render_region, render_region_automated,
};
use fluxion_core::automation::{Automation, Curve, Lane, Point, Timing};
use fluxion_core::{Graph, OpKind, Signal};
use std::f32::consts::TAU;

const FS: u32 = 48_000;

/// Something with content across the band, so a filter has work to do and a difference would show.
fn material(frames: usize) -> Signal {
    let mut state = 0x9e37_79b9u32;
    Signal::new(
        FS,
        vec![
            (0..frames)
                .map(|i| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    let noise = (state >> 8) as f32 / 8_388_608.0 - 1.0;
                    (TAU * 220.0 * i as f32 / FS as f32).sin() * 0.4 + noise * 0.1
                })
                .collect(),
        ],
    )
}

/// A chain where every op remembers something: a filter's biquad state, an echo's ring, a
/// compressor's envelope, and a parallel split so both branches have to line up too.
fn stateful_chain() -> Graph {
    Graph::op(OpKind::Highpass, [80.0, 4.0])
        | (Graph::op(OpKind::Lowpass, [3_000.0, 4.0])
            + Graph::op(OpKind::Peaking, [900.0, 6.0, 1.2]))
        | Graph::op(OpKind::Echo, [0.05, 0.4, 0.3])
        | Graph::op(OpKind::Compand, [0.01, 0.1, -18.0, 4.0, 6.0, 0.0])
}

/// D4's check, in its own words.
#[test]
fn a_whole_render_equals_the_same_render_in_random_pieces() {
    let frames = 40_000;
    let input = material(frames);
    let graph = stateful_chain();

    let whole = process(&graph, &input);

    // Ragged pieces, in a deliberately scrambled order — nothing about the result may depend on
    // the order they were asked for, or on the boundaries falling anywhere convenient.
    let cuts = [
        0usize, 1, 999, 1_000, 7_777, 12_288, 12_289, 30_000, 39_999, 40_000,
    ];
    let mut pieces = vec![f32::NAN; frames];
    let mut order: Vec<usize> = (0..cuts.len() - 1).collect();
    // A fixed scramble, so a failure is reproducible.
    order.swap(0, 5);
    order.swap(1, 7);
    order.swap(2, 4);

    for &i in &order {
        let (from, to) = (cuts[i], cuts[i + 1]);
        let piece = render_region(&graph, &input, from, to).expect("the chain is regionable");
        assert_eq!(
            piece.frames(),
            to - from,
            "piece [{from}, {to}) has the wrong length"
        );
        pieces[from..to].copy_from_slice(&piece.channels[0]);
    }

    assert!(
        pieces.iter().all(|s| !s.is_nan()),
        "a frame was never written"
    );
    for (n, (a, b)) in whole.channels[0].iter().zip(&pieces).enumerate() {
        assert_eq!(a, b, "frame {n}: whole {a}, in pieces {b}");
    }
}

/// The same, with the parameters moving — a region has to see the curve values the whole render
/// saw at those frames, not the curve restarted at the window.
#[test]
fn an_automated_render_matches_in_pieces_too() {
    let frames = 24_000;
    let input = material(frames);
    let graph = Graph::named("g", Graph::op(OpKind::Gain, [1.0]))
        | Graph::named("lp", Graph::op(OpKind::Lowpass, [4_000.0, 4.0]));
    let automation = Automation::new()
        .with(Lane::new("g", "gain", Curve::db_ramp(1.0, 0.01, 0.5)))
        .with(Lane::new(
            "lp",
            "cutoff",
            Curve::new(
                [Point::new(0.0, 4_000.0), Point::new(0.5, 400.0)],
                Timing::Once,
            ),
        ));

    let whole = process_automated(&graph, &input, &automation).unwrap();

    let mut pieces = vec![f32::NAN; frames];
    for &(from, to) in &[
        (12_000usize, 24_000usize),
        (0, 64),
        (64, 11_111),
        (11_111, 12_000),
    ] {
        let piece = render_region_automated(&graph, &input, &automation, from, to).unwrap();
        pieces[from..to].copy_from_slice(&piece.channels[0]);
    }
    for (n, (a, b)) in whole.channels[0].iter().zip(&pieces).enumerate() {
        assert_eq!(a, b, "frame {n} differs once automated");
    }
}

/// A single-frame window and a zero-length one are both legitimate asks.
#[test]
fn degenerate_windows_are_windows() {
    let input = material(1_000);
    let graph = Graph::op(OpKind::Lowpass, [1_000.0, 2.0]);
    let whole = process(&graph, &input);

    let one = render_region(&graph, &input, 500, 501).unwrap();
    assert_eq!(one.frames(), 1);
    assert_eq!(one.channels[0][0], whole.channels[0][500]);

    assert_eq!(render_region(&graph, &input, 500, 500).unwrap().frames(), 0);

    // Past the end clamps rather than panicking or padding.
    let over = render_region(&graph, &input, 900, 5_000).unwrap();
    assert_eq!(over.frames(), 100);
    assert_eq!(over.channels[0], whole.channels[0][900..]);

    assert_eq!(
        render_region(&graph, &input, 600, 500).unwrap_err(),
        RegionError::Backwards { from: 600, to: 500 }
    );
}

/// An op whose output depends on the whole signal is refused by name, rather than returning a
/// window that looks plausible and is wrong.
#[test]
fn a_whole_signal_op_is_refused_with_its_reason() {
    let input = material(1_000);
    for (op, name) in [
        (Graph::op(OpKind::Normalize, [1.0]), "normalize"),
        (Graph::op(OpKind::Reverse, []), "reverse"),
        (Graph::op(OpKind::Loudnorm, [-14.0, -1.0]), "loudnorm"),
        (Graph::op(OpKind::Limiter, [-1.0, 0.005, 0.05]), "limiter"),
    ] {
        let err =
            render_region(&(Graph::op(OpKind::Gain, [1.0]) | op), &input, 0, 100).unwrap_err();
        assert!(
            matches!(err, RegionError::WholeSignalOp { op: o, .. } if o == name),
            "{name}: {err}"
        );
        // The message says what to do about it.
        assert!(err.to_string().contains("render the whole signal"), "{err}");
    }
}

/// The cost is not hidden: rendering a late window computes everything before it, and the API says
/// so rather than leaving a caller to discover it by timing.
#[test]
fn the_cost_of_a_late_window_is_stated() {
    use fluxion_backend::region::frames_to_compute;
    assert_eq!(frames_to_compute(0, 1_000), 1_000);
    assert_eq!(
        frames_to_compute(900_000, 1_000_000),
        1_000_000,
        "a 100k window at the end of a timeline still computes the timeline"
    );
}

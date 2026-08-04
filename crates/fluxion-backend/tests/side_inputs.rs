//! Side inputs and keys in the chain algebra (ROADMAP S1, and S3's use of them).
//!
//! Two things have to be true at once. A chain that uses a second input has to *get* it, sample for
//! sample — the check S1 names. And every chain that does not use one has to behave exactly as it
//! did before, which is the part that would be easy to break and hard to notice.

use fluxion_backend::{process, process_with};
use fluxion_core::{Graph, OpKind, Signal};
use std::f32::consts::TAU;

const FS: u32 = 48_000;

fn tone(freq: f32, amp: f32, frames: usize) -> Signal {
    Signal::new(
        FS,
        vec![
            (0..frames)
                .map(|i| (TAU * freq * i as f32 / FS as f32).sin() * amp)
                .collect(),
        ],
    )
}

fn peak(s: &Signal) -> f32 {
    s.channels
        .iter()
        .flatten()
        .fold(0.0f32, |m, v| m.max(v.abs()))
}

/// The two-input check: a graph that reads both inputs sees frame `i` of each at the same time.
///
/// `a + side(0)` is the smallest two-input op there is — the sum has to be the sum of the two
/// signals *at the same instant*, which is a property no amount of level checking would catch. A
/// one-frame slip between them shows up here as a large error, because the two tones are chosen to
/// interfere: 1 kHz against 1 kHz phase-shifted by half a period cancels exactly when aligned.
#[test]
fn a_two_input_graph_sees_both_signals_sample_aligned() {
    let frames = 4_800;
    let main = tone(1000.0, 0.5, frames);
    // The same tone inverted: summed in alignment it cancels to nothing, and one frame out it does
    // not — at 1 kHz a single frame of slip leaves 6.5 % of the amplitude behind.
    let inverted = Signal::new(
        FS,
        vec![main.channels[0].iter().map(|s| -s).collect::<Vec<f32>>()],
    );

    let sum = Graph::Id + Graph::side(0);
    let out = process_with(&sum, &main, &[&inverted]);

    assert_eq!(out.frames(), frames);
    assert!(
        peak(&out) < 1e-6,
        "aligned signals should cancel; {} left over",
        peak(&out)
    );

    // And the control: deliberately slipped by one frame, the same graph does not cancel.
    let slipped = Signal::new(
        FS,
        vec![
            std::iter::once(0.0)
                .chain(inverted.channels[0].iter().copied())
                .take(frames)
                .collect(),
        ],
    );
    let out = process_with(&sum, &main, &[&slipped]);
    assert!(
        peak(&out) > 0.01,
        "one frame of slip should be visible, and it was not: {}",
        peak(&out)
    );
}

/// A side signal is read at its own frame positions, and it is silence where it does not reach —
/// not a repeat, not the last value held.
#[test]
fn a_side_input_is_itself_and_silence_past_its_end() {
    let main = tone(440.0, 0.5, 1_000);
    let short = tone(440.0, 0.5, 400);

    let out = process_with(&Graph::side(0), &main, &[&short]);
    assert_eq!(out.frames(), 1_000);
    assert_eq!(&out.channels[0][..400], &short.channels[0][..]);
    assert!(out.channels[0][400..].iter().all(|s| *s == 0.0));
}

/// A `side(n)` nobody connected reads as silence rather than failing: a chain written for two
/// inputs still runs on one, which is what lets the same chain text be shared with an interface
/// that has no way to pass a second signal.
#[test]
fn an_unconnected_side_input_is_silence() {
    let main = tone(440.0, 0.5, 500);
    let out = process(&(Graph::Id + Graph::side(0)), &main);
    assert_eq!(out.channels[0], main.channels[0]);
}

/// A mono key on a stereo programme is the common case, and it means the same key for both.
#[test]
fn a_mono_side_input_feeds_every_channel() {
    let main = Signal::new(FS, vec![vec![0.25; 100], vec![0.5; 100]]);
    let side = Signal::new(FS, vec![vec![1.0; 100]]);
    let out = process_with(&Graph::side(0), &main, &[&side]);

    assert_eq!(out.channels.len(), 2);
    assert!(out.channels.iter().all(|c| c.iter().all(|s| *s == 1.0)));
}

/// The half that is easy to break: keying a subchain must be invisible to every op that does not
/// declare a key input. Same chain, same samples, keyed or not.
#[test]
fn keying_a_chain_of_unkeyed_ops_changes_nothing() {
    let main = tone(300.0, 0.4, 2_000);
    let key = tone(50.0, 0.9, 2_000);

    let chain = Graph::op(OpKind::Lowpass, [800.0, 4.0])
        | Graph::op(OpKind::Gain, [0.5])
        | Graph::op(OpKind::Compand, [0.01, 0.1, -20.0, 4.0, 6.0, 0.0]);

    let plain = process(&chain, &main);
    let keyed = process_with(&chain.clone().keyed(Graph::side(0)), &main, &[&key]);
    assert_eq!(plain.channels, keyed.channels);
}

/// The key is evaluated on the input the node was given, so a filter on the key side filters the
/// **key** — the reading that makes `gate < side(0) | lowpass(200)` mean what it looks like.
#[test]
fn the_key_chain_runs_on_the_key_not_on_the_programme() {
    let frames = FS as usize / 4;
    let programme = tone(1000.0, 0.5, frames);
    // A key with all its energy at 5 kHz: low-passed to 200 Hz there is nothing left of it, so a
    // gate keyed by it stays shut. If the key chain ran on the programme instead, the programme's
    // own 1 kHz would also be filtered away and the gate would *still* shut — so the discriminating
    // case is the opposite one, below.
    let key = tone(5_000.0, 0.9, frames);

    let gate = Graph::op(OpKind::Gate, [-40.0, 60.0, 0.001, 0.0, 0.005]);
    let shut = process_with(
        &gate
            .clone()
            .keyed(Graph::side(0) | Graph::op(OpKind::Lowpass, [200.0, 4.0])),
        &programme,
        &[&key],
    );
    let reduction = 20.0
        * (peak(&programme)
            / peak(&Signal::new(
                FS,
                vec![shut.channels[0][frames / 2..].to_vec()],
            )))
        .log10();
    assert!(
        reduction > 50.0,
        "a key with nothing under 200 Hz should shut the gate; it reduced by {reduction:.1} dB"
    );

    // Now key it with something the low-pass *keeps*. The programme is unchanged, so the only thing
    // that can open the gate is the key having run through its own chain.
    let bass = tone(50.0, 0.9, frames);
    let open = process_with(
        &gate.keyed(Graph::side(0) | Graph::op(OpKind::Lowpass, [200.0, 4.0])),
        &programme,
        &[&bass],
    );
    let worst = open.channels[0][frames / 2..]
        .iter()
        .zip(&programme.channels[0][frames / 2..])
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst < 1e-4,
        "a key that survives its own low-pass should hold the gate open; off by {worst}"
    );
}

/// S3's own check at the graph level: the gate follows the key, not the programme it is gating.
#[test]
fn a_keyed_gate_follows_the_key() {
    let frames = FS as usize / 2;
    let loud = tone(1000.0, 0.5, frames);
    let silent = Signal::new(FS, vec![vec![0.0; frames]]);
    let gate = Graph::op(OpKind::Gate, [-40.0, 60.0, 0.001, 0.0, 0.005]);

    // Loud programme, silent key: shut.
    let out = process_with(&gate.clone().keyed(Graph::side(0)), &loud, &[&silent]);
    assert!(
        peak(&Signal::new(
            FS,
            vec![out.channels[0][frames / 2..].to_vec()]
        )) < 0.001,
        "a silent key left the gate open"
    );

    // The same gate, unkeyed, on the same programme: wide open, because now it is listening to the
    // programme. Same op, same parameters — only the routing differs.
    let out = process(&gate, &loud);
    let worst = out.channels[0][frames / 4..]
        .iter()
        .zip(&loud.channels[0][frames / 4..])
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst < 1e-3,
        "an unkeyed gate on loud material closed: {worst}"
    );
}

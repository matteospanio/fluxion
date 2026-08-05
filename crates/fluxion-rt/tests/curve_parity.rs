//! One curve, two engines (ROADMAP S4 and D3).
//!
//! Both roadmap rows ask for the same thing in the same words — S4: "the same description gives
//! identical curves in the batch and realtime engines"; D3: "the same breakpoints give identical
//! envelopes offline and in the realtime engine" — so they are checked together, here.
//!
//! **Identical means bit-identical.** Not "within a tolerance": the realtime ramp and the offline
//! curve call the same function in `fluxion_core::automation`, so there is nothing left that could
//! make them differ, and a tolerance would only hide it if something did. The tests below assert
//! `==` on `f32` deliberately.
//!
//! What makes that possible is that neither side accumulates. The value at sample `n` is computed
//! from `n`, so a ramp cannot drift, a block boundary cannot shift it, and a seek cannot
//! desynchronize it. The measurement of what accumulating would have cost is in
//! `an_accumulating_ramp_would_not_have_matched`.

use fluxion_core::automation::{Curve, Point, Shape, Timing};
use fluxion_rt::param::SmoothedValue;

const FS: u32 = 48_000;

/// Run the live ramp for `n` samples, the way the audio thread does.
fn live(from: f32, to: f32, n: u32, shape: Shape) -> Vec<f32> {
    let mut v = SmoothedValue::new(from);
    v.set_target_shaped(to, n, shape);
    (0..n).map(|_| v.tick()).collect()
}

/// The same ramp as an offline curve, read at frames 1..=n.
///
/// Frame 1 rather than 0 because the two describe the same instant differently: `set_target` lands
/// on a block boundary and the *next* sample the callback produces is the first one on the ramp,
/// so the live ramp's k-th output is the curve at frame k. Getting this off by one would show up
/// as a one-sample lag between the preview and the render, which is exactly the class of bug D3
/// exists to prevent.
fn offline(from: f32, to: f32, n: u32, shape: Shape) -> Vec<f32> {
    let curve = Curve::new(
        [
            Point::shaped(0.0, from, shape),
            Point::new(f64::from(n) / f64::from(FS), to),
        ],
        Timing::Once,
    )
    .compile(FS);
    (1..=u64::from(n)).map(|f| curve.at(f)).collect()
}

/// The check both S4 and D3 name: the same description, the same numbers, bit for bit.
#[test]
fn a_live_ramp_is_bit_identical_to_the_offline_curve() {
    for shape in [
        Shape::Linear,
        Shape::Cosine,
        Shape::Exp { k: -3.0 },
        Shape::Exp { k: 2.5 },
        Shape::Step,
    ] {
        for (from, to, n) in [
            (0.0f32, 1.0f32, 4u32),
            (0.0, 1.0, 48_000),
            (1.0, 0.0, 48_000),
            (-0.25, 0.75, 12_345),
            (0.5, 0.5, 1_000),
        ] {
            let live = live(from, to, n, shape);
            let offline = offline(from, to, n, shape);
            assert_eq!(
                live.len(),
                offline.len(),
                "{shape:?} {from}->{to} over {n}: lengths differ"
            );
            for (i, (l, o)) in live.iter().zip(&offline).enumerate() {
                assert_eq!(
                    l, o,
                    "{shape:?} {from}->{to} over {n}: sample {i} is {l} live and {o} offline"
                );
            }
        }
    }
}

/// The measurement that justifies the design. Had the live ramp accumulated — `current += step`,
/// which is the obvious way to write it and what this used to do — it would not have matched, and
/// the gap is far too big to call rounding.
#[test]
fn an_accumulating_ramp_would_not_have_matched() {
    let n = 48_000u32;
    let exact = offline(0.0, 1.0, n, Shape::Linear);

    let step = 1.0f32 / n as f32;
    let mut current = 0.0f32;
    let mut worst = 0.0f32;
    let mut identical = 0usize;
    for (i, want) in exact.iter().enumerate() {
        // The old arithmetic, including its snap-to-target on the final sample.
        if i as u32 == n - 1 {
            current = 1.0;
        } else {
            current += step;
        }
        worst = worst.max((current - want).abs());
        if current == *want {
            identical += 1;
        }
    }

    assert!(
        worst > 1e-4,
        "expected accumulation to drift by ~6.45e-4 over a 1 s ramp; it drifted {worst:e}"
    );
    assert!(
        identical < n as usize / 100,
        "expected almost no samples to land bit-identical; {identical} of {n} did"
    );
    println!(
        "accumulating drifts {worst:e} over {n} samples; {identical}/{n} samples bit-identical"
    );

    // And the live ramp, which does not accumulate, matches every one of them.
    assert_eq!(live(0.0, 1.0, n, Shape::Linear), exact);
}

/// S4's own subject: an LFO is a parameter source, and it has to give the same numbers wherever it
/// is read from — including a long way into a session, where a phase accumulator would have
/// wandered off.
#[test]
fn an_lfo_reads_the_same_at_any_point_in_a_session() {
    let lfo = Curve::lfo(3.0, 0.2, 0.8, 0.0).compile(FS);
    let cycle = u64::from(FS) / 3;

    // An hour in, the cycle is the same shape it was at the start — every sample of it.
    let hour = 3_600 * u64::from(FS);
    let base = hour - hour % cycle;
    for offset in 0..cycle {
        assert_eq!(
            lfo.at(base + offset),
            lfo.at(offset),
            "sample {offset} of the cycle an hour in differs from the first cycle"
        );
    }
}

/// And an ADSR: held, the level is flat however long the note lasts; released, it falls from where
/// it was. Read the same way by whichever engine asks.
#[test]
fn an_adsr_is_flat_while_held_and_falls_when_released() {
    let env = Curve::adsr(0.005, 0.05, 0.4, 0.15).compile(FS);
    let held: Vec<f32> = (0..10)
        .map(|s| env.at_gated(u64::from(FS) * (1 + s), None))
        .collect();
    assert!(
        held.windows(2).all(|w| w[0] == w[1]),
        "a held note should not move: {held:?}"
    );
    assert!((held[0] - 0.4).abs() < 1e-3, "sustain level {}", held[0]);

    let release = u64::from(FS) * 5;
    assert_eq!(env.at_gated(release, Some(release)), held[0]);
    let ended = release + (0.15 * f64::from(FS)) as u64;
    assert!(env.at_gated(ended, Some(release)).abs() < 1e-3);
}

/// A curve is read at absolute frames, so a render that starts in the middle sees what a render
/// from the beginning saw at that point. This is what D4's region rendering leans on.
#[test]
fn a_curve_does_not_care_where_the_render_started() {
    let curve = Curve::new(
        [
            Point::shaped(0.0, 0.0, Shape::Cosine),
            Point::shaped(0.5, 1.0, Shape::Exp { k: -2.0 }),
            Point::new(1.5, 0.25),
        ],
        Timing::Once,
    )
    .compile(FS);

    let whole: Vec<f32> = (0..u64::from(FS) * 2).map(|n| curve.at(n)).collect();
    // Read the same span in ragged pieces, in a scrambled order.
    let mut pieces = vec![0.0f32; whole.len()];
    for &(from, len) in &[
        (60_000u64, 12_000u64),
        (0, 7),
        (7, 59_993),
        (72_000, 24_000),
    ] {
        for i in 0..len {
            pieces[(from + i) as usize] = curve.at(from + i);
        }
    }
    assert_eq!(whole, pieces);
}

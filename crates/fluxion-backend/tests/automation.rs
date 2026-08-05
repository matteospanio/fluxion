//! Automation: curves driving op parameters (ROADMAP D2).
//!
//! The headline check is D2's own — a gain automated 0 to -60 dB over a second has to match the
//! exact envelope *sample by sample*, not approximately. It does, because a gain is a multiply and
//! there is nothing to approximate: the renderer asks the curve for the value at each frame.
//!
//! The filters are the harder half, and honest about it: a cutoff is an input to a coefficient
//! design, so it is redesigned every 64 frames rather than every sample. The staircase that
//! creates is measured here rather than asserted away.

use fluxion_backend::{AutomationError, process, process_automated, process_automated_from};
use fluxion_core::automation::{Automation, Curve, Lane, Point, Shape, Timing};
use fluxion_core::{Graph, OpKind, Signal};
use std::f32::consts::TAU;

const FS: u32 = 48_000;

fn dc(frames: usize) -> Signal {
    Signal::new(FS, vec![vec![1.0f32; frames]])
}

fn tone(freq: f32, frames: usize) -> Signal {
    Signal::new(
        FS,
        vec![
            (0..frames)
                .map(|i| (TAU * freq * i as f32 / FS as f32).sin() * 0.5)
                .collect(),
        ],
    )
}

/// D2's check, in its own words: "a gain automated 0 → −60 dB over 1 s matches the exact envelope
/// sample by sample".
///
/// The curve is drawn in decibels — which is how a fade is described and how a user thinks about
/// it — and the exact envelope is therefore `10^(dB/20)` at each frame. Rendering a constant 1.0
/// through it makes the output *be* the envelope, so this compares the two directly with no
/// signal in the way.
#[test]
fn a_gain_automated_over_a_second_matches_the_exact_envelope() {
    let frames = FS as usize;
    let graph = Graph::named("fade", Graph::op(OpKind::Gain, [1.0]));

    // 0 dB to -60 dB over one second. `gain` is a linear parameter, so the lane carries the linear
    // values those decibels mean — and the fade is `db_ramp`, which travels between them at a
    // constant rate in decibels. A straight line in amplitude would be a different fade: it is at
    // -6 dB half way through, where this is at -30.
    let silent = 10f32.powf(-60.0 / 20.0);
    let curve = Curve::db_ramp(1.0, silent, 1.0);
    let automation = Automation::new().with(Lane::new("fade", "gain", curve.clone()));

    let out = process_automated(&graph, &dc(frames), &automation).expect("the lane resolves");
    assert_eq!(out.frames(), frames);

    let compiled = curve.compile(FS);
    for n in 0..frames {
        let want = compiled.at(n as u64);
        let got = out.channels[0][n];
        assert_eq!(
            got, want,
            "frame {n}: rendered {got}, the envelope is {want} — sample by sample means exactly"
        );
    }
    // And it really is the fade that was asked for: 60 dB over the second, at a constant rate.
    // The last *rendered* frame is 47 999, one short of the second, so the check is against the
    // curve's own endpoint rather than against a frame that is not there.
    assert_eq!(compiled.at(frames as u64), silent);
    let half = 20.0 * out.channels[0][frames / 2].log10();
    assert!(
        (half + 30.0).abs() < 0.01,
        "half way through a 60 dB fade should be -30 dB, got {half:.2}"
    );
}

/// Nothing automated must be nothing changed: an automated render of a graph with no lanes, and of
/// a graph whose lanes touch other nodes, is bit-identical to an ordinary render.
#[test]
fn an_unautomated_op_renders_exactly_as_before() {
    let input = tone(440.0, 8_000);
    let graph = Graph::named("a", Graph::op(OpKind::Gain, [0.5]))
        | Graph::named("b", Graph::op(OpKind::Lowpass, [2_000.0, 4.0]))
        | Graph::op(OpKind::Highpass, [100.0, 2.0]);

    let plain = process(&graph, &input);

    // No lanes at all.
    let empty = process_automated(&graph, &input, &Automation::new()).unwrap();
    assert_eq!(
        plain.channels, empty.channels,
        "an empty lane set changed the audio"
    );

    // A lane on 'a' only: 'b' and the unnamed high-pass must be untouched. Drive 'a' with a
    // constant equal to its own value, so the *whole* render should still be identical.
    let held = Automation::new().with(Lane::new("a", "gain", Curve::constant(0.5)));
    let out = process_automated(&graph, &input, &held).unwrap();
    let worst = out.channels[0]
        .iter()
        .zip(&plain.channels[0])
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst < 1e-6,
        "a constant lane changed the render by {worst}"
    );
}

/// A filter's cutoff is an input to a design, so it moves in 64-frame steps. What matters is that
/// it *moves*, that it ends up where the curve says, and that the staircase is small — all three
/// measured rather than asserted.
#[test]
fn a_swept_filter_follows_its_curve() {
    let frames = FS as usize;
    let graph = Graph::named("lp", Graph::op(OpKind::Lowpass, [8_000.0, 4.0]));

    // Sweep the cutoff from 8 kHz down to 200 Hz over a second — geometrically, because a
    // frequency sweep is heard in octaves. 8 kHz rather than something nearer Nyquist: a
    // Butterworth designed at 20 kHz against a 24 kHz Nyquist is badly warped, and this test is
    // about automation, not about filter design at the edge of the band.
    let curve = Curve::db_ramp(8_000.0, 200.0, 1.0);
    let automation = Automation::new().with(Lane::new("lp", "cutoff", curve));

    // A 1 kHz tone: three octaves inside the passband at the start, and far outside it by the end.
    let input = tone(1_000.0, frames);
    let out = process_automated(&graph, &input, &automation).unwrap();

    let peak = |x: &[f32]| x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let start = peak(&out.channels[0][1_000..3_000]);
    let end = peak(&out.channels[0][frames - 3_000..frames - 1_000]);
    assert!(
        (start - 0.5).abs() < 0.02,
        "at the start the tone is in the passband, got {start}"
    );
    assert!(
        end < 0.005,
        "by the end the cutoff is 200 Hz and a 1 kHz tone should be gone, got {end}"
    );

    // The staircase: a redesign every 64 frames must not show up as a *step*. Comparing the
    // largest sample-to-sample jump against the input's own is the wrong measurement — a
    // time-varying filter legitimately moves a little faster than the signal it is filtering, and
    // it measures 5% higher here with nothing wrong. What distinguishes a click is that it is an
    // **outlier**: one jump far larger than the rest, where a smooth signal's largest jump is
    // barely above its typical one.
    let mut steps: Vec<f32> = out.channels[0]
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .collect();
    steps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let worst = steps[steps.len() - 1];
    let typical = steps[steps.len() * 999 / 1000];
    assert!(
        worst <= typical * 1.5,
        "the largest step {worst} stands out against the 99.9th percentile {typical} — \
         that is a coefficient change showing as a discontinuity, not a filter sweeping"
    );
}

/// A curve read at absolute frames means a region rendered from the middle sees what a whole
/// render saw there. This is the property D4 builds on, checked at the automation level.
#[test]
fn a_render_from_the_middle_sees_the_same_curve() {
    let graph = Graph::named("g", Graph::op(OpKind::Gain, [1.0]));
    let automation = Automation::new().with(Lane::new(
        "g",
        "gain",
        Curve::new([Point::new(0.0, 1.0), Point::new(1.0, 0.0)], Timing::Once),
    ));

    let whole = process_automated(&graph, &dc(FS as usize), &automation).unwrap();
    let half = FS as usize / 2;
    let tail = process_automated_from(&graph, &dc(half), &automation, half as u64).unwrap();

    assert_eq!(
        &whole.channels[0][half..],
        &tail.channels[0][..],
        "the second half rendered on its own differs from the same span rendered whole"
    );
}

/// An LFO on a parameter is exactly a curve on a parameter — S4's sources reach D2's lanes with no
/// adapter in between.
#[test]
fn an_lfo_drives_a_parameter() {
    let graph = Graph::named("trem", Graph::op(OpKind::Gain, [1.0]));
    let automation =
        Automation::new().with(Lane::new("trem", "gain", Curve::lfo(4.0, 0.25, 1.0, 0.0)));
    let out = process_automated(&graph, &dc(FS as usize), &automation).unwrap();

    let env = &out.channels[0];
    assert!(
        (env[0] - 0.25).abs() < 1e-3,
        "the LFO starts low: {}",
        env[0]
    );
    // 4 Hz: a peak an eighth of a second in, back down a quarter of a second in.
    assert!((env[FS as usize / 8] - 1.0).abs() < 1e-3, "peak");
    assert!((env[FS as usize / 4] - 0.25).abs() < 1e-3, "trough");
}

/// A mistyped lane is an error before a sample is rendered, not a render that quietly ignored it.
#[test]
fn a_lane_that_does_not_fit_is_refused_by_name() {
    let graph = Graph::named("g", Graph::op(OpKind::Gain, [1.0]));
    let input = dc(100);

    let missing = Automation::new().with(Lane::new("nope", "gain", Curve::constant(1.0)));
    assert_eq!(
        process_automated(&graph, &input, &missing).unwrap_err(),
        AutomationError::UnknownNode {
            node: "nope".to_string()
        }
    );

    let wrong_param = Automation::new().with(Lane::new("g", "cutoff", Curve::constant(1.0)));
    let err = process_automated(&graph, &input, &wrong_param).unwrap_err();
    assert!(
        matches!(err, AutomationError::UnknownParam { ref param, .. } if param == "cutoff"),
        "{err}"
    );
    // The message names what it does have, so the fix is in the error.
    assert!(err.to_string().contains("gain"), "{err}");

    // An op with no streaming form says so rather than freezing the parameter silently.
    let graph = Graph::named("v", Graph::op(OpKind::Reverb, [0.5, 0.3, 0.3]));
    let bad = Automation::new().with(Lane::new("v", "room", Curve::constant(0.5)));
    let err = process_automated(&graph, &input, &bad).unwrap_err();
    assert!(
        matches!(err, AutomationError::NotAutomatable { op: "reverb", .. }),
        "{err}"
    );

    // A label around a whole subchain is ambiguous about which op it meant.
    let graph = Graph::named(
        "pair",
        Graph::op(OpKind::Gain, [1.0]) | Graph::op(OpKind::Gain, [1.0]),
    );
    let ambiguous = Automation::new().with(Lane::new("pair", "gain", Curve::constant(1.0)));
    assert_eq!(
        process_automated(&graph, &input, &ambiguous).unwrap_err(),
        AutomationError::NotAnOp {
            node: "pair".to_string()
        }
    );
}

/// A curve that wanders outside a parameter's static bounds pins to them rather than failing a
/// render half way through.
#[test]
fn a_curve_past_a_bound_pins_rather_than_failing() {
    let graph = Graph::named("lp", Graph::op(OpKind::Lowpass, [1_000.0, 4.0]));
    let automation = Automation::new().with(Lane::new(
        "lp",
        "order",
        // The registry bounds order at 1..16; ask for 40.
        Curve::new(
            [Point::shaped(0.0, 40.0, Shape::Step), Point::new(1.0, 40.0)],
            Timing::Once,
        ),
    ));
    let out = process_automated(&graph, &tone(500.0, 4_800), &automation)
        .expect("an out-of-range curve pins, it does not fail");
    assert!(out.channels[0].iter().all(|s| s.is_finite()));
}

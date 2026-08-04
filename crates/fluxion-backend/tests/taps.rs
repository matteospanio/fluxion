//! Observer taps (ROADMAP A1, A2, A3).
//!
//! The claim a tap has to earn is that it is invisible: a chain with analysers in it renders the
//! same audio as the chain without them, bit for bit. Everything else a tap does is only useful if
//! that holds.

use fluxion_backend::{process, process_taps};
use fluxion_core::{Graph, OpKind, Signal, TapData, TapKind};
use std::f32::consts::TAU;

const FS: u32 = 48_000;

fn tone(freq: f32, amp: f32, frames: usize) -> Vec<f32> {
    (0..frames)
        .map(|i| (TAU * freq * i as f32 / FS as f32).sin() * amp)
        .collect()
}

fn spectrum(size: usize) -> Graph {
    Graph::Tap(TapKind::Spectrum { size, overlap: 0.5 })
}

/// A1's own check, and the reason a tap is a different kind of node rather than an op: **bit**
/// identical, not nearly. Six taps of both kinds, scattered through a chain with a filter, a
/// nonlinearity and a parallel split.
#[test]
fn taps_do_not_touch_the_audio() {
    let input = Signal::new(FS, vec![tone(440.0, 0.4, 24_000), tone(660.0, 0.3, 24_000)]);

    let bare = Graph::op(OpKind::Lowpass, [2000.0, 4.0])
        | (Graph::op(OpKind::Gain, [0.5]) + Graph::op(OpKind::Highpass, [300.0, 2.0]))
        | Graph::op(OpKind::Compand, [0.01, 0.1, -20.0, 4.0, 6.0, 0.0]);

    let tapped = Graph::Tap(TapKind::Meter)
        | spectrum(1024)
        | Graph::op(OpKind::Lowpass, [2000.0, 4.0])
        | Graph::Tap(TapKind::Meter)
        | ((Graph::op(OpKind::Gain, [0.5]) | spectrum(512))
            + (Graph::op(OpKind::Highpass, [300.0, 2.0]) | Graph::Tap(TapKind::Meter)))
        | Graph::op(OpKind::Compand, [0.01, 0.1, -20.0, 4.0, 6.0, 0.0])
        | spectrum(2048);

    let plain = process(&bare, &input);
    let (observed, readings) = process_taps(&tapped, &input);

    assert_eq!(plain.channels, observed.channels, "taps changed the audio");
    assert_eq!(readings.len(), 6, "not every tap reported");
}

/// The same graph run through `process`, which has nowhere to publish: the taps cost nothing and
/// still change nothing.
#[test]
fn a_tap_is_free_when_nobody_is_listening() {
    let input = Signal::new(FS, vec![tone(440.0, 0.4, 4_800)]);
    let bare = Graph::op(OpKind::Gain, [0.5]);
    let tapped = spectrum(1024) | Graph::op(OpKind::Gain, [0.5]) | Graph::Tap(TapKind::Meter);
    assert_eq!(
        process(&bare, &input).channels,
        process(&tapped, &input).channels
    );
}

/// Readings come back in the order the chain reaches them, and the enclosing label names them.
#[test]
fn readings_are_ordered_and_labelled() {
    let input = Signal::new(FS, vec![tone(440.0, 0.5, 8_192)]);
    let graph = Graph::named("input", Graph::Tap(TapKind::Meter))
        | Graph::op(OpKind::Gain, [0.5])
        | Graph::named("output", Graph::Tap(TapKind::Meter))
        | Graph::Tap(TapKind::Meter);

    let (_, readings) = process_taps(&graph, &input);
    assert_eq!(readings.len(), 3);
    assert_eq!(readings[0].label.as_deref(), Some("input"));
    assert_eq!(readings[1].label.as_deref(), Some("output"));
    assert_eq!(readings[2].label, None, "an unlabelled tap has no label");

    // And they saw different things, in the right order: the gain halved the signal between them.
    let level = |i: usize| match readings[i].data {
        TapData::Meter { peak_db, .. } => peak_db,
        _ => panic!("expected a meter"),
    };
    assert!(
        (level(0) - level(1) - 6.02).abs() < 0.1,
        "the tap after gain(0.5) should read 6 dB lower: {} then {}",
        level(0),
        level(1)
    );
}

/// A2 at the chain level: the spectrum a tap reports is of the audio **at that point**, so a tap
/// before and after a low-pass disagree in exactly the way the filter does.
#[test]
fn a_spectrum_tap_sees_the_signal_where_it_sits() {
    // Two tones: one that survives a 1 kHz low-pass and one that does not. Both sit on exact bin
    // centres of a 2048-point FFT at 48 kHz (23.4375 Hz per bin, so bins 16 and 384) — a tone
    // between two bins loses up to 1.4 dB to the window's scalloping, and this test is about the
    // tap seeing the right signal, not about that.
    let mixed: Vec<f32> = tone(375.0, 0.4, 48_000)
        .iter()
        .zip(tone(9_000.0, 0.4, 48_000))
        .map(|(a, b)| a + b)
        .collect();
    let input = Signal::new(FS, vec![mixed]);

    let graph = Graph::named("before", spectrum(2048))
        | Graph::op(OpKind::Lowpass, [1000.0, 8.0])
        | Graph::named("after", spectrum(2048));
    let (_, readings) = process_taps(&graph, &input);

    let at = |i: usize, hz: f32| match &readings[i].data {
        TapData::Spectrum { bin_hz, magnitude } => magnitude[(hz / bin_hz).round() as usize],
        _ => panic!("expected a spectrum"),
    };

    // 375 Hz is in the passband and survives; 9 kHz is over three octaves up and does not.
    assert!(
        (at(0, 375.0) - 0.4).abs() < 0.02,
        "before: {}",
        at(0, 375.0)
    );
    assert!(
        (at(0, 9000.0) - 0.4).abs() < 0.02,
        "before: {}",
        at(0, 9000.0)
    );
    assert!((at(1, 375.0) - 0.4).abs() < 0.02, "after: {}", at(1, 375.0));
    assert!(
        at(1, 9000.0) < 1e-3,
        "an 8th-order low-pass at 1 kHz should have removed 9 kHz; {} left",
        at(1, 9000.0)
    );
}

/// A3's check: the meter tap's short-term loudness is the offline meter's, within 0.1 LU. It has
/// to be — the tap calls the same BS.1770 code — and this is what says so out loud, on material
/// with enough dynamics that a wrong window or a wrong gate would show.
#[test]
fn the_meter_tap_agrees_with_the_offline_meter() {
    // Four seconds: two loud, one quiet, one loud again — long enough for several 3 s windows and
    // uneven enough that the loudest one is not the average.
    let mut samples = tone(1_000.0, 0.5, FS as usize * 2);
    samples.extend(tone(1_000.0, 0.05, FS as usize));
    samples.extend(tone(1_000.0, 0.5, FS as usize));
    let input = Signal::new(FS, vec![samples.clone()]);

    let (_, readings) = process_taps(&Graph::Tap(TapKind::Meter), &input);
    let TapData::Meter {
        peak_db,
        rms_db,
        short_term_lufs,
    } = readings[0].data
    else {
        panic!("expected a meter");
    };

    let channels = vec![samples];
    let want_peak = fluxion_ops::loudness::sample_peak(&channels);
    let want_short = fluxion_ops::loudness::short_term_loudness(&channels, FS)
        .into_iter()
        .fold(f32::NEG_INFINITY, f32::max);

    assert!(
        (peak_db - want_peak).abs() < 1e-4,
        "peak {peak_db} vs {want_peak}"
    );
    assert!(
        (short_term_lufs - want_short).abs() < 0.1,
        "short-term {short_term_lufs} LUFS vs the offline meter's {want_short}"
    );
    // The quiet second drags the whole-signal RMS down: it has to land between the loud sections'
    // own RMS (-9.03 dBFS for a 0.5 sine) and the quiet one's (-29.0), rather than at either.
    // Deliberately not compared against `short_term_lufs` — dBFS and LUFS are different scales,
    // and putting them in one inequality is the mistake `TapData::Meter` warns about.
    assert!(
        (-29.0..-9.03).contains(&rms_db),
        "rms {rms_db} dBFS should sit between the quiet and loud sections"
    );
}

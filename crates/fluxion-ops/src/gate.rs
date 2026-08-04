//! Noise gate, with or without a key (ROADMAP S3).
//!
//! Below the threshold the signal is turned down by `range` decibels. Not to silence: a gate that
//! slams shut is more audible than the hiss it removed, and `range` is what lets a user pick how
//! much of the room to keep.
//!
//! Three times, because a gate that only had attack and release would chatter. `attack` is how fast
//! it opens, `hold` is how long it stays open after the level drops back under the threshold — the
//! part that stops it flickering on every dip inside a word — and `release` is how slowly it closes
//! after that.
//!
//! # The key
//!
//! [`gate_keyed`] listens to one signal and acts on another. That is the whole reason side inputs
//! exist ([`Graph::Side`](fluxion_core::Graph::Side)): a drum overhead opened by the snare's own
//! microphone hears the snare, not the bleed. With no key the gate listens to itself, which is the
//! ordinary case and is exactly [`gate`].

use crate::follower::{Detector, Follower};

/// How fast the level detector forgets a peak, in seconds.
///
/// This is not the user's `release` — that smooths the *gain*. This is what stops the detector
/// seeing a zero crossing as silence: a peak decays over a millisecond, which spans a whole cycle
/// of anything above about 1 kHz. Below that a cycle is longer than a millisecond and the level
/// really does dip within it, which is what `hold` is for and why its default is not zero.
const DETECTOR_RELEASE_S: f32 = 0.001;

/// Gate `x`, keyed by itself.
pub fn gate(
    x: &[f32],
    threshold_db: f32,
    range_db: f32,
    attack_s: f32,
    hold_s: f32,
    release_s: f32,
    fs: u32,
) -> Vec<f32> {
    gate_keyed(
        x,
        x,
        threshold_db,
        range_db,
        attack_s,
        hold_s,
        release_s,
        fs,
    )
}

/// Gate `x`, keyed by `key`: the level is measured on `key`, the gain is applied to `x`.
///
/// `key` shorter than `x` is read as silence past its end — a key that stops is a gate that closes,
/// which is the only reading that does not invent signal.
#[allow(clippy::too_many_arguments)]
pub fn gate_keyed(
    x: &[f32],
    key: &[f32],
    threshold_db: f32,
    range_db: f32,
    attack_s: f32,
    hold_s: f32,
    release_s: f32,
    fs: u32,
) -> Vec<f32> {
    let threshold = 10f32.powf(threshold_db / 20.0);
    // The floor the gate closes to. `range` of 0 is a gate that does nothing, which is a legitimate
    // setting and not worth special-casing.
    let floor = 10f32.powf(-range_db.max(0.0) / 20.0);
    let hold_samples = (hold_s.max(0.0) * fs as f32).round() as u64;

    // The detector is peak with an instantaneous attack: a gate has to open on the transient that
    // starts the note, and an RMS detector would still be thinking about it.
    let mut detector = Follower::new(0.0, DETECTOR_RELEASE_S, Detector::Peak, fs);
    // The gain itself is smoothed rather than switched, which is what attack and release *are*.
    let attack = crate::follower::smoothing_coef(attack_s, fs);
    let release = crate::follower::smoothing_coef(release_s, fs);

    let mut gain = floor;
    let mut held = 0u64;

    x.iter()
        .enumerate()
        .map(|(i, &sample)| {
            let level = detector.step(key.get(i).copied().unwrap_or(0.0));

            // Open while the key is loud enough, and for `hold` samples after it stops being.
            let open = if level >= threshold {
                held = hold_samples;
                true
            } else if held > 0 {
                held -= 1;
                true
            } else {
                false
            };

            let (target, coef) = if open {
                (1.0, attack)
            } else {
                (floor, release)
            };
            gain = coef * gain + (1.0 - coef) * target;
            sample * gain
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    const FS: u32 = 48_000;

    fn tone(freq: f32, amp: f32, frames: usize) -> Vec<f32> {
        (0..frames)
            .map(|i| (TAU * freq * i as f32 / FS as f32).sin() * amp)
            .collect()
    }

    fn peak(x: &[f32]) -> f32 {
        x.iter().fold(0.0f32, |m, v| m.max(v.abs()))
    }

    /// S3's first check: below the threshold the signal drops by exactly `range`, not by however
    /// much the implementation felt like.
    #[test]
    fn a_quiet_signal_drops_by_exactly_the_range() {
        // -60 dBFS, well under a -40 dB threshold.
        let quiet = tone(1000.0, 0.001, FS as usize);
        for range_db in [6.0f32, 20.0, 60.0] {
            let out = gate(&quiet, -40.0, range_db, 0.001, 0.0, 0.010, FS);
            // Skip the release ramp at the start, where the gain is still on its way down.
            let settled = &out[FS as usize / 2..];
            let reduction = 20.0 * (peak(&quiet) / peak(settled)).log10();
            assert!(
                (reduction - range_db).abs() < 0.1,
                "range {range_db} dB asked for, {reduction:.2} dB delivered"
            );
        }
    }

    /// And the other side of it: a signal above the threshold comes through untouched.
    #[test]
    fn a_loud_signal_passes_at_unity() {
        let loud = tone(1000.0, 0.5, FS as usize);
        let out = gate(&loud, -40.0, 60.0, 0.001, 0.010, 0.100, FS);
        let settled = &out[FS as usize / 4..];
        let want = &loud[FS as usize / 4..];
        let worst = settled
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-3, "an open gate changed the signal by {worst}");
    }

    /// S3's second check, and the reason S1 exists: with a key, the gate follows the key and not
    /// the programme. The programme here is *loud* and the key is *silent*, so a gate listening to
    /// the wrong one would leave it wide open.
    #[test]
    fn with_a_key_the_gate_follows_the_key_not_the_programme() {
        let loud = tone(1000.0, 0.5, FS as usize);
        let silent = vec![0.0f32; FS as usize];

        let keyed_shut = gate_keyed(&loud, &silent, -40.0, 60.0, 0.001, 0.0, 0.010, FS);
        let settled = &keyed_shut[FS as usize / 2..];
        let reduction = 20.0 * (peak(&loud) / peak(settled)).log10();
        assert!(
            (reduction - 60.0).abs() < 0.1,
            "a silent key should close the gate; it reduced by {reduction:.2} dB"
        );

        // The mirror image: a quiet programme held open by a loud key.
        let quiet = tone(1000.0, 0.001, FS as usize);
        let keyed_open = gate_keyed(&quiet, &loud, -40.0, 60.0, 0.001, 0.010, 0.100, FS);
        let settled = &keyed_open[FS as usize / 4..];
        let worst = settled
            .iter()
            .zip(&quiet[FS as usize / 4..])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst < 1e-5,
            "a loud key should hold the gate open; it changed the signal by {worst}"
        );
    }

    /// Hold is what stops a gate chattering: a key that dips below the threshold for less than the
    /// hold time must not close it. This is a property of the gate, not of the follower.
    #[test]
    fn hold_rides_over_a_short_dip() {
        // Loud, then 30 ms of silence, then loud again. 30 ms is chosen against the numbers rather
        // than by feel: the detector takes ~4 ms to fall from 0.5 to the -40 dB threshold, so a dip
        // has to outlast that before the gate can even begin to close.
        let dip_start = FS as usize / 20; // 50 ms in
        let dip = FS as usize * 30 / 1000; // 30 ms
        let mut key = tone(1000.0, 0.5, FS as usize / 5);
        for s in key.iter_mut().skip(dip_start).take(dip) {
            *s = 0.0;
        }
        let programme = vec![0.5f32; key.len()];

        // 50 ms of hold outlasts the dip; no hold at all does not.
        let held = gate_keyed(&programme, &key, -40.0, 60.0, 0.001, 0.050, 0.001, FS);
        let unheld = gate_keyed(&programme, &key, -40.0, 60.0, 0.001, 0.0, 0.001, FS);

        let during = dip_start + dip - 1;
        assert!(
            held[during] > 0.45,
            "hold should have ridden over the dip; gain fell to {}",
            held[during] / 0.5
        );
        assert!(
            unheld[during] < 0.05,
            "with no hold the gate should have closed; gain was {}",
            unheld[during] / 0.5
        );
    }

    /// A key that runs out is a key that went silent, not a key that repeats or holds.
    #[test]
    fn a_short_key_closes_the_gate() {
        let programme = tone(1000.0, 0.5, FS as usize);
        let key = tone(1000.0, 0.5, FS as usize / 10);
        let out = gate_keyed(&programme, &key, -40.0, 60.0, 0.001, 0.0, 0.010, FS);
        assert!(
            peak(&out[FS as usize / 2..]) < 0.001,
            "the gate stayed open past the end of the key"
        );
    }
}

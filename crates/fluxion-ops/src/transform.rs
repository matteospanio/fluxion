//! Geometry transforms on a whole [`Signal`] — the SoX "geometry" verbs (trim, pad, repeat, silence,
//! rate, speed, remix, channels) plus the multi-input primitives (concat, mix).
//!
//! These are **not** [`OpKind`](fluxion_core::OpKind)s: unlike the graph ops (which are per-channel,
//! length-preserving, and fs-preserving so they compose in the `|`/`+` algebra), every function here
//! deliberately changes the frame count, the channel count, or the sample rate. They are plain
//! functions over `fluxion_core::Signal`, applied before/after a graph rather than inside it.
//!
//! [`resample()`] is a real sample-rate converter (windowed-sinc, anti-aliased for downsampling) — the
//! SoX `rate` replacement; [`speed`] reuses it to change pitch+tempo together (SoX `speed`);
//! [`ensure_fs`] is the one a host calls on the way in, to pin everything to the project rate.
//!
//! All three run [`crate::resample::Resampler`], the same converter the streaming and realtime
//! paths use — there is one sample-rate conversion in fluxion, reached from different directions.

use fluxion_core::Signal;

use crate::resample;

/// Keep the window `[start_s, start_s + len_s)` seconds of every channel (clamped to the signal),
/// dropping the rest. Sample rate unchanged.
pub fn trim(sig: &Signal, start_s: f32, len_s: f32) -> Signal {
    let start = (start_s.max(0.0) * sig.fs as f32).round() as usize;
    let len = (len_s.max(0.0) * sig.fs as f32).round() as usize;
    let channels = sig
        .channels
        .iter()
        .map(|c| {
            let s = start.min(c.len());
            let e = (s + len).min(c.len());
            c[s..e].to_vec()
        })
        .collect();
    Signal::new(sig.fs, channels)
}

/// Prepend `start_s` and append `end_s` seconds of silence to every channel. Sample rate unchanged.
pub fn pad(sig: &Signal, start_s: f32, end_s: f32) -> Signal {
    let pre = (start_s.max(0.0) * sig.fs as f32).round() as usize;
    let post = (end_s.max(0.0) * sig.fs as f32).round() as usize;
    let channels = sig
        .channels
        .iter()
        .map(|c| {
            let mut out = Vec::with_capacity(pre + c.len() + post);
            out.resize(pre, 0.0);
            out.extend_from_slice(c);
            out.resize(pre + c.len() + post, 0.0);
            out
        })
        .collect();
    Signal::new(sig.fs, channels)
}

/// Concatenate every channel with itself `count` times (`count = 0` yields empty channels). Sample
/// rate unchanged.
pub fn repeat(sig: &Signal, count: usize) -> Signal {
    let channels = sig
        .channels
        .iter()
        .map(|c| {
            let mut out = Vec::with_capacity(c.len() * count);
            for _ in 0..count {
                out.extend_from_slice(c);
            }
            out
        })
        .collect();
    Signal::new(sig.fs, channels)
}

/// Trim leading and/or trailing near-silence. A frame is "silent" when its peak across channels is
/// below `threshold_db` dBFS; `min_s` seconds of silence are retained as a guard band at each trimmed
/// edge. `leading`/`trailing` select which ends are trimmed. An all-silent signal becomes empty.
pub fn silence_trim(
    sig: &Signal,
    threshold_db: f32,
    min_s: f32,
    leading: bool,
    trailing: bool,
) -> Signal {
    let frames = sig.frames();
    let thr = 10f32.powf(threshold_db / 20.0);
    let peak = |f: usize| {
        sig.channels
            .iter()
            .fold(0.0f32, |m, c| m.max(c.get(f).copied().unwrap_or(0.0).abs()))
    };
    let first_loud = (0..frames).find(|&f| peak(f) >= thr);
    let Some(first_loud) = first_loud else {
        // Entirely silent -> drop everything (keep the channel count).
        return Signal::new(sig.fs, sig.channels.iter().map(|_| Vec::new()).collect());
    };
    let last_loud = (0..frames).rev().find(|&f| peak(f) >= thr).unwrap();
    let guard = (min_s.max(0.0) * sig.fs as f32).round() as usize;
    let start = if leading {
        first_loud.saturating_sub(guard)
    } else {
        0
    };
    let end = if trailing {
        (last_loud + 1 + guard).min(frames)
    } else {
        frames
    };
    let channels = sig
        .channels
        .iter()
        .map(|c| c[start.min(c.len())..end.min(c.len())].to_vec())
        .collect();
    Signal::new(sig.fs, channels)
}

/// Resample to `to_fs` Hz with a real windowed-sinc converter (the SoX `rate` replacement). Preserves
/// frequency content and DC; anti-aliases when downsampling. Frame count scales by `to_fs / fs`.
pub fn resample(sig: &Signal, to_fs: u32) -> Signal {
    if to_fs == sig.fs || sig.frames() == 0 {
        return Signal::new(to_fs, sig.channels.clone());
    }
    let channels = sig
        .channels
        .iter()
        .map(|c| resample::convert(c, sig.fs, to_fs, resample::Quality::Hq))
        .collect();
    Signal::new(to_fs, channels)
}

/// Pin a signal to a project rate, converting only if it is not already there (ROADMAP R2).
///
/// A host sets its rate once and puts every input through here: what comes back is at `rate`, with
/// exactly `round(frames · rate/fs)` frames. It takes the signal by value so that the common case —
/// an input already at the project rate — costs nothing at all, not even a copy. A `rate` of 0 is
/// not a rate; it means no project rate is set, and the signal passes through.
pub fn ensure_fs(sig: Signal, rate: u32) -> Signal {
    if rate == 0 || rate == sig.fs {
        return sig;
    }
    resample(&sig, rate)
}

/// Change playback speed by `factor` (pitch **and** tempo together, SoX `speed`): resample the data
/// by `1/factor` but keep `fs`, so `factor > 1` is faster and higher-pitched. Anti-aliased.
pub fn speed(sig: &Signal, factor: f32) -> Signal {
    let factor = factor.max(1e-6);
    let ratio = 1.0 / f64::from(factor);
    let channels = sig
        .channels
        .iter()
        .map(|c| resample::convert_ratio(c, ratio, resample::Quality::Hq))
        .collect();
    Signal::new(sig.fs, channels) // same fs — pitch changes with tempo
}

/// Change tempo by `factor` without changing pitch (ROADMAP R3): `factor > 1` is faster, so the
/// output is `frames / factor` long. The opposite trade from [`speed`], which moves both together.
///
/// Frame count is exact — `round(frames / factor)` — which is what a timeline needs and what the
/// reference stretchers do not provide.
pub fn stretch(sig: &Signal, factor: f32) -> Signal {
    let factor = factor.max(1e-6);
    let channels = sig
        .channels
        .iter()
        .map(|c| crate::stretch::time_stretch(c, sig.fs, 1.0 / factor))
        .collect();
    Signal::new(sig.fs, channels)
}

/// Build each output channel as a weighted sum of input channels: `spec[j]` is a list of
/// `(input_channel, weight)` pairs for output channel `j`. Out-of-range input indices are ignored.
/// Frame count and sample rate unchanged; channel count becomes `spec.len()`.
pub fn remix(sig: &Signal, spec: &[Vec<(usize, f32)>]) -> Signal {
    let frames = sig.frames();
    let channels = spec
        .iter()
        .map(|mixdown| {
            let mut out = vec![0.0f32; frames];
            for &(src, w) in mixdown {
                if let Some(c) = sig.channels.get(src) {
                    for (o, &x) in out.iter_mut().zip(c) {
                        *o += w * x;
                    }
                }
            }
            out
        })
        .collect();
    Signal::new(sig.fs, channels)
}

/// Up/down-mix to `n` channels, energy-preserving. Uses a mono bridge (every input contributes with
/// weight `1/√(C·n)`), so for uncorrelated equal-power channels the total energy is preserved. A
/// no-op when `n` already equals the channel count.
pub fn channels(sig: &Signal, n: usize) -> Signal {
    let c = sig.channel_count();
    if n == c {
        return sig.clone();
    }
    if c == 0 || n == 0 {
        return Signal::new(sig.fs, (0..n).map(|_| Vec::new()).collect());
    }
    let w = 1.0 / ((c * n) as f32).sqrt();
    let spec: Vec<Vec<(usize, f32)>> = (0..n).map(|_| (0..c).map(|s| (s, w)).collect()).collect();
    remix(sig, &spec)
}

/// Which pair of gain curves a [`crossfade`] uses over the overlap.
///
/// The two laws are not interchangeable, and which one is right depends on what the two signals
/// have to do with each other:
///
/// | | curves | sums to 1 | right for |
/// |---|---|---|---|
/// | [`Linear`](CrossfadeLaw::Linear) | `1-t`, `t` | **amplitude** | material that is correlated — the same take, a loop point, a repeated bar |
/// | [`EqualPower`](CrossfadeLaw::EqualPower) | `cos(tπ/2)`, `sin(tπ/2)` | **power** | material that is unrelated — two different takes, a scene change |
///
/// Using one where the other belongs is audible in a specific way. Equal-power on identical
/// material sums to `cos + sin`, which peaks at `√2` — **+3.01 dB** in the middle of the fade.
/// Linear on unrelated material sums in power to `(1-t)² + t²`, which dips to 0.5 in the middle —
/// **-3.01 dB**, the classic hole in the middle of a crossfade.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CrossfadeLaw {
    /// `1-t` out, `t` in. The gains sum to 1 at every point, so identical material passes through
    /// completely unchanged.
    Linear,
    /// `cos(tπ/2)` out, `sin(tπ/2)` in. The gains *square* to 1 at every point, so unrelated
    /// material — whose powers add rather than whose amplitudes add — holds a steady level.
    EqualPower,
}

impl CrossfadeLaw {
    /// The `(fade-out, fade-in)` gains at normalized position `t ∈ [0, 1]`.
    ///
    /// `t = 0` is entirely the outgoing signal, `t = 1` entirely the incoming one.
    pub fn gains(self, t: f32) -> (f32, f32) {
        let t = t.clamp(0.0, 1.0);
        match self {
            CrossfadeLaw::Linear => (1.0 - t, t),
            CrossfadeLaw::EqualPower => {
                let theta = t * std::f32::consts::FRAC_PI_2;
                (theta.cos(), theta.sin())
            }
        }
    }
}

/// Concatenate signals with their ends overlapped and faded into each other (ROADMAP D1).
///
/// [`concat`] butt-joins, which puts a step at the seam unless both sides happen to be at zero.
/// This overlaps each adjacent pair by `overlap_s` seconds and crossfades across it, so the join is
/// continuous. Sample-accurate: the overlap is `round(overlap_s · fs)` frames, and the result is
/// exactly
///
/// ```text
/// sum(frames) - sum(overlap of each adjacent pair)
/// ```
///
/// frames long. The overlap of a pair is clamped to what the two sides actually have, so joining a
/// 10 ms signal with a 1 s overlap asked for overlaps by 10 ms rather than failing or inventing
/// audio — and the length formula still holds, using the clamped values.
///
/// Channel counts are unified to the maximum (missing channels are silent) and the first signal's
/// `fs` is used, both exactly as [`concat`] does. An empty slice yields an empty signal; a single
/// signal is returned unchanged, since there is no seam to fade.
///
/// See [`CrossfadeLaw`] for which law to pick — it is not a stylistic choice.
pub fn crossfade(sigs: &[&Signal], overlap_s: f32, law: CrossfadeLaw) -> Signal {
    let Some(first) = sigs.first() else {
        return Signal::new(0, Vec::new());
    };
    let nch = sigs.iter().map(|s| s.channel_count()).max().unwrap_or(0);
    let fs = first.fs;
    let want_overlap = (overlap_s.max(0.0) * fs as f32).round() as usize;

    // Fold left: the accumulator is everything joined so far, and each step overlaps its tail with
    // the next signal's head. Folding rather than joining all at once keeps the length formula a
    // sum over adjacent pairs, which is what makes it predictable when a signal is shorter than
    // the overlap.
    let mut acc: Vec<Vec<f32>> = (0..nch)
        .map(|ci| first.channels.get(ci).cloned().unwrap_or_default())
        .collect();
    let mut acc_frames = first.frames();

    for next in &sigs[1..] {
        let next_frames = next.frames();
        let overlap = want_overlap.min(acc_frames).min(next_frames);
        let joined = acc_frames + next_frames - overlap;

        for (ci, channel) in acc.iter_mut().enumerate() {
            channel.resize(acc_frames, 0.0); // a channel this signal did not have is silence
            let src = next.channels.get(ci);
            channel.resize(joined, 0.0);

            // The overlap sits at the end of what was there and the start of what is arriving.
            let start = acc_frames - overlap;
            for i in 0..overlap {
                // `overlap - 1` in the denominator so the ramp reaches exactly 1 on its last
                // sample, which makes the join continuous with the pure incoming signal that
                // follows. A 1-frame overlap has no span to ramp across, and the two signals meet
                // at that single sample — so it is the midpoint, where both contribute. Taking
                // `t = 0` there would silently drop the incoming signal's first frame.
                let t = if overlap > 1 {
                    i as f32 / (overlap - 1) as f32
                } else {
                    0.5
                };
                let (out_gain, in_gain) = law.gains(t);
                let incoming = src.and_then(|c| c.get(i)).copied().unwrap_or(0.0);
                channel[start + i] = channel[start + i] * out_gain + incoming * in_gain;
            }
            // Everything after the overlap is the incoming signal as it is.
            for i in overlap..next_frames {
                channel[start + i] = src.and_then(|c| c.get(i)).copied().unwrap_or(0.0);
            }
        }
        acc_frames = joined;
    }

    Signal::new(fs, acc)
}

/// Concatenate signals end-to-end (in time). Channel counts are unified to the maximum (missing
/// channels are silent). Uses the first signal's `fs`; an empty slice yields an empty signal.
pub fn concat(sigs: &[&Signal]) -> Signal {
    let Some(first) = sigs.first() else {
        return Signal::new(0, Vec::new());
    };
    let nch = sigs.iter().map(|s| s.channel_count()).max().unwrap_or(0);
    let channels = (0..nch)
        .map(|ci| {
            let mut out = Vec::new();
            for s in sigs {
                match s.channels.get(ci) {
                    Some(c) => out.extend_from_slice(c),
                    None => out.resize(out.len() + s.frames(), 0.0),
                }
            }
            out
        })
        .collect();
    Signal::new(first.fs, channels)
}

/// Sum signals sample-by-sample, zero-padding shorter ones to the longest. Channel counts are unified
/// to the maximum. Uses the first signal's `fs`; an empty slice yields an empty signal.
pub fn mix(sigs: &[&Signal]) -> Signal {
    let Some(first) = sigs.first() else {
        return Signal::new(0, Vec::new());
    };
    let nch = sigs.iter().map(|s| s.channel_count()).max().unwrap_or(0);
    let frames = sigs.iter().map(|s| s.frames()).max().unwrap_or(0);
    let channels = (0..nch)
        .map(|ci| {
            let mut out = vec![0.0f32; frames];
            for s in sigs {
                if let Some(c) = s.channels.get(ci) {
                    for (o, &x) in out.iter_mut().zip(c) {
                        *o += x;
                    }
                }
            }
            out
        })
        .collect();
    Signal::new(first.fs, channels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn mono(fs: u32, samples: Vec<f32>) -> Signal {
        Signal::new(fs, vec![samples])
    }

    #[test]
    fn trim_and_pad_are_inverse_lengths() {
        let s = mono(1_000, (0..1_000).map(|i| i as f32).collect());
        let t = trim(&s, 0.1, 0.2); // 100..300
        assert_eq!(t.frames(), 200);
        assert_eq!(t.channels[0][0], 100.0);
        let p = pad(&t, 0.05, 0.05); // +50 each side
        assert_eq!(p.frames(), 300);
        assert_eq!(p.channels[0][0], 0.0);
        assert_eq!(p.channels[0][50], 100.0);
    }

    #[test]
    fn repeat_multiplies_length() {
        let s = mono(48_000, vec![1.0, 2.0, 3.0]);
        let r = repeat(&s, 3);
        assert_eq!(
            r.channels[0],
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn silence_trim_strips_quiet_edges() {
        // 20 silent, 10 loud, 20 silent; threshold -40 dB, no guard band.
        let mut x = vec![0.0f32; 50];
        for v in x.iter_mut().take(30).skip(20) {
            *v = 0.5;
        }
        let s = mono(1_000, x);
        let out = silence_trim(&s, -40.0, 0.0, true, true);
        assert_eq!(out.frames(), 10);
        assert!(out.channels[0].iter().all(|&v| (v - 0.5).abs() < 1e-6));
    }

    #[test]
    fn resample_preserves_dc() {
        let s = mono(48_000, vec![0.5f32; 4_000]);
        let up = resample(&s, 44_100);
        // Interior stays at the DC level (edges taper as the kernel runs off the signal).
        let mid = &up.channels[0][200..up.frames() - 200];
        for &v in mid {
            assert!((v - 0.5).abs() < 1e-2, "DC drifted: {v}");
        }
        assert_eq!(up.fs, 44_100);
    }

    #[test]
    fn resample_roundtrip_keeps_a_sine() {
        // 1 kHz tone: 48k -> 44.1k -> 48k must return the same tone with small passband ripple.
        let fs = 48_000u32;
        let f0 = 1_000.0f32;
        let x: Vec<f32> = (0..4_800)
            .map(|i| (2.0 * PI * f0 * i as f32 / fs as f32).sin())
            .collect();
        let s = mono(fs, x.clone());
        let back = resample(&resample(&s, 44_100), 48_000);
        // Lengths line up within a sample of the original.
        assert!((back.frames() as i64 - x.len() as i64).abs() <= 2);
        // Compare the steady interior (skip the windowed-sinc edge transients).
        let y = &back.channels[0];
        let n = x.len().min(y.len());
        let mut worst = 0.0f32;
        for i in 200..n - 200 {
            worst = worst.max((y[i] - x[i]).abs());
        }
        assert!(worst < 5e-2, "roundtrip ripple too large: {worst}");
    }

    /// R2's check: whatever rate comes in, the project rate comes out, at the length the caller
    /// can compute from the two rates.
    #[test]
    fn ensure_fs_pins_every_input_rate_to_the_project_rate() {
        const PROJECT: u32 = 48_000;
        for from in [8_000, 22_050, 44_100, 48_000, 96_000, 192_000] {
            let frames = from as usize / 10; // 100 ms, whatever the rate
            let s = mono(from, vec![0.25f32; frames]);
            let out = ensure_fs(s, PROJECT);
            assert_eq!(out.fs, PROJECT, "from {from}");
            let want = (frames as f64 * f64::from(PROJECT) / f64::from(from)).round() as usize;
            assert_eq!(out.frames(), want, "from {from}");
        }
    }

    /// The common case, and the reason this takes the signal by value: an input already at the
    /// project rate is handed straight back, not run through a filter that would soften its top end
    /// for nothing.
    #[test]
    fn ensure_fs_leaves_a_matching_rate_alone() {
        let x: Vec<f32> = (0..1_000).map(|i| (0.05 * i as f32).sin()).collect();
        let out = ensure_fs(mono(48_000, x.clone()), 48_000);
        assert_eq!(out.channels[0], x);
        // 0 means "no project rate is set", so it is not a conversion request either.
        assert_eq!(ensure_fs(mono(44_100, x.clone()), 0).fs, 44_100);
    }

    #[test]
    fn speed_changes_tempo_keeps_fs() {
        let s = mono(
            48_000,
            (0..2_000).map(|i| (0.05 * i as f32).sin()).collect(),
        );
        let fast = speed(&s, 2.0);
        assert_eq!(fast.fs, 48_000); // pitch+tempo change, sample rate identical
        assert!((fast.frames() as i64 - 1_000).abs() <= 1); // half as many frames
    }

    #[test]
    fn remix_swaps_channels() {
        let s = Signal::new(48_000, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let swapped = remix(&s, &[vec![(1, 1.0)], vec![(0, 1.0)]]);
        assert_eq!(swapped.channels[0], vec![3.0, 4.0]);
        assert_eq!(swapped.channels[1], vec![1.0, 2.0]);
    }

    #[test]
    fn channels_upmix_is_energy_preserving() {
        // Mono -> stereo: each output = x/√2, so the summed energy across the two channels equals
        // the mono energy.
        let s = mono(48_000, vec![1.0, -1.0, 0.5]);
        let st = channels(&s, 2);
        assert_eq!(st.channel_count(), 2);
        for f in 0..3 {
            let e_out = st.channels[0][f].powi(2) + st.channels[1][f].powi(2);
            assert!((e_out - s.channels[0][f].powi(2)).abs() < 1e-6);
        }
    }

    // --- crossfade (ROADMAP D1) ---------------------------------------------------------------

    fn ramp(fs: u32, frames: usize) -> Signal {
        Signal::new(fs, vec![(0..frames).map(|i| i as f32 * 0.001).collect()])
    }

    /// The property that makes a law correct for *correlated* material: where the two sides of the
    /// seam carry the same value, the crossfade has to give that value back.
    ///
    /// Note what "a signal with itself" actually overlaps — the **tail** of the first copy against
    /// the **head** of the second, which are the same samples only if the signal is stationary.
    /// A constant is the clean case, so that is what this uses; a ramp would compare 2.4 against
    /// 0.0 and measure something else entirely.
    ///
    /// The law that holds here is **linear**, not equal-power, and the arithmetic is not subtle:
    /// linear's gains sum to `(1-t) + t = 1` everywhere. Equal-power's sum to `cos + sin`, which is
    /// `√2` at the midpoint — the +3.01 dB checked below. ROADMAP D1 names equal-power for this
    /// property; that is the wrong law for it, and this pair of tests is the evidence.
    #[test]
    fn a_linear_crossfade_of_a_signal_with_itself_is_unchanged() {
        let x = Signal::new(48_000, vec![vec![0.5f32; 4_800]]);
        let out = crossfade(&[&x, &x], 0.05, CrossfadeLaw::Linear);

        // Overlap 2400 frames: 4800 + 4800 - 2400.
        assert_eq!(out.frames(), 7_200);
        let worst = out.channels[0]
            .iter()
            .map(|s| (s - 0.5).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-6, "level moved by {worst}, the D1 bound is 1e-6");

        // And the same signal under equal-power does *not* hold its level — it is 3 dB up in the
        // middle of the seam. Stated here so the two tests cannot be reconciled by weakening one.
        let eq = crossfade(&[&x, &x], 0.05, CrossfadeLaw::EqualPower);
        let peak = eq.channels[0].iter().fold(0.0f32, |m, s| m.max(s.abs()));
        assert!(
            (20.0 * (peak / 0.5).log10() - 3.01).abs() < 0.01,
            "equal-power on identical material peaked at {peak}, expected 0.5·√2"
        );
    }

    /// And the mirror: on *uncorrelated* material — where power adds rather than amplitude —
    /// equal-power is the law that holds the level and linear is the one that digs a hole.
    ///
    /// Measured on white noise across the seam, comparing the RMS in the middle of the overlap
    /// against the RMS of the material going in.
    #[test]
    fn equal_power_holds_the_level_where_linear_digs_a_hole() {
        // Two independent noise signals, so the overlap really is uncorrelated.
        let mut state = 0x2545_f491u32;
        let mut noise = |n: usize| {
            let samples = (0..n)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (state >> 8) as f32 / 8_388_608.0 - 1.0
                })
                .collect();
            Signal::new(48_000, vec![samples])
        };
        let a = noise(48_000);
        let b = noise(48_000);

        let rms = |x: &[f32]| (x.iter().map(|s| s * s).sum::<f32>() / x.len() as f32).sqrt();
        let reference = rms(&a.channels[0]);

        // The middle of the overlap is where the two laws differ most.
        let overlap = 24_000; // 0.5 s
        let mid = 48_000 - overlap / 2;
        let window = 1_000;

        for (law, want_db, tolerance) in [
            (CrossfadeLaw::EqualPower, 0.0f32, 0.35f32),
            (CrossfadeLaw::Linear, -3.01, 0.35),
        ] {
            let out = crossfade(&[&a, &b], 0.5, law);
            let got = rms(&out.channels[0][mid - window / 2..mid + window / 2]);
            let db = 20.0 * (got / reference).log10();
            assert!(
                (db - want_db).abs() < tolerance,
                "{law:?} at the middle of the seam: {db:.2} dB, expected about {want_db}"
            );
        }
    }

    /// And the property that makes equal-power correct for *unrelated* material: the gains square
    /// to 1 everywhere, so power — which is what adds when two signals are uncorrelated — is
    /// constant across the fade. Exact to float noise, which is the ±1e-6 claim, correctly aimed.
    #[test]
    fn equal_power_gains_hold_the_power_constant() {
        for i in 0..=1_000 {
            let t = i as f32 / 1_000.0;
            let (out_gain, in_gain) = CrossfadeLaw::EqualPower.gains(t);
            let power = out_gain * out_gain + in_gain * in_gain;
            assert!(
                (power - 1.0).abs() < 1e-6,
                "t = {t}: power {power}, expected 1"
            );
        }
    }

    /// The other half of the same coin, stated so nobody swaps the laws back: equal-power on
    /// identical material is +3.01 dB in the middle, and linear on unrelated material is -3.01 dB.
    /// Both are the well-known audible failure, and both are what the *other* law is for.
    #[test]
    fn using_the_wrong_law_is_audible_by_exactly_three_decibels() {
        let (eq_out, eq_in) = CrossfadeLaw::EqualPower.gains(0.5);
        let amplitude_bump = 20.0 * (eq_out + eq_in).log10();
        assert!(
            (amplitude_bump - 3.01).abs() < 0.01,
            "equal-power on identical material: {amplitude_bump:.2} dB"
        );

        let (lin_out, lin_in) = CrossfadeLaw::Linear.gains(0.5);
        let power_dip = 10.0 * (lin_out * lin_out + lin_in * lin_in).log10();
        assert!(
            (power_dip + 3.01).abs() < 0.01,
            "linear on unrelated material: {power_dip:.2} dB"
        );
    }

    /// The length formula, including the case that would otherwise be a panic or a surprise: an
    /// overlap longer than one of the signals is clamped to what is there.
    #[test]
    fn the_length_is_the_sum_less_every_clamped_overlap() {
        let long = ramp(48_000, 4_800);
        let short = ramp(48_000, 100);

        // Three signals, two seams, 0.05 s = 2400 frames each: 4800*3 - 2400*2.
        let out = crossfade(&[&long, &long, &long], 0.05, CrossfadeLaw::EqualPower);
        assert_eq!(out.frames(), 4_800 * 3 - 2_400 * 2);

        // The short signal can only give 100 frames of overlap, so that seam is 100, not 2400.
        let out = crossfade(&[&long, &short], 0.05, CrossfadeLaw::EqualPower);
        assert_eq!(out.frames(), 4_800 + 100 - 100);

        // Degenerate inputs stay sane rather than panicking.
        assert_eq!(crossfade(&[], 0.05, CrossfadeLaw::Linear).frames(), 0);
        assert_eq!(
            crossfade(&[&long], 0.05, CrossfadeLaw::Linear).frames(),
            4_800
        );
        assert_eq!(
            crossfade(&[&long, &long], 0.0, CrossfadeLaw::Linear).frames(),
            9_600,
            "a zero overlap is exactly concat"
        );
    }

    /// The degenerate overlap: one frame, where the two signals meet at a single sample. Both
    /// have to be in it — an implementation that puts the ramp at t = 0 there keeps the outgoing
    /// signal and silently drops the incoming signal's first frame.
    #[test]
    fn a_one_frame_overlap_keeps_both_signals() {
        let a = Signal::new(48_000, vec![vec![1.0f32; 4]]);
        let b = Signal::new(48_000, vec![vec![3.0f32; 4]]);
        // 1/48000 s rounds to a 1-frame overlap.
        let out = crossfade(&[&a, &b], 1.0 / 48_000.0, CrossfadeLaw::Linear);

        assert_eq!(out.frames(), 7);
        // The shared frame is half of each, not all of one.
        assert!(
            (out.channels[0][3] - 2.0).abs() < 1e-6,
            "the meeting frame is {}, expected both signals at half",
            out.channels[0][3]
        );
        // And nothing was lost: three frames of `b` follow it at full level.
        assert_eq!(&out.channels[0][4..], &[3.0, 3.0, 3.0]);
    }

    /// A zero-length overlap must give back precisely what `concat` gives, or the two helpers
    /// disagree about the same operation.
    #[test]
    fn a_zero_overlap_is_concat() {
        let a = ramp(48_000, 500);
        let b = ramp(48_000, 300);
        assert_eq!(
            crossfade(&[&a, &b], 0.0, CrossfadeLaw::EqualPower).channels,
            concat(&[&a, &b]).channels
        );
    }

    /// Channels are unified like `concat` does, and a channel one side lacks fades against silence
    /// rather than being dropped or left at full level.
    #[test]
    fn a_missing_channel_fades_against_silence() {
        let stereo = Signal::new(48_000, vec![vec![1.0; 400], vec![1.0; 400]]);
        let mono = Signal::new(48_000, vec![vec![1.0; 400]]);
        let out = crossfade(&[&stereo, &mono], 0.005, CrossfadeLaw::Linear);

        assert_eq!(out.channels.len(), 2);
        // Channel 1 has nothing arriving, so across the overlap it is the outgoing fade alone.
        let overlap = 240;
        let start = 400 - overlap;
        let last = out.channels[1][start + overlap - 1];
        assert!(
            last.abs() < 1e-6,
            "the faded-out tail should reach 0, got {last}"
        );
    }

    #[test]
    fn concat_and_mix_combine_signals() {
        let a = mono(48_000, vec![1.0, 2.0]);
        let b = mono(48_000, vec![3.0, 4.0, 5.0]);
        assert_eq!(concat(&[&a, &b]).channels[0], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        // mix zero-pads the shorter to the longer.
        assert_eq!(mix(&[&a, &b]).channels[0], vec![4.0, 6.0, 5.0]);
    }
}

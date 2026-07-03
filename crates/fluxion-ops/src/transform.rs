//! Geometry transforms on a whole [`Signal`] — the SoX "geometry" verbs (trim, pad, repeat, silence,
//! rate, speed, remix, channels) plus the multi-input primitives (concat, mix).
//!
//! These are **not** [`OpKind`](fluxion_core::OpKind)s: unlike the graph ops (which are per-channel,
//! length-preserving, and fs-preserving so they compose in the `|`/`+` algebra), every function here
//! deliberately changes the frame count, the channel count, or the sample rate. They are plain
//! functions over `fluxion_core::Signal`, applied before/after a graph rather than inside it.
//!
//! [`resample`] is a real sample-rate converter (windowed-sinc, anti-aliased for downsampling) — the
//! SoX `rate` replacement; [`speed`] reuses it to change pitch+tempo together (SoX `speed`).

use std::f32::consts::PI;

use fluxion_core::Signal;

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

/// A symmetric Blackman window over `u ∈ [-1, 1]` (0 outside), for tapering the sinc.
fn blackman(u: f32) -> f32 {
    if u.abs() > 1.0 {
        0.0
    } else {
        0.42 + 0.5 * (PI * u).cos() + 0.08 * (2.0 * PI * u).cos()
    }
}

/// Normalized sinc `sin(πx)/(πx)`, `sinc(0) = 1`.
fn sinc(x: f32) -> f32 {
    if x.abs() < 1e-8 {
        1.0
    } else {
        let px = PI * x;
        px.sin() / px
    }
}

/// Zero-crossings of the windowed sinc on each side (taps/phase ≈ `2·ZEROS`).
const ZEROS: f32 = 32.0;

/// Resample one channel by `ratio = out_fs / in_fs` to `out_len` samples, with a windowed-sinc
/// (Blackman) kernel whose cutoff drops to `ratio` when downsampling (anti-aliasing). The kernel has
/// unit DC gain, so constants pass unchanged.
fn resample_channel(input: &[f32], ratio: f64, out_len: usize) -> Vec<f32> {
    let n = input.len();
    if n == 0 || out_len == 0 {
        return vec![0.0; out_len];
    }
    let cutoff = (ratio.min(1.0)) as f32; // normalized to the input Nyquist
    let half = ZEROS / cutoff.max(1e-6); // kernel half-width in input samples
    (0..out_len)
        .map(|m| {
            let center = m as f64 / ratio; // position in input-sample coordinates
            let i0 = (center - half as f64).ceil().max(0.0) as usize;
            let i1 = ((center + half as f64).floor() as i64).min(n as i64 - 1);
            let mut acc = 0.0f32;
            for k in i0 as i64..=i1 {
                let dx = center as f32 - k as f32;
                let w = cutoff * sinc(cutoff * dx) * blackman(dx / half);
                acc += input[k as usize] * w;
            }
            acc
        })
        .collect()
}

/// Resample to `to_fs` Hz with a real windowed-sinc converter (the SoX `rate` replacement). Preserves
/// frequency content and DC; anti-aliases when downsampling. Frame count scales by `to_fs / fs`.
pub fn resample(sig: &Signal, to_fs: u32) -> Signal {
    if to_fs == sig.fs || sig.frames() == 0 {
        return Signal::new(to_fs, sig.channels.clone());
    }
    let ratio = to_fs as f64 / sig.fs as f64;
    let out_len = (sig.frames() as f64 * ratio).round() as usize;
    let channels = sig
        .channels
        .iter()
        .map(|c| resample_channel(c, ratio, out_len))
        .collect();
    Signal::new(to_fs, channels)
}

/// Change playback speed by `factor` (pitch **and** tempo together, SoX `speed`): resample the data
/// by `1/factor` but keep `fs`, so `factor > 1` is faster and higher-pitched. Anti-aliased.
pub fn speed(sig: &Signal, factor: f32) -> Signal {
    let factor = factor.max(1e-6);
    let ratio = 1.0 / factor as f64;
    let out_len = (sig.frames() as f64 * ratio).round() as usize;
    let channels = sig
        .channels
        .iter()
        .map(|c| resample_channel(c, ratio, out_len))
        .collect();
    Signal::new(sig.fs, channels) // same fs — pitch changes with tempo
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

    #[test]
    fn concat_and_mix_combine_signals() {
        let a = mono(48_000, vec![1.0, 2.0]);
        let b = mono(48_000, vec![3.0, 4.0, 5.0]);
        assert_eq!(concat(&[&a, &b]).channels[0], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        // mix zero-pads the shorter to the longer.
        assert_eq!(mix(&[&a, &b]).channels[0], vec![4.0, 6.0, 5.0]);
    }
}

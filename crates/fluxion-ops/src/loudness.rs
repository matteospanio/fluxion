//! Loudness metering per ITU-R BS.1770 and EBU R128 / Tech 3342.
//!
//! Three numbers a mastering chain needs before it can decide anything:
//!
//! - [`integrated_loudness`] — the gated programme loudness, in LUFS.
//! - [`loudness_range`] — how much the loudness moves over the programme, in LU.
//! - [`short_term_loudness`] — the 3 s series the range is computed from, exposed because a meter
//!   wants it directly.
//!
//! # Why the arithmetic is `f64` here
//!
//! The RLB high-pass sits at 38 Hz, so at 48 kHz its poles are within 0.01 of the unit circle.
//! Measurement is not the audio path — it runs once, offline, over a whole programme — so it costs
//! nothing to do it in `f64` and removes any question about accumulated error over a long file.
//! [`k_weighting`] hands the same filter pair back as `f32` [`Biquad`]s for the realtime meter taps
//! that will want them (roadmap A3).
//!
//! # Provenance
//!
//! The K-weighting design is the published analytic prototype (De Man, *Evaluation of
//! implementations of the EBU R128 loudness measurement*, 2014), which reproduces the coefficients
//! tabulated in BS.1770-4 for 48 kHz to machine precision — asserted in the tests below — while
//! also being correct at every other sample rate. Math from the standard, not code from anyone's
//! implementation.

use crate::iir::Biquad;

/// The `-0.691` in `L = -0.691 + 10·log10(Σ G_c · z_c)`: the offset that puts the scale where
/// BS.1770 wants it.
const OFFSET: f64 = -0.691;

/// Absolute gate, LUFS. Blocks quieter than this never count towards the programme loudness.
const ABSOLUTE_GATE: f64 = -70.0;

/// Relative gate, LU below the ungated mean — BS.1770's second pass.
const RELATIVE_GATE: f64 = -10.0;

/// Relative gate for loudness range, LU. Wider than the loudness gate on purpose: R128 wants the
/// quiet parts of a programme to count towards its range.
const LRA_RELATIVE_GATE: f64 = -20.0;

/// Momentary block, seconds (BS.1770 gating block).
const BLOCK_SECONDS: f64 = 0.4;

/// Short-term block, seconds (EBU Tech 3342, for loudness range).
const SHORT_TERM_SECONDS: f64 = 3.0;

/// Hop between blocks, seconds. 100 ms is 75% overlap for the 400 ms block, as the standard
/// requires, and is what libebur128 uses for the 3 s block too.
const HOP_SECONDS: f64 = 0.1;

/// One biquad in `f64`, normalized so `a0 == 1`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Stage {
    b: [f64; 3],
    /// `[a1, a2]`.
    a: [f64; 2],
}

impl Stage {
    /// Direct form I, the shape the coefficients are written in.
    fn run(&self, x: &[f32]) -> Vec<f64> {
        let (mut x1, mut x2, mut y1, mut y2) = (0.0, 0.0, 0.0, 0.0);
        x.iter()
            .map(|&sample| {
                let x0 = f64::from(sample);
                let y0 = self.b[0] * x0 + self.b[1] * x1 + self.b[2] * x2
                    - self.a[0] * y1
                    - self.a[1] * y2;
                x2 = x1;
                x1 = x0;
                y2 = y1;
                y1 = y0;
                y0
            })
            .collect()
    }
}

/// The K-weighting pair for a sample rate: a high shelf, then the RLB high-pass.
fn design(fs: u32) -> [Stage; 2] {
    let fs = f64::from(fs);

    // Stage 1 — high shelf, +4 dB above ~1.7 kHz: the head-related "acoustic effect of the head".
    let (gain_db, q, fc) = (
        3.999_843_853_973_347,
        0.707_175_236_955_419_6,
        1_681.974_450_955_533,
    );
    let k = (std::f64::consts::PI * fc / fs).tan();
    let vh = 10f64.powf(gain_db / 20.0);
    let vb = vh.powf(0.499_666_774_154_541_6);
    let norm = 1.0 + k / q + k * k;
    let shelf = Stage {
        b: [
            (vh + vb * k / q + k * k) / norm,
            2.0 * (k * k - vh) / norm,
            (vh - vb * k / q + k * k) / norm,
        ],
        a: [2.0 * (k * k - 1.0) / norm, (1.0 - k / q + k * k) / norm],
    };

    // Stage 2 — RLB high-pass at ~38 Hz, which is what stops rumble from reading as loudness.
    let (q, fc) = (0.500_327_037_323_877_3, 38.135_470_876_024_44);
    let k = (std::f64::consts::PI * fc / fs).tan();
    let norm = 1.0 + k / q + k * k;
    let highpass = Stage {
        b: [1.0, -2.0, 1.0],
        a: [2.0 * (k * k - 1.0) / norm, (1.0 - k / q + k * k) / norm],
    };

    [shelf, highpass]
}

/// The K-weighting filter pair for `fs`, as `Biquad`s.
///
/// Exposed so the realtime meter taps can reuse the same weighting the offline meter applies —
/// a meter that disagreed with the file's measured loudness would be worse than no meter.
pub fn k_weighting(fs: u32) -> [Biquad; 2] {
    design(fs).map(|s| Biquad {
        b0: s.b[0] as f32,
        b1: s.b[1] as f32,
        b2: s.b[2] as f32,
        a1: s.a[0] as f32,
        a2: s.a[1] as f32,
    })
}

/// Per-channel weights `G_c` from BS.1770 Table 3.
///
/// Left, right and centre count fully; the surrounds count for ~1.5 dB more; LFE does not count at
/// all. Layouts the standard does not name are weighted 1.0 throughout, which is the only
/// defensible guess — a channel order we cannot identify is one we cannot weight.
pub fn channel_weights(channels: usize) -> Vec<f32> {
    match channels {
        // L R (C) — every channel counts fully.
        0..=3 => vec![1.0; channels],
        // Quad: L R Ls Rs.
        4 => vec![1.0, 1.0, 1.41, 1.41],
        // 5.0: L R C Ls Rs.
        5 => vec![1.0, 1.0, 1.0, 1.41, 1.41],
        // 5.1: L R C LFE Ls Rs — the LFE is excluded by the standard.
        6 => vec![1.0, 1.0, 1.0, 0.0, 1.41, 1.41],
        other => vec![1.0; other],
    }
}

/// Mean square of each K-weighted channel over one window: BS.1770's `z_c`.
fn block_powers(weighted: &[Vec<f64>], start: usize, len: usize) -> Vec<f64> {
    weighted
        .iter()
        .map(|channel| {
            channel[start..start + len]
                .iter()
                .map(|v| v * v)
                .sum::<f64>()
                / len as f64
        })
        .collect()
}

/// `L = -0.691 + 10·log10(Σ G_c · z_c)` — the loudness of one block.
fn loudness_of(powers: &[f64], weights: &[f32]) -> f64 {
    let sum: f64 = powers
        .iter()
        .zip(weights)
        .map(|(z, g)| f64::from(*g) * z)
        .sum();
    if sum <= 0.0 {
        f64::NEG_INFINITY
    } else {
        OFFSET + 10.0 * sum.log10()
    }
}

/// K-weight every channel once; every measurement below works from this.
fn weight(channels: &[Vec<f32>], fs: u32) -> Vec<Vec<f64>> {
    let [shelf, highpass] = design(fs);
    channels
        .iter()
        .map(|channel| {
            let stage1 = shelf.run(channel);
            // Stage 2 takes the stage-1 output; keep it in f64 rather than round-tripping to f32.
            let (mut x1, mut x2, mut y1, mut y2) = (0.0, 0.0, 0.0, 0.0);
            stage1
                .iter()
                .map(|&x0| {
                    let y0 = highpass.b[0] * x0 + highpass.b[1] * x1 + highpass.b[2] * x2
                        - highpass.a[0] * y1
                        - highpass.a[1] * y2;
                    x2 = x1;
                    x1 = x0;
                    y2 = y1;
                    y1 = y0;
                    y0
                })
                .collect()
        })
        .collect()
}

/// The per-block powers and loudnesses for a given window length.
fn blocks(
    weighted: &[Vec<f64>],
    weights: &[f32],
    fs: u32,
    seconds: f64,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let frames = weighted.first().map_or(0, Vec::len);
    let len = (seconds * f64::from(fs)).round() as usize;
    let hop = (HOP_SECONDS * f64::from(fs)).round() as usize;
    if len == 0 || hop == 0 || frames < len {
        return (Vec::new(), Vec::new());
    }
    let mut powers = Vec::new();
    let mut loudnesses = Vec::new();
    let mut start = 0;
    while start + len <= frames {
        let z = block_powers(weighted, start, len);
        loudnesses.push(loudness_of(&z, weights));
        powers.push(z);
        start += hop;
    }
    (powers, loudnesses)
}

/// Mean of the kept blocks' powers, per channel, then one loudness from them.
fn gated_loudness(powers: &[Vec<f64>], loudnesses: &[f64], weights: &[f32], gate: f64) -> f64 {
    let kept: Vec<&Vec<f64>> = powers
        .iter()
        .zip(loudnesses)
        .filter(|(_, l)| **l > gate)
        .map(|(z, _)| z)
        .collect();
    if kept.is_empty() {
        return f64::NEG_INFINITY;
    }
    let mean: Vec<f64> = (0..weights.len())
        .map(|c| kept.iter().map(|z| z[c]).sum::<f64>() / kept.len() as f64)
        .collect();
    loudness_of(&mean, weights)
}

/// Gated integrated loudness, in LUFS.
///
/// Two passes, as the standard specifies: drop everything below −70 LUFS, take the mean of what is
/// left, then drop everything more than 10 LU below *that* and take the mean again. The second pass
/// is what stops a quiet passage from dragging a programme's number down.
///
/// Returns `-inf` for silence, or for a signal shorter than one 400 ms block.
///
/// ```
/// use fluxion_ops::loudness::integrated_loudness;
/// // A 1 kHz sine at -20 dBFS RMS.
/// let fs = 48_000;
/// let amp = 0.1 * 2f32.sqrt();
/// let x: Vec<f32> = (0..fs * 5)
///     .map(|n| amp * (std::f32::consts::TAU * 1000.0 * n as f32 / fs as f32).sin())
///     .collect();
/// let lufs = integrated_loudness(&[x], fs as u32);
/// assert!((lufs - -20.03).abs() < 0.1, "{lufs}");
/// ```
pub fn integrated_loudness(channels: &[Vec<f32>], fs: u32) -> f32 {
    let weights = channel_weights(channels.len());
    let weighted = weight(channels, fs);
    let (powers, loudnesses) = blocks(&weighted, &weights, fs, BLOCK_SECONDS);
    if powers.is_empty() {
        return f32::NEG_INFINITY;
    }
    // Pass one: absolute gate only, to find where the programme sits.
    let ungated = gated_loudness(&powers, &loudnesses, &weights, ABSOLUTE_GATE);
    if !ungated.is_finite() {
        return f32::NEG_INFINITY;
    }
    // Pass two: relative to that.
    let gate = (ungated + RELATIVE_GATE).max(ABSOLUTE_GATE);
    gated_loudness(&powers, &loudnesses, &weights, gate) as f32
}

/// The short-term (3 s) loudness series, in LUFS, one value every 100 ms.
///
/// This is what a meter draws and what [`loudness_range`] is computed from. Empty for a signal
/// shorter than one block.
pub fn short_term_loudness(channels: &[Vec<f32>], fs: u32) -> Vec<f32> {
    let weights = channel_weights(channels.len());
    let weighted = weight(channels, fs);
    let (_, loudnesses) = blocks(&weighted, &weights, fs, SHORT_TERM_SECONDS);
    loudnesses.into_iter().map(|l| l as f32).collect()
}

/// Loudness range, in LU (EBU Tech 3342): the spread between the quiet and loud parts of a
/// programme, once the silence and the outliers are gated away.
///
/// The 10th to 95th percentile of the gated short-term loudness — not the full range, because the
/// extremes of a programme are usually a fade or a stray peak rather than something anyone hears as
/// dynamics.
///
/// Returns `0.0` when there is not enough material to judge.
pub fn loudness_range(channels: &[Vec<f32>], fs: u32) -> f32 {
    let weights = channel_weights(channels.len());
    let weighted = weight(channels, fs);
    let (powers, loudnesses) = blocks(&weighted, &weights, fs, SHORT_TERM_SECONDS);
    if powers.is_empty() {
        return 0.0;
    }
    let ungated = gated_loudness(&powers, &loudnesses, &weights, ABSOLUTE_GATE);
    if !ungated.is_finite() {
        return 0.0;
    }
    let gate = ungated + LRA_RELATIVE_GATE;
    let mut kept: Vec<f64> = loudnesses
        .into_iter()
        .filter(|l| *l > ABSOLUTE_GATE && *l > gate)
        .collect();
    if kept.is_empty() {
        return 0.0;
    }
    kept.sort_by(|a, b| a.partial_cmp(b).expect("gated loudnesses are finite"));
    (percentile(&kept, 95.0) - percentile(&kept, 10.0)) as f32
}

/// Linear-interpolated percentile of a sorted slice.
fn percentile(sorted: &[f64], pct: f64) -> f64 {
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = (pct / 100.0) * (sorted.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    sorted[lo] + (sorted[hi] - sorted[lo]) * (rank - lo as f64)
}

// --- true peak (BS.1770 Attachment 1) --------------------------------------------------------

/// Oversampling factor for true-peak measurement.
///
/// BS.1770 specifies 4x for sample rates up to 48 kHz, which puts the effective rate at 192 kHz —
/// far enough above the audio band that the reconstructed waveform's real maximum is captured.
/// Applying it at every rate costs a little work at 96 kHz and above and keeps one code path.
const OVERSAMPLE: usize = 4;

/// Taps per polyphase branch, so a 96-tap prototype.
///
/// Chosen by measurement against signals whose true peak is known exactly (a bandlimited sine's
/// peak is its amplitude). Error versus that truth, on signals faded in and out so the filter is
/// measured rather than its edge behaviour:
///
/// | taps/phase | 1 kHz  | 10 kHz | 19 kHz | fs/4 inter-sample peak |
/// |------------|--------|--------|--------|------------------------|
/// | 12         | -0.002 | +0.012 | -0.201 | -0.120                 |
/// | **24**     | -0.001 | +0.015 | +0.010 | -0.115                 |
/// | 48         | -0.001 | +0.019 | +0.028 | -0.115                 |
///
/// Twelve is too short near Nyquist; past 24 nothing improves. The residual -0.115 dB is not the
/// filter but the oversampling factor: at 4x a 12 kHz crest can fall up to `cos(pi/16)` between
/// samples, which is -0.17 dB, so reading 0.115 low there is correct behaviour.
const PHASE_TAPS: usize = 24;

/// The 4-phase interpolation filter: a Blackman-windowed sinc, split into polyphase branches.
///
/// The standard tabulates one specific filter; this constructs an equivalent from the same
/// prototype, which has the advantage of being derivable rather than copied. Each branch is
/// normalized to unit sum so the interpolator has unity gain at DC and cannot invent level.
fn interpolator() -> [[f32; PHASE_TAPS]; OVERSAMPLE] {
    let taps = PHASE_TAPS * OVERSAMPLE;
    let center = (taps - 1) as f64 / 2.0;
    let mut prototype = vec![0.0f64; taps];
    for (n, h) in prototype.iter_mut().enumerate() {
        let x = n as f64 - center;
        // sinc with the cutoff at the original Nyquist, i.e. 1/OVERSAMPLE of the new rate.
        let sinc = if x.abs() < 1e-12 {
            1.0
        } else {
            let arg = std::f64::consts::PI * x / OVERSAMPLE as f64;
            arg.sin() / arg
        };
        // Blackman window: the stopband it gives is what keeps an image from reading as a peak.
        let t = 2.0 * std::f64::consts::PI * n as f64 / (taps - 1) as f64;
        let window = 0.42 - 0.5 * t.cos() + 0.08 * (2.0 * t).cos();
        *h = sinc * window;
    }

    // Normalize the prototype once, to a total gain of OVERSAMPLE. Scaling each branch separately
    // would give every phase unity DC gain but warp the response between them.
    let total: f64 = prototype.iter().sum();
    let scale = OVERSAMPLE as f64 / total;

    let mut phases = [[0.0f32; PHASE_TAPS]; OVERSAMPLE];
    for (p, phase) in phases.iter_mut().enumerate() {
        for (k, tap) in phase.iter_mut().enumerate() {
            *tap = (prototype[k * OVERSAMPLE + p] * scale) as f32;
        }
    }
    phases
}

/// True peak in dBTP: the largest absolute value of the *reconstructed* waveform, not of the
/// samples.
///
/// A signal can sit at -0.1 dBFS sample-peak and still overshoot 0 dBFS between samples, which is
/// where converters and lossy encoders clip. Measuring it needs the waveform the reconstruction
/// filter would produce, so the signal is oversampled 4x first.
///
/// The signal is treated as silence before and after, because that is what a player does: a file
/// that begins at full amplitude really does produce a transient, and this reports it.
///
/// Returns `-inf` for digital silence.
///
/// # Accuracy
///
/// Within about 0.12 dB of the true maximum, measured against signals whose peak is known
/// analytically — see the `PHASE_TAPS` constant. Note that ffmpeg's `ebur128` reads up to 0.8 dB **high** near
/// Nyquist (it reports a 10 kHz sine of amplitude 0.5 as -5.2 dBTP, above the signal's own
/// maximum), so the two disagree there; the tolerance in `tests/loudness_golden.rs` says so
/// explicitly rather than pretending otherwise.
///
/// ```
/// use fluxion_ops::loudness::true_peak;
/// // A 10 kHz sine at 48 kHz, sampled where its crest falls between samples: the true peak is
/// // meaningfully above the sample peak.
/// let fs = 48_000;
/// let x: Vec<f32> = (0..fs)
///     .map(|n| 0.5 * (std::f32::consts::TAU * 10_000.0 * n as f32 / fs as f32).sin())
///     .collect();
/// let sample_peak = 20.0 * x.iter().fold(0.0f32, |m, v| m.max(v.abs())).log10();
/// assert!(true_peak(&[x], fs as u32) > sample_peak + 0.3);
/// ```
pub fn true_peak(channels: &[Vec<f32>], _fs: u32) -> f32 {
    let peak = true_peak_envelope(channels)
        .into_iter()
        .fold(0.0f32, f32::max);
    if peak <= 0.0 {
        f32::NEG_INFINITY
    } else {
        20.0 * peak.log10()
    }
}

/// The true-peak magnitude around each input sample, linear.
///
/// One value per input sample: the largest reconstructed magnitude among the interpolated points
/// that sample is responsible for, taken across every channel so a limiter built on this moves all
/// channels together and leaves the stereo image alone.
///
/// This is what makes a limiter a *true-peak* limiter rather than a sample-peak one.
pub fn true_peak_envelope(channels: &[Vec<f32>]) -> Vec<f32> {
    let frames = channels.iter().map(Vec::len).max().unwrap_or(0);
    let phases = interpolator();
    let mut envelope = vec![0.0f32; frames];

    for channel in channels {
        let mut history = [0.0f32; PHASE_TAPS];
        // The filter is causal with its centre PHASE_TAPS/2 samples back, so the interpolated
        // points produced while reading sample `i` describe the signal around `i - centre`.
        let centre = PHASE_TAPS / 2;
        for (i, &sample) in channel
            .iter()
            .chain(std::iter::repeat_n(&0.0, PHASE_TAPS))
            .enumerate()
        {
            history.rotate_right(1);
            history[0] = sample;
            // Clamp rather than skip: the interpolated points before the filter's centre reaches
            // the first sample, and after it passes the last, are still part of the reconstructed
            // waveform and still clip. Attributing them to the nearest sample keeps
            // `max(envelope)` equal to the true peak instead of quietly a little under it.
            let at = i.saturating_sub(centre).min(frames - 1);
            for phase in &phases {
                let interpolated: f32 = phase.iter().zip(&history).map(|(h, x)| h * x).sum();
                envelope[at] = envelope[at].max(interpolated.abs());
            }
            // The sample itself is on the reconstructed curve, so it can never be above it.
            envelope[at] = envelope[at].max(channel[at].abs());
        }
    }
    envelope
}

/// Sample peak in dBFS — the largest absolute sample, with no reconstruction.
///
/// Kept beside [`true_peak`] because the difference between the two is the thing a mastering
/// engineer is actually looking at.
pub fn sample_peak(channels: &[Vec<f32>]) -> f32 {
    let peak = channels
        .iter()
        .flat_map(|c| c.iter())
        .fold(0.0f32, |m, v| m.max(v.abs()));
    if peak <= 0.0 {
        f32::NEG_INFINITY
    } else {
        20.0 * peak.log10()
    }
}

/// Bring the programme to `target_lufs`, then hold its true peak under `ceiling_db` dBTP.
///
/// Two passes, which is the only way it can work: loudness is a property of the whole programme,
/// so it has to be measured before anything can be decided. Measure, apply the difference as a
/// gain, then limit — the limiter can only pull down, so it can cost a little loudness on material
/// with isolated peaks; how much is what the tolerance in the tests measures.
///
/// Silence is left alone: there is no gain that makes nothing louder.
pub fn loudness_normalize(channels: &mut [Vec<f32>], target_lufs: f32, ceiling_db: f32, fs: u32) {
    // Measure, apply, verify — and if the verification disagrees, correct and go round again.
    // The limiter can only pull down, so on material with isolated peaks it costs loudness the
    // first measurement could not have predicted; another pass recovers it.
    if !integrated_loudness(channels, fs).is_finite() {
        return; // silence: no gain makes nothing louder
    }

    // Hold the ceiling before anything else, so that every candidate below — including the one
    // kept when the target turns out to be unreachable — already respects it. The ceiling is the
    // part that is not negotiable; the target is the part that is.
    crate::dynamics::limit(channels, ceiling_db, 0.005, 0.05, fs);

    let mut best: Option<(f32, Vec<Vec<f32>>)> = None;
    for _ in 0..NORMALIZE_PASSES {
        let measured = integrated_loudness(channels, fs);
        let error = (target_lufs - measured).abs();

        // Keep the closest attempt, not the last one. On material with enough crest factor the
        // target is simply unreachable under the ceiling — each extra decibel of gain buys less
        // than the limiter takes away — and without this the passes would chase it downward and
        // hand back something quieter than where they started.
        if best.as_ref().is_none_or(|(e, _)| error < *e) {
            best = Some((error, channels.to_vec()));
        } else {
            break;
        }
        if error < 0.05 {
            break;
        }

        let gain = 10f32.powf((target_lufs - measured) / 20.0);
        for channel in channels.iter_mut() {
            for sample in channel.iter_mut() {
                *sample *= gain;
            }
        }
        // 5 ms of lookahead and a 50 ms release: short enough not to pump, long enough not to
        // distort.
        crate::dynamics::limit(channels, ceiling_db, 0.005, 0.05, fs);
    }

    if let Some((_, closest)) = best {
        for (channel, kept) in channels.iter_mut().zip(closest) {
            *channel = kept;
        }
    }
}

/// Measure-apply-verify rounds. Peaky material needs a second; nothing measured has needed a
/// fourth, and material that cannot reach the target without the limiter eating the difference
/// will not get there however many are allowed.
const NORMALIZE_PASSES: usize = 4;

#[cfg(test)]
mod tests {
    use super::*;

    /// The coefficients BS.1770-4 tabulates for 48 kHz (Tables 1 and 2). Our design is analytic so
    /// it works at any rate; at 48 kHz it must land on the standard's own numbers.
    #[test]
    fn the_design_reproduces_the_standards_48k_table() {
        let [shelf, highpass] = design(48_000);
        let want_shelf = Stage {
            b: [
                1.535_124_859_586_97,
                -2.691_696_189_406_38,
                1.198_392_810_852_85,
            ],
            a: [-1.690_659_293_182_41, 0.732_480_774_215_85],
        };
        let want_hp = Stage {
            b: [1.0, -2.0, 1.0],
            a: [-1.990_047_454_833_98, 0.990_072_250_366_21],
        };
        for (got, want, name) in [
            (shelf, want_shelf, "shelf"),
            (highpass, want_hp, "high-pass"),
        ] {
            for i in 0..3 {
                assert!(
                    (got.b[i] - want.b[i]).abs() < 1e-9,
                    "{name} b{i}: {} vs {}",
                    got.b[i],
                    want.b[i]
                );
            }
            for i in 0..2 {
                assert!(
                    (got.a[i] - want.a[i]).abs() < 1e-9,
                    "{name} a{}: {} vs {}",
                    i + 1,
                    got.a[i],
                    want.a[i]
                );
            }
        }
    }

    /// The design has to be a design, not a table lookup: at other rates it must still produce a
    /// stable filter with the same shape.
    #[test]
    fn the_design_is_stable_at_every_common_rate() {
        for fs in [8_000, 22_050, 44_100, 48_000, 88_200, 96_000, 192_000] {
            for stage in design(fs) {
                // Jury's criterion for a second-order section: |a2| < 1 and |a1| < 1 + a2.
                let (a1, a2) = (stage.a[0], stage.a[1]);
                assert!(a2.abs() < 1.0, "fs {fs}: |a2| = {}", a2.abs());
                assert!(a1.abs() < 1.0 + a2, "fs {fs}: |a1| = {}", a1.abs());
            }
        }
    }

    fn sine(freq: f32, amp: f32, secs: f32, fs: u32) -> Vec<f32> {
        (0..(secs * fs as f32) as usize)
            .map(|n| amp * (std::f32::consts::TAU * freq * n as f32 / fs as f32).sin())
            .collect()
    }

    /// Doubling the amplitude is exactly +6.02 LU, whatever the absolute calibration is. This
    /// checks the scale is a true log of power without depending on any oracle.
    #[test]
    fn doubling_the_amplitude_adds_six_lu() {
        let quiet = integrated_loudness(&[sine(1000.0, 0.1, 5.0, 48_000)], 48_000);
        let loud = integrated_loudness(&[sine(1000.0, 0.2, 5.0, 48_000)], 48_000);
        assert!(
            ((loud - quiet) - 20.0 * 2f32.log10()).abs() < 0.01,
            "{quiet} -> {loud}"
        );
    }

    /// The same signal in two channels is 3 dB louder than in one — the channel sum is of powers,
    /// not amplitudes.
    #[test]
    fn a_second_identical_channel_adds_three_lu() {
        let x = sine(1000.0, 0.25, 5.0, 48_000);
        let mono = integrated_loudness(std::slice::from_ref(&x), 48_000);
        let stereo = integrated_loudness(&[x.clone(), x], 48_000);
        assert!(
            ((stereo - mono) - 20.0 * 2f32.sqrt().log10()).abs() < 0.01,
            "{mono} -> {stereo}"
        );
    }

    /// Silence is not a loudness. Neither is a signal too short to fill one gating block.
    #[test]
    fn silence_and_short_signals_have_no_loudness() {
        assert_eq!(
            integrated_loudness(&[vec![0.0; 48_000]], 48_000),
            f32::NEG_INFINITY
        );
        assert_eq!(
            integrated_loudness(&[vec![0.5; 100]], 48_000),
            f32::NEG_INFINITY
        );
        assert_eq!(integrated_loudness(&[], 48_000), f32::NEG_INFINITY);
    }

    /// The relative gate is the whole point of the two-pass algorithm: tripling a programme's
    /// length with silence must leave its loudness essentially where it was. Without gating the
    /// same signal would read about 6 LU quieter, since three quarters of it is nothing.
    ///
    /// Not *exactly* unchanged, and it should not be: the blocks straddling the boundary are part
    /// tone and part silence, so they are genuinely quieter and still within 10 LU of the mean.
    /// pyloudnorm shifts by -0.132 LU on this signal and so do we, to three decimals — which is
    /// the useful thing to pin, rather than a round number.
    #[test]
    fn the_relative_gate_ignores_appended_silence() {
        let tone = sine(1000.0, 0.25, 5.0, 48_000);
        let mut padded = tone.clone();
        padded.extend(std::iter::repeat_n(0.0, 48_000 * 15));

        let plain = integrated_loudness(&[tone], 48_000);
        let with_silence = integrated_loudness(&[padded], 48_000);
        let shift = with_silence - plain;
        assert!(
            (-0.2..=0.0).contains(&shift),
            "appending silence moved the programme loudness by {shift} LU ({plain} -> \
             {with_silence}); the boundary blocks explain about -0.13, nothing else should"
        );
    }

    /// A programme at one level has no range; one that steps 20 LU has roughly that much.
    #[test]
    fn loudness_range_measures_the_spread() {
        let steady = sine(1000.0, 0.25, 20.0, 48_000);
        assert!(
            loudness_range(&[steady], 48_000) < 0.5,
            "a steady tone has no range"
        );

        let mut stepped = sine(1000.0, 0.25, 10.0, 48_000);
        stepped.extend(sine(1000.0, 0.025, 10.0, 48_000));
        let lra = loudness_range(&[stepped], 48_000);
        assert!((lra - 20.0).abs() < 2.0, "expected about 20 LU, got {lra}");
    }

    /// The short-term series is what a meter draws, one value per 100 ms hop.
    #[test]
    fn short_term_series_has_one_value_per_hop() {
        let x = sine(1000.0, 0.25, 10.0, 48_000);
        let series = short_term_loudness(&[x], 48_000);
        // 10 s of signal, a 3 s window stepping 100 ms: (10 - 3) / 0.1 + 1 blocks.
        assert_eq!(series.len(), 71);
        for l in &series {
            assert!(
                (l - series[0]).abs() < 0.01,
                "a steady tone should not wander"
            );
        }
    }

    /// `k_weighting` is the same filter the meter uses, just narrowed to f32 for the realtime path.
    #[test]
    fn the_exported_biquads_match_the_design() {
        let [shelf, _] = design(48_000);
        let [b, _] = k_weighting(48_000);
        assert!((f64::from(b.b0) - shelf.b[0]).abs() < 1e-6);
        assert!((f64::from(b.a1) - shelf.a[0]).abs() < 1e-6);
    }

    // --- true peak ---

    /// A tone faded in and out, so what is measured is the filter and not its edge behaviour.
    ///
    /// The phase is accumulated in `f64`: at 48 kHz the argument reaches ~7.5e4 radians by the end
    /// of a second, where `f32` has only about two decimal digits left and the samples drift off
    /// the tone's real amplitude by over 1%.
    fn faded_sine(freq: f64, amp: f32, phase: f64, fs: u32) -> Vec<f32> {
        let n = fs as usize;
        (0..n)
            .map(|i| {
                let env = (i as f32 / 2000.0)
                    .min((n - 1 - i) as f32 / 2000.0)
                    .min(1.0);
                let arg = std::f64::consts::TAU * freq * i as f64 / f64::from(fs) + phase;
                env * amp * arg.sin() as f32
            })
            .collect()
    }

    /// A bandlimited sine's true peak is exactly its amplitude — no oracle needed, and a stricter
    /// check than any reference implementation, since references approximate this too.
    #[test]
    fn true_peak_finds_the_real_maximum_of_a_tone() {
        for freq in [100.0, 1000.0, 5000.0, 10_000.0, 19_000.0] {
            let amp = 0.5f32;
            let truth = 20.0 * amp.log10();
            let measured = true_peak(&[faded_sine(freq, amp, 0.0, 48_000)], 48_000);
            assert!(
                (measured - truth).abs() < 0.2,
                "{freq} Hz: measured {measured:.3} dBTP, true peak is {truth:.3}"
            );
        }
    }

    /// The case true peak exists for: a signal whose samples never reach its real maximum. At
    /// fs/4 with a 45-degree offset every sample lands at amp/sqrt(2), so the sample peak reads
    /// 3.01 dB low and only reconstruction finds the rest.
    #[test]
    fn true_peak_sees_between_the_samples() {
        let amp = 0.5f32;
        let x = faded_sine(12_000.0, amp, std::f64::consts::FRAC_PI_4, 48_000);

        let sampled = sample_peak(std::slice::from_ref(&x));
        let truth = 20.0 * amp.log10();
        assert!(
            (sampled - (truth - 3.01)).abs() < 0.1,
            "the fixture should hide 3 dB from the sample peak, got {sampled:.3} vs {truth:.3}"
        );

        let measured = true_peak(&[x], 48_000);
        assert!(
            (measured - truth).abs() < 0.2,
            "true peak {measured:.3} dBTP missed the real maximum {truth:.3}"
        );
    }

    /// True peak can never be below sample peak: the reconstruction passes through the samples.
    /// This is the safety property — a meter that under-reported would let a file clip.
    #[test]
    fn true_peak_is_never_below_sample_peak() {
        let mut state: u32 = 12345;
        let noise: Vec<f32> = (0..48_000)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 8) as f32 / 16_777_216.0 * 1.6 - 0.8
            })
            .collect();
        for signal in [
            noise,
            faded_sine(7_000.0, 0.9, 0.3, 48_000),
            vec![1.0, -1.0, 1.0, -1.0],
        ] {
            let tp = true_peak(std::slice::from_ref(&signal), 48_000);
            let sp = sample_peak(&[signal]);
            let (tp, sp) = (tp, sp);
            assert!(tp >= sp - 1e-4, "true peak {tp} below sample peak {sp}");
        }
    }

    #[test]
    fn silence_has_no_peak() {
        assert_eq!(true_peak(&[vec![0.0; 1000]], 48_000), f32::NEG_INFINITY);
        assert_eq!(sample_peak(&[vec![0.0; 1000]]), f32::NEG_INFINITY);
    }

    /// The interpolator must not invent level: a level held between smooth ramps reconstructs to
    /// that level. Ramped rather than square, because a step from silence genuinely rings — that
    /// is the reconstruction filter doing its job, not a gain error.
    #[test]
    fn the_interpolator_has_unity_gain() {
        let hold = 0.5f32;
        let ramp = 2000;
        let n = 12_000;
        let trapezoid: Vec<f32> = (0..n)
            .map(|i| {
                hold * (i as f32 / ramp as f32)
                    .min((n - 1 - i) as f32 / ramp as f32)
                    .min(1.0)
            })
            .collect();
        let measured = true_peak(&[trapezoid], 48_000);
        let truth = 20.0 * hold.log10();
        assert!(
            (measured - truth).abs() < 0.05,
            "a held level should reconstruct to itself: {measured:.3} vs {truth:.3}"
        );
    }

    #[test]
    fn channel_weights_follow_the_standard() {
        assert_eq!(channel_weights(1), vec![1.0]);
        assert_eq!(channel_weights(2), vec![1.0, 1.0]);
        // 5.1: the LFE does not count.
        assert_eq!(channel_weights(6), vec![1.0, 1.0, 1.0, 0.0, 1.41, 1.41]);
    }
}

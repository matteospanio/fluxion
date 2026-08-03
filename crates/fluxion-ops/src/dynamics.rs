//! Dynamic-range processing: a feed-forward compressor / expander (compand).
//!
//! [`CompandCoeffs`] holds the design-stage coefficients (a one-pole peak-envelope follower plus a
//! soft-knee gain computer) and exposes a single per-sample [`step`](CompandCoeffs::step). The
//! offline [`compand`] kernel and the realtime `RtGraph::Compand` node both drive that same `step`,
//! so streaming is sample-for-sample identical to the batch pass.
//!
//! Coefficients are computed here in an explicit design stage, never lazily inside
//! the sample loop.

/// Designed coefficients for a soft-knee feed-forward compressor.
///
/// The envelope follower is a one-pole peak detector with separate attack/release smoothing; the
/// gain computer is the standard soft-knee downward-compression curve in decibels (Reiss &
/// McPherson, *Audio Effects*), plus a static `makeup` gain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompandCoeffs {
    /// Attack smoothing coefficient `exp(-1/(attack·fs))` (0 = instantaneous).
    pub attack: f32,
    /// Release smoothing coefficient `exp(-1/(release·fs))`.
    pub release: f32,
    /// Threshold in dBFS above which gain reduction begins.
    pub threshold_db: f32,
    /// Compression ratio (`>= 1`; `1` is a no-op).
    pub ratio: f32,
    /// Knee width in dB (`0` = hard knee).
    pub knee_db: f32,
    /// Static make-up gain in dB, applied to every sample.
    pub makeup_db: f32,
}

impl CompandCoeffs {
    /// Design from times (seconds) and levels. `attack`/`release` of `0` follow the peak instantly.
    pub fn design(
        attack_s: f32,
        release_s: f32,
        threshold_db: f32,
        ratio: f32,
        knee_db: f32,
        makeup_db: f32,
        fs: u32,
    ) -> CompandCoeffs {
        // exp(-1/(t·fs)): t = 0 -> 1/0 = +inf -> coefficient 0 (env jumps straight to |x|).
        let coef = |t: f32| (-1.0 / (t.max(0.0) * fs as f32)).exp();
        CompandCoeffs {
            attack: coef(attack_s),
            release: coef(release_s),
            threshold_db,
            ratio: ratio.max(1.0),
            knee_db: knee_db.max(0.0),
            makeup_db,
        }
    }

    /// Advance the envelope by one sample and apply the gain. Returns `(new_env, output_sample)`.
    ///
    /// `env` is the previous peak-envelope state (start a fresh signal at `0.0`).
    pub fn step(&self, env: f32, x: f32) -> (f32, f32) {
        let level = x.abs();
        // One-pole peak follower: rise with `attack`, fall with `release`.
        let coef = if level > env {
            self.attack
        } else {
            self.release
        };
        let env = coef * env + (1.0 - coef) * level;

        // Gain computer in dB. `1e-9` floors the log so digital silence is finite.
        let env_db = 20.0 * (env + 1e-9).log10();
        let over = env_db - self.threshold_db;
        let slope = 1.0 / self.ratio - 1.0; // <= 0 for compression
        let w = self.knee_db;
        let gain_db = if w > 0.0 {
            if 2.0 * over < -w {
                0.0 // below the knee: unity
            } else if 2.0 * over.abs() <= w {
                // inside the knee: quadratic interpolation
                slope * (over + w * 0.5).powi(2) / (2.0 * w)
            } else {
                slope * over // above the knee: full ratio
            }
        } else if over > 0.0 {
            slope * over // hard knee
        } else {
            0.0
        };

        let gain = 10f32.powf((gain_db + self.makeup_db) / 20.0);
        (env, x * gain)
    }
}

/// Feed-forward compressor / expander (compand): compress a channel's dynamic range with a soft-knee
/// gain computer driven by a one-pole peak-envelope follower. Stateful (the envelope carries across
/// samples), length-preserving, and fs-preserving.
#[allow(clippy::too_many_arguments)]
pub fn compand(
    input: &[f32],
    attack_s: f32,
    release_s: f32,
    threshold_db: f32,
    ratio: f32,
    knee_db: f32,
    makeup_db: f32,
    fs: u32,
) -> Vec<f32> {
    let c = CompandCoeffs::design(
        attack_s,
        release_s,
        threshold_db,
        ratio,
        knee_db,
        makeup_db,
        fs,
    );
    let mut env = 0.0f32;
    input
        .iter()
        .map(|&x| {
            let (e, y) = c.step(env, x);
            env = e;
            y
        })
        .collect()
}

// --- true-peak limiter (ROADMAP M3) ----------------------------------------------------------

/// Limit the signal so its **true** peak never exceeds `ceiling_db` dBTP.
///
/// A sample-peak limiter leaves inter-sample overshoot behind, and that is exactly what clips in a
/// converter or a lossy encoder — so the gain here is computed from the reconstructed waveform
/// (see [`crate::loudness::true_peak_envelope`]), not from the samples.
///
/// One gain curve is applied to every channel, so a loud left channel ducks the right with it and
/// the image does not wander.
///
/// `lookahead` seconds of anticipation lets the gain be already down when a peak arrives instead of
/// clamping after the fact; `release` seconds sets how quickly it comes back. Offline this costs no
/// latency, because the whole signal is present — a realtime lowering of the same op would carry
/// exactly `lookahead` samples of delay.
///
/// ```
/// use fluxion_ops::{limit, loudness::true_peak};
/// let fs = 48_000;
/// let mut loud: Vec<Vec<f32>> = vec![(0..fs)
///     .map(|n| 0.9 * (std::f32::consts::TAU * 800.0 * n as f32 / fs as f32).sin())
///     .collect()];
/// limit(&mut loud, -6.0, 0.005, 0.05, fs as u32);
/// assert!(true_peak(&loud, fs as u32) <= -6.0 + 0.1);
/// ```
pub fn limit(channels: &mut [Vec<f32>], ceiling_db: f32, lookahead: f32, release: f32, fs: u32) {
    let frames = channels.iter().map(Vec::len).max().unwrap_or(0);
    if frames == 0 {
        return;
    }
    let ceiling = 10f32.powf(ceiling_db / 20.0);

    // Applying a *varying* gain moves the reconstructed curve, so one pass computed from the input
    // can leave a little overshoot behind — the gain modulation adds spectrum of its own. Each pass
    // measures the signal it is actually about to change, and the residual falls off fast: full
    // scale noise, the worst case found, needs two.
    for _ in 0..PASSES {
        if !limit_once(channels, ceiling, lookahead, release, fs, frames) {
            return;
        }
    }

    // Last resort. The passes above converge quickly, but M3's property is "never exceeds the
    // ceiling, on **any** input", and a claim like that cannot rest on convergence being fast
    // enough. If anything is still over, one flat gain across the whole signal settles it — at the
    // cost of a little level, which is the right trade against handing back something that clips.
    let residual = crate::loudness::true_peak_envelope(channels)
        .into_iter()
        .fold(0.0f32, f32::max);
    if residual > ceiling {
        let trim = ceiling / residual;
        for channel in channels.iter_mut() {
            for sample in channel.iter_mut() {
                *sample *= trim;
            }
        }
    }
}

/// How many times to re-measure and re-limit. Three is one more than the worst case measured, and
/// each pass after the first is nearly free because there is almost nothing left to do.
const PASSES: usize = 3;

/// One pass. Returns whether anything exceeded the ceiling, i.e. whether another pass is worth it.
fn limit_once(
    channels: &mut [Vec<f32>],
    ceiling: f32,
    lookahead: f32,
    release: f32,
    fs: u32,
    frames: usize,
) -> bool {
    // What each sample needs, from the reconstructed curve rather than the samples.
    let envelope = crate::loudness::true_peak_envelope(channels);
    if envelope.iter().fold(0.0f32, |m, v| m.max(*v)) <= ceiling {
        return false;
    }
    let required: Vec<f32> = envelope
        .iter()
        .map(|&peak| if peak > ceiling { ceiling / peak } else { 1.0 })
        .collect();

    // Anticipate: the gain at n is the worst thing arriving within the lookahead window, so the
    // reduction can be in place before the peak instead of clamping on top of it.
    let window = ((lookahead * fs as f32).round() as usize).max(1);
    let mut anticipated = vec![1.0f32; frames];
    for (n, slot) in anticipated.iter_mut().enumerate() {
        let hi = (n + window + 1).min(frames);
        *slot = required[n..hi].iter().copied().fold(1.0f32, f32::min);
    }

    // Follow it with an asymmetric one-pole: fast enough down to complete inside the lookahead,
    // slow up so the recovery does not pump. Smoothness is not cosmetic here — a gain that moves
    // sample-to-sample modulates the signal and *creates* the inter-sample peaks it is meant to
    // remove, which is what an earlier version of this did.
    let attack = coefficient((window / 3).max(1));
    let release_samples = ((release * fs as f32).round() as usize).max(1);
    let release_coeff = coefficient(release_samples);

    let mut gain = vec![1.0f32; frames];
    let mut g = 1.0f32;
    for (n, slot) in gain.iter_mut().enumerate() {
        let target = anticipated[n];
        let coeff = if target < g { attack } else { release_coeff };
        g += (target - g) * coeff;
        *slot = g;
    }

    for channel in channels.iter_mut() {
        for (sample, g) in channel.iter_mut().zip(&gain) {
            *sample *= g;
        }
    }
    true
}

/// One-pole coefficient that settles in about `samples` steps (three time constants).
fn coefficient(samples: usize) -> f32 {
    1.0 - (-3.0 / samples as f32).exp()
}

#[cfg(test)]
mod tests {
    use super::compand;

    const FS: u32 = 48_000;

    #[test]
    fn loud_signal_is_compressed_toward_threshold() {
        // A steady tone at 0 dBFS peak, threshold -20 dB, ratio 4: after the envelope settles the
        // output peak sits well below the input but above the threshold's linear level.
        let x: Vec<f32> = (0..FS as usize)
            .map(|i| (2.0 * std::f32::consts::PI * 1_000.0 * i as f32 / FS as f32).sin())
            .collect();
        let y = compand(&x, 0.005, 0.05, -20.0, 4.0, 6.0, 0.0, FS);
        let settled = &y[FS as usize / 2..];
        let peak = settled.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        assert!(peak < 0.9, "expected gain reduction, peak = {peak}");
        // -20 dBFS threshold, 4:1: a 0 dB input maps to about -15 dBFS out (~0.178) plus knee, so
        // the peak lands in a sane compressed band, not silence.
        assert!(peak > 0.1, "over-compressed, peak = {peak}");
    }

    #[test]
    fn quiet_signal_below_threshold_passes_through() {
        // Peak -40 dBFS (0.01), threshold -20 dB, no make-up: gain ~= unity.
        let x: Vec<f32> = (0..4_000).map(|i| 0.01 * (0.05 * i as f32).sin()).collect();
        let y = compand(&x, 0.01, 0.1, -20.0, 4.0, 6.0, 0.0, FS);
        for (a, b) in y.iter().zip(&x) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }

    #[test]
    fn makeup_gain_scales_quiet_signal() {
        // Below threshold with +6 dB make-up: output ~= input * 2.
        let x: Vec<f32> = (0..2_000).map(|i| 0.01 * (0.05 * i as f32).sin()).collect();
        let y = compand(&x, 0.01, 0.1, -20.0, 4.0, 6.0, 6.0, FS);
        let g = 10f32.powf(6.0 / 20.0);
        for (a, b) in y.iter().zip(&x) {
            assert!((a - b * g).abs() < 1e-3, "{a} vs {}", b * g);
        }
    }
}

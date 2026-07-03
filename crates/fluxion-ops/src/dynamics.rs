//! Dynamic-range processing: a feed-forward compressor / expander (compand).
//!
//! [`CompandCoeffs`] holds the design-stage coefficients (a one-pole peak-envelope follower plus a
//! soft-knee gain computer) and exposes a single per-sample [`step`](CompandCoeffs::step). The
//! offline [`compand`] kernel and the realtime `RtGraph::Compand` node both drive that same `step`,
//! so streaming is sample-for-sample identical to the batch pass.
//!
//! Coefficients are computed here in an explicit design stage (`PROJECT.md` §3), never lazily inside
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

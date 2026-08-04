//! Envelope follower (ROADMAP S2) — how loud the signal is *right now*.
//!
//! The block under gates, duckers, compressors and meters. It is not an effect: what comes out is a
//! control signal, not audio, which is why it is a plain type here rather than an
//! [`OpKind`](fluxion_core::OpKind).
//!
//! One pole, two time constants: the envelope rises with `attack` and falls with `release`. That
//! asymmetry is the whole point — a gate wants to open fast and close slowly, and a symmetric
//! smoother cannot do both.
//!
//! Two detectors, because the two answers differ and the difference matters. [`Detector::Peak`]
//! follows `|x|` and is what stops a converter clipping. [`Detector::Rms`] follows the power and is
//! closer to what a listener calls loudness — for a sine it reads 3 dB lower, which is the
//! crest factor, not an error.

/// What "level" means to a follower.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Detector {
    /// The rectified sample, `|x|`. Reads the crest — what a limiter has to see.
    Peak,
    /// The root of the smoothed power. Reads the body — closer to loudness, and 3 dB below `Peak`
    /// on a sine.
    Rms,
}

/// One-pole smoothing coefficient for a time constant of `seconds`: `exp(-1/(t·fs))`.
///
/// `t` is the RC time constant — the envelope covers `1 - 1/e` (63 %) of a step in that long. A
/// time of 0 gives a coefficient of 0, which is "follow instantly", not "never move".
///
/// Shared with [`CompandCoeffs::design`](crate::dynamics::CompandCoeffs::design) so that an attack
/// time means the same thing everywhere in the crate.
pub fn smoothing_coef(seconds: f32, fs: u32) -> f32 {
    // t = 0 -> 1/0 = +inf -> exp(-inf) = 0, and the envelope jumps straight to the level.
    (-1.0 / (seconds.max(0.0) * fs as f32)).exp()
}

/// A one-pole envelope follower with separate attack and release.
///
/// ```
/// use fluxion_ops::follower::{Detector, Follower};
///
/// // Fast attack, slow release — the shape a gate or a meter wants.
/// let mut f = Follower::new(0.001, 0.100, Detector::Peak, 48_000);
/// for _ in 0..48_000 {
///     f.step(1.0);
/// }
/// assert!((f.value() - 1.0).abs() < 1e-3);
/// ```
#[derive(Clone, Debug)]
pub struct Follower {
    attack: f32,
    release: f32,
    detector: Detector,
    /// The envelope, or for [`Detector::Rms`] the mean square — smoothing the power and rooting it
    /// at the end is what makes it an RMS rather than a smoothed rectifier.
    state: f32,
}

impl Follower {
    /// Build a follower. `attack_s` and `release_s` are RC time constants in seconds; 0 is
    /// instantaneous.
    pub fn new(attack_s: f32, release_s: f32, detector: Detector, fs: u32) -> Follower {
        Follower {
            attack: smoothing_coef(attack_s, fs),
            release: smoothing_coef(release_s, fs),
            detector,
            state: 0.0,
        }
    }

    /// Advance by one sample and return the envelope.
    #[inline]
    pub fn step(&mut self, x: f32) -> f32 {
        let level = match self.detector {
            Detector::Peak => x.abs(),
            Detector::Rms => x * x,
        };
        // Rising or falling is decided on the smoothed quantity itself, so for RMS the comparison
        // is between powers — the same ordering, since the root is monotonic.
        let coef = if level > self.state {
            self.attack
        } else {
            self.release
        };
        self.state = coef * self.state + (1.0 - coef) * level;
        self.value()
    }

    /// The envelope now, without advancing.
    #[inline]
    pub fn value(&self) -> f32 {
        match self.detector {
            Detector::Peak => self.state,
            Detector::Rms => self.state.max(0.0).sqrt(),
        }
    }

    /// Follow a whole block, writing one envelope sample per input sample.
    ///
    /// Allocation-free, so it can run in a callback. `out` shorter than `x` stops early; longer is
    /// left untouched past the end.
    pub fn process(&mut self, x: &[f32], out: &mut [f32]) {
        for (sample, slot) in x.iter().zip(out.iter_mut()) {
            *slot = self.step(*sample);
        }
    }

    /// Start again from silence. The times are kept.
    pub fn reset(&mut self) {
        self.state = 0.0;
    }

    /// Which detector this was built with.
    pub fn detector(&self) -> Detector {
        self.detector
    }
}

/// The envelope of a whole signal — [`Follower`] over a slice, for offline use.
pub fn envelope(x: &[f32], attack_s: f32, release_s: f32, detector: Detector, fs: u32) -> Vec<f32> {
    let mut f = Follower::new(attack_s, release_s, detector, fs);
    x.iter().map(|&s| f.step(s)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    const FS: u32 = 48_000;

    /// The attack curve in closed form: fed a step of 1 from rest, the envelope after `n` samples
    /// is `1 - a^n`. This is the check S2 names, and it pins the coefficient convention — a
    /// follower built on `1 - exp(-2.2/(t·fs))` (10–90 % rise time rather than RC) would be a
    /// perfectly good follower and would fail this by a wide margin.
    #[test]
    fn a_step_follows_the_attack_curve() {
        let attack_s = 0.010;
        let a = smoothing_coef(attack_s, FS);
        let mut f = Follower::new(attack_s, 1.0, Detector::Peak, FS);

        for n in 1..=4_800 {
            let got = f.step(1.0);
            let want = 1.0 - a.powi(n);
            assert!(
                (got - want).abs() < 1e-4,
                "sample {n}: envelope {got}, closed form {want}"
            );
        }
        // One time constant in, it is 63 % of the way there — the definition of the constant.
        let mut f = Follower::new(attack_s, 1.0, Detector::Peak, FS);
        for _ in 0..(attack_s * FS as f32) as usize {
            f.step(1.0);
        }
        assert!((f.value() - 0.632).abs() < 1e-3, "one tau: {}", f.value());
    }

    /// And the other half: from a settled 1, silence decays as `r^n`.
    #[test]
    fn silence_follows_the_release_curve() {
        let release_s = 0.050;
        let r = smoothing_coef(release_s, FS);
        let mut f = Follower::new(0.0, release_s, Detector::Peak, FS);

        f.step(1.0); // instantaneous attack, so this lands exactly on 1
        assert_eq!(f.value(), 1.0);
        for n in 1..=4_800 {
            let got = f.step(0.0);
            let want = r.powi(n);
            assert!(
                (got - want).abs() < 1e-4,
                "sample {n}: envelope {got}, closed form {want}"
            );
        }
    }

    /// The two detectors disagree by the crest factor, and that is the reason both exist: a sine of
    /// amplitude A peaks at A and measures A/√2 in RMS, which is 3.01 dB lower.
    #[test]
    fn the_detectors_differ_by_the_crest_factor() {
        let x: Vec<f32> = (0..FS as usize)
            .map(|i| (TAU * 100.0 * i as f32 / FS as f32).sin() * 0.8)
            .collect();

        // Instantaneous attack, so the peak reading is the crest itself rather than a smoothed
        // approach to it — a 1 ms attack on a 100 Hz sine lands 0.16 dB low, which is the follower
        // working, not failing, and would only make this test about the attack time instead.
        let peak = envelope(&x, 0.0, 0.200, Detector::Peak, FS);
        let rms = envelope(&x, 0.050, 0.050, Detector::Rms, FS);

        // A peak follower ripples between the crests of a 100 Hz sine — the release is what it does
        // *between* peaks — so the level it reads is the top of that ripple, not any one sample of
        // it. The RMS follower has no such ripple to speak of, which is rather the point of it.
        let half = FS as usize / 2;
        let settled_peak = peak[half..].iter().fold(0.0f32, |m, v| m.max(*v));
        let settled_rms = rms[half..].iter().sum::<f32>() / rms[half..].len() as f32;
        assert!(
            (settled_peak - 0.8).abs() < 0.01,
            "peak read {settled_peak}, expected 0.8"
        );
        assert!(
            (settled_rms - 0.8 / 2f32.sqrt()).abs() < 0.01,
            "rms read {settled_rms}, expected {}",
            0.8 / 2f32.sqrt()
        );
        let db = 20.0 * (settled_peak / settled_rms).log10();
        assert!((db - 3.01).abs() < 0.15, "crest factor {db:.2} dB");
    }

    /// Zero times mean "no smoothing at all", not "never move" — the edge the coefficient formula
    /// reaches through a division by zero.
    #[test]
    fn zero_times_follow_instantly() {
        let mut f = Follower::new(0.0, 0.0, Detector::Peak, FS);
        assert_eq!(f.step(0.7), 0.7);
        assert_eq!(f.step(-0.2), 0.2);
        assert_eq!(f.step(0.0), 0.0);
    }

    /// Block and sample-at-a-time are the same computation, which is what lets a realtime node and
    /// an offline pass share one follower.
    #[test]
    fn a_block_is_the_same_as_one_sample_at_a_time() {
        let x: Vec<f32> = (0..1_000).map(|i| (i as f32 * 0.1).sin()).collect();
        let whole = envelope(&x, 0.005, 0.050, Detector::Rms, FS);

        let mut f = Follower::new(0.005, 0.050, Detector::Rms, FS);
        let mut blocked = vec![0.0f32; x.len()];
        for (chunk, out) in x.chunks(128).zip(blocked.chunks_mut(128)) {
            f.process(chunk, out);
        }
        assert_eq!(whole, blocked);
    }

    #[test]
    fn reset_starts_again_from_silence() {
        let mut f = Follower::new(0.010, 0.010, Detector::Peak, FS);
        for _ in 0..1000 {
            f.step(1.0);
        }
        assert!(f.value() > 0.5);
        f.reset();
        assert_eq!(f.value(), 0.0);
    }
}

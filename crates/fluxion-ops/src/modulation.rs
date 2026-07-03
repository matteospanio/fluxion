//! Modulated delay/phase effects: [`chorus`], [`flanger`], and [`phaser`].
//!
//! Each is a single-voice, length-preserving, fs-preserving kernel (kept minimally correct rather
//! than feature-maximal — one LFO, one voice). Chorus and flanger use a linearly-interpolated
//! fractional delay whose length is swept by a low-frequency oscillator; the phaser sweeps a cascade
//! of first-order all-pass stages. Flanger and phaser add a feedback path (bounded `< 1` for BIBO
//! stability); chorus is purely feed-forward.
//!
//! ponytail: single-voice, sine-LFO only — multi-voice/stereo-spread chorus and triangle/adjustable
//! LFO shapes are a later refinement. The offline read is `O(N)`; a realtime modulated-delay node is
//! a follow-up (today these lower to `op_rt = None`).

use std::f32::consts::PI;

/// Linearly-interpolated read of `x` at fractional index `pos` (0 before the signal starts, 0 past
/// its end).
fn read_interp(x: &[f32], pos: f32) -> f32 {
    if pos < 0.0 {
        return 0.0;
    }
    let i = pos.floor() as usize;
    let frac = pos - i as f32;
    let a = x.get(i).copied().unwrap_or(0.0);
    let b = x.get(i + 1).copied().unwrap_or(0.0);
    a * (1.0 - frac) + b * frac
}

/// Chorus: a single LFO-modulated fractional-delay voice blended with the dry signal.
///
/// The delay sweeps in `[delay, delay + depth]` seconds at `rate` Hz; `mix` (0..1) is the wet/dry
/// blend. Feed-forward (no feedback), so it is unconditionally stable. Length-preserving.
pub fn chorus(
    input: &[f32],
    rate_hz: f32,
    depth_s: f32,
    delay_s: f32,
    mix: f32,
    fs: u32,
) -> Vec<f32> {
    let base = delay_s * fs as f32;
    let depth = depth_s * fs as f32;
    let w = 2.0 * PI * rate_hz / fs as f32;
    (0..input.len())
        .map(|n| {
            let lfo = 0.5 - 0.5 * (w * n as f32).cos(); // 0..1
            let d = base + depth * lfo;
            let xd = read_interp(input, n as f32 - d);
            (1.0 - mix) * input[n] + mix * xd
        })
        .collect()
}

/// Flanger: a short LFO-modulated delay with `feedback`, blended with the dry signal.
///
/// The delay sweeps in `[delay, delay + depth]` seconds at `rate` Hz; `feedback` (`|f| < 1`) recirculates
/// the delayed signal for the characteristic resonant sweep; `mix` (0..1) is the wet/dry blend.
/// Length-preserving.
#[allow(clippy::too_many_arguments)]
pub fn flanger(
    input: &[f32],
    rate_hz: f32,
    depth_s: f32,
    delay_s: f32,
    feedback: f32,
    mix: f32,
    fs: u32,
) -> Vec<f32> {
    let base = delay_s * fs as f32;
    let depth = depth_s * fs as f32;
    let w = 2.0 * PI * rate_hz / fs as f32;
    let n = input.len();
    let mut state = vec![0.0f32; n]; // delay-line state: x[n] + feedback·delayed
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let lfo = 0.5 - 0.5 * (w * i as f32).cos();
        let d = base + depth * lfo;
        let delayed = read_interp(&state, i as f32 - d);
        state[i] = input[i] + feedback * delayed;
        out[i] = (1.0 - mix) * input[i] + mix * delayed;
    }
    out
}

/// Phaser: an LFO-swept cascade of first-order all-pass stages with `feedback`, blended with the dry
/// signal.
///
/// Four all-pass stages share one swept coefficient `a ∈ [−0.9·depth, 0.9·depth]` at `rate` Hz; each
/// stage is `y[n] = a·x[n] + x[n−1] − a·y[n−1]` (unity magnitude, so the moving phase notches are the
/// audible effect). `feedback` (`|f| < 1`) recirculates the cascade output; `mix` (0..1) is the
/// wet/dry blend. Length-preserving.
pub fn phaser(
    input: &[f32],
    rate_hz: f32,
    depth: f32,
    feedback: f32,
    mix: f32,
    fs: u32,
) -> Vec<f32> {
    const STAGES: usize = 4;
    let w = 2.0 * PI * rate_hz / fs as f32;
    let depth = depth.clamp(0.0, 1.0);
    let mut ap_x = [0.0f32; STAGES]; // per-stage previous input
    let mut ap_y = [0.0f32; STAGES]; // per-stage previous output
    let mut fb = 0.0f32;
    let mut out = vec![0.0f32; input.len()];
    for (i, o) in out.iter_mut().enumerate() {
        let lfo = 0.5 - 0.5 * (w * i as f32).cos(); // 0..1
        let a = 0.9 * depth * (2.0 * lfo - 1.0); // -0.9·depth .. 0.9·depth, |a| < 1
        let mut s = input[i] + feedback * fb;
        for k in 0..STAGES {
            let y = a * s + ap_x[k] - a * ap_y[k];
            ap_x[k] = s;
            ap_y[k] = y;
            s = y;
        }
        fb = s;
        *o = (1.0 - mix) * input[i] + mix * s;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{chorus, flanger, phaser};

    fn sig(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (0.05 * i as f32).sin() + 0.3 * (0.21 * i as f32).sin())
            .collect()
    }

    #[test]
    fn chorus_is_dry_with_zero_delay_and_depth() {
        // delay 0, depth 0 -> the wet tap reads the current sample, so wet == dry for any mix.
        let x = sig(200);
        let y = chorus(&x, 1.5, 0.0, 0.0, 1.0, 48_000);
        assert_eq!(y.len(), x.len());
        for (a, b) in y.iter().zip(&x) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn chorus_taps_a_fixed_delay_when_depth_zero() {
        // depth 0, delay = 10 samples (integer), mix 1 -> pure 10-sample delay.
        let x = sig(300);
        let d = 10usize;
        let y = chorus(&x, 1.5, 0.0, d as f32 / 48_000.0, 1.0, 48_000);
        for i in d..x.len() {
            assert!(
                (y[i] - x[i - d]).abs() < 1e-4,
                "at {i}: {} vs {}",
                y[i],
                x[i - d]
            );
        }
    }

    #[test]
    fn flanger_matches_delay_when_feedback_zero() {
        // feedback 0, depth 0, integer delay -> (1-mix)·x + mix·x[n-D].
        let x = sig(400);
        let (d, mix) = (7usize, 0.6f32);
        let y = flanger(&x, 0.5, 0.0, d as f32 / 48_000.0, 0.0, mix, 48_000);
        for i in d..x.len() {
            let want = (1.0 - mix) * x[i] + mix * x[i - d];
            assert!((y[i] - want).abs() < 1e-4, "at {i}: {} vs {want}", y[i]);
        }
    }

    #[test]
    fn flanger_feedback_stays_bounded() {
        let x = sig(4_000);
        let y = flanger(&x, 0.5, 0.002, 0.001, 0.9, 0.5, 48_000);
        assert_eq!(y.len(), x.len());
        assert!(
            y.iter().all(|v| v.is_finite() && v.abs() < 20.0),
            "flanger blew up"
        );
    }

    #[test]
    fn phaser_is_dry_when_mix_zero_and_bounded_otherwise() {
        let x = sig(2_000);
        // mix 0 -> exactly the dry signal.
        let dry = phaser(&x, 0.5, 0.5, 0.5, 0.0, 48_000);
        for (a, b) in dry.iter().zip(&x) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
        // wet output stays finite/bounded (all-pass stages are unity-gain, feedback < 1).
        let wet = phaser(&x, 0.5, 1.0, 0.9, 1.0, 48_000);
        assert!(
            wet.iter().all(|v| v.is_finite() && v.abs() < 20.0),
            "phaser blew up"
        );
    }
}

//! Simple amplitude effects and their backward passes.

use std::f32::consts::PI;

/// Multiply a channel in place by a linear gain factor.
pub fn gain(channel: &mut [f32], factor: f32) {
    for x in channel {
        *x *= factor;
    }
}

/// Fade-curve shape (matches [`OpKind::Fade`](fluxion_core::OpKind)'s `shape` parameter).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FadeShape {
    /// Straight-line ramp `f(t) = t`.
    Linear,
    /// Quarter-sine ramp `f(t) = sin(t·π/2)` — the SoX default (a gentle, click-free curve).
    QuarterSine,
    /// Half-sine (raised-cosine) ramp `f(t) = (1 − cos(t·π))/2`.
    HalfSine,
}

impl FadeShape {
    /// Decode from the `shape` parameter value (0 = linear, 1 = quarter-sine, 2 = half-sine);
    /// anything else falls back to quarter-sine (the SoX default).
    pub fn from_param(shape: f32) -> FadeShape {
        match shape.round() as i32 {
            0 => FadeShape::Linear,
            2 => FadeShape::HalfSine,
            _ => FadeShape::QuarterSine,
        }
    }

    /// The ramp value at normalized position `t ∈ [0, 1]` (`f(0) = 0`, `f(1) = 1`).
    fn curve(self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            FadeShape::Linear => t,
            FadeShape::QuarterSine => (t * PI * 0.5).sin(),
            FadeShape::HalfSine => 0.5 - 0.5 * (t * PI).cos(),
        }
    }
}

/// Apply a fade-in over the first `fade_in` samples and a fade-out over the last `fade_out` samples,
/// in place, using the given `shape` curve. Both envelopes multiply, so overlapping regions compose.
///
/// The fade-in reaches unity at sample `fade_in − 1` (gain `shape((i+1)/fade_in)`); the fade-out
/// reaches zero at the final sample (gain `shape((n−1−i)/fade_out)`). Passing `0` for either length
/// disables that side. Length-preserving.
pub fn fade(channel: &mut [f32], fade_in: usize, fade_out: usize, shape: FadeShape) {
    let n = channel.len();
    let fade_in = fade_in.min(n);
    let fade_out = fade_out.min(n);
    for (i, x) in channel.iter_mut().enumerate() {
        if fade_in > 0 && i < fade_in {
            *x *= shape.curve((i + 1) as f32 / fade_in as f32);
        }
        if fade_out > 0 && i >= n - fade_out {
            let k = n - 1 - i; // 0 at the last sample .. fade_out-1 at the start of the tail
            *x *= shape.curve(k as f32 / fade_out as f32);
        }
    }
}

/// Tremolo: multiply a channel in place by a low-frequency amplitude LFO,
/// `g[n] = 1 − depth·½·(1 − cos(2π·rate·n/fs))` — unity at `n = 0`, dipping by up to `depth` (0..1)
/// at `rate` Hz. Length-preserving.
pub fn tremolo(channel: &mut [f32], rate_hz: f32, depth: f32, fs: u32) {
    let depth = depth.clamp(0.0, 1.0);
    let w = 2.0 * PI * rate_hz / fs as f32;
    for (n, x) in channel.iter_mut().enumerate() {
        let g = 1.0 - depth * 0.5 * (1.0 - (w * n as f32).cos());
        *x *= g;
    }
}

/// Overdrive: `gain_db` of linear drive through a `tanh` soft-clipper with a `colour` asymmetry bias
/// (adds even harmonics), in place. `y = tanh(drive·x + colour) − tanh(colour)` keeps silence silent.
/// Memoryless and nonlinear (not differentiable here). Length-preserving.
pub fn overdrive(channel: &mut [f32], gain_db: f32, colour: f32) {
    let drive = 10f32.powf(gain_db / 20.0);
    let bias = colour.tanh();
    for x in channel.iter_mut() {
        *x = (drive * *x + colour).tanh() - bias;
    }
}

/// Reverse a channel in time. Length-preserving; **not** realtime (needs the whole signal).
pub fn reverse(input: &[f32]) -> Vec<f32> {
    let mut out = input.to_vec();
    out.reverse();
    out
}

/// Backward pass for [`gain`] (`y = factor · x`).
///
/// Returns the input cotangent (`factor · grad_out`) and the scalar gradient of the gain factor
/// (`Σ grad_out · x`).
pub fn gain_vjp(input: &[f32], factor: f32, grad_out: &[f32]) -> (Vec<f32>, f32) {
    let grad_in = grad_out.iter().map(|&g| g * factor).collect();
    let grad_factor = grad_out.iter().zip(input).map(|(&g, &x)| g * x).sum();
    (grad_in, grad_factor)
}

/// Peak-normalize a multichannel signal in place so its largest absolute sample equals `target`.
///
/// The gain is computed from the global peak across all channels, preserving inter-channel balance.
/// A silent signal (peak 0) is left unchanged.
pub fn normalize_peak(channels: &mut [Vec<f32>], target: f32) {
    let peak = channels
        .iter()
        .flat_map(|c| c.iter())
        .fold(0.0f32, |m, &x| m.max(x.abs()));
    if peak > 0.0 {
        let g = target / peak;
        for c in channels.iter_mut() {
            gain(c, g);
        }
    }
}

/// Backward pass for single-channel peak-normalization (`y = x · target / peak`, `peak = maxₖ|xₖ|`).
///
/// Returns the input cotangent. The gain depends on `x` through the peak, so the argmax sample `m`
/// carries an extra term: `grad_in = (target/peak)·grad_out`, then
/// `grad_in[m] -= target·sign(x[m])·⟨grad_out, x⟩ / peak²`. A silent input (peak 0) is an identity,
/// so the gradient passes straight through.
///
/// ponytail: `peak = max` is non-smooth at ties (two samples of equal magnitude); the gradient there
/// is a valid subgradient. Fine for training — real signals don't sit exactly on a tie.
pub fn normalize_vjp(input: &[f32], target: f32, grad_out: &[f32]) -> Vec<f32> {
    assert_eq!(
        input.len(),
        grad_out.len(),
        "normalize_vjp: grad_out ({}) must match input length ({})",
        grad_out.len(),
        input.len()
    );
    let (mut peak, mut m) = (0.0f32, 0usize);
    for (i, &x) in input.iter().enumerate() {
        if x.abs() > peak {
            peak = x.abs();
            m = i;
        }
    }
    if peak == 0.0 {
        return grad_out.to_vec(); // identity forward → pass-through gradient
    }
    let s = target / peak;
    let mut grad: Vec<f32> = grad_out.iter().map(|&g| s * g).collect();
    let dot: f32 = grad_out.iter().zip(input).map(|(&g, &x)| g * x).sum();
    grad[m] -= target * input[m].signum() * dot / (peak * peak);
    grad
}

#[cfg(test)]
mod tests {
    use super::{
        FadeShape, fade, gain, gain_vjp, normalize_peak, normalize_vjp, overdrive, reverse, tremolo,
    };

    #[test]
    fn gain_scales() {
        let mut c = vec![1.0, -2.0, 0.5];
        gain(&mut c, 2.0);
        assert_eq!(c, vec![2.0, -4.0, 1.0]);
    }

    #[test]
    fn linear_fade_ramps_in_and_out() {
        // Constant 1.0; linear fade-in over 4, fade-out over 4, on an 8-sample buffer.
        let mut c = vec![1.0f32; 8];
        fade(&mut c, 4, 4, FadeShape::Linear);
        // fade-in: (i+1)/4 -> 0.25, 0.5, 0.75, 1.0; fade-out: (n-1-i)/4 -> 0.75, 0.5, 0.25, 0.0.
        assert_eq!(c, vec![0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25, 0.0]);
    }

    #[test]
    fn fade_shapes_hit_endpoints() {
        for shape in [
            FadeShape::Linear,
            FadeShape::QuarterSine,
            FadeShape::HalfSine,
        ] {
            let mut c = vec![1.0f32; 16];
            fade(&mut c, 8, 8, shape);
            assert!(c[0] > 0.0 && c[0] < 1.0); // ramps up from near-zero
            assert!((c[7] - 1.0).abs() < 1e-6); // unity at the fade-in boundary
            assert!(c[15].abs() < 1e-6); // silent at the very end
        }
    }

    #[test]
    fn tremolo_dips_by_depth() {
        // fs=4, rate=1 -> w = π/2. At n=0 gain is 1; at n=2 cos(π)=-1 so gain = 1-depth.
        let mut c = vec![1.0f32; 4];
        tremolo(&mut c, 1.0, 1.0, 4);
        assert!((c[0] - 1.0).abs() < 1e-6);
        assert!(c[2].abs() < 1e-5); // full dip to 1-depth = 0
    }

    #[test]
    fn overdrive_saturates_and_keeps_silence() {
        let mut c = vec![0.0f32, 10.0, -10.0];
        overdrive(&mut c, 20.0, 0.0);
        assert!(c[0].abs() < 1e-6); // silence stays silent (colour 0)
        assert!(c[1] > 0.99 && c[1] <= 1.0); // hard-driven positive saturates near +1
        assert!(c[2] < -0.99 && c[2] >= -1.0); // and negative near -1
    }

    #[test]
    fn reverse_flips_time() {
        assert_eq!(reverse(&[1.0, 2.0, 3.0]), vec![3.0, 2.0, 1.0]);
    }

    #[test]
    fn gain_vjp_matches_finite_difference() {
        let x = vec![0.5f32, -1.0, 2.0, 0.25];
        let gbar = vec![1.0f32, 0.5, -0.5, 2.0];
        let factor = 1.7f32;
        let (grad_in, grad_factor) = gain_vjp(&x, factor, &gbar);

        // grad_in is exact: factor · gbar.
        for (gi, &g) in grad_in.iter().zip(&gbar) {
            assert!((gi - factor * g).abs() < 1e-6);
        }
        // grad_factor by central difference of f(c) = <gbar, c·x>.
        let dot = |c: f32| x.iter().zip(&gbar).map(|(&xi, &g)| g * c * xi).sum::<f32>();
        let eps = 1e-3;
        let num = (dot(factor + eps) - dot(factor - eps)) / (2.0 * eps);
        assert!((num - grad_factor).abs() < 1e-2 * (1.0 + grad_factor.abs()));
    }

    #[test]
    fn normalize_vjp_matches_finite_difference() {
        // Unique peak (index 2) so the argmax is stable under the FD perturbation.
        let x = vec![0.3f32, -0.5, 2.0, 0.25, -1.1];
        let seed = vec![1.0f32, 0.5, -0.5, 2.0, 0.7];
        let target = 0.8f32;

        let norm = |v: &[f32]| {
            let peak = v.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
            let s = if peak > 0.0 { target / peak } else { 1.0 };
            v.iter().map(|&x| x * s).collect::<Vec<_>>()
        };
        let g = normalize_vjp(&x, target, &seed);

        let loss = |v: &[f32]| norm(v).iter().zip(&seed).map(|(a, b)| a * b).sum::<f32>();
        let eps = 1e-3;
        for j in 0..x.len() {
            let (mut hi, mut lo) = (x.clone(), x.clone());
            hi[j] += eps;
            lo[j] -= eps;
            let fd = (loss(&hi) - loss(&lo)) / (2.0 * eps);
            assert!(
                (g[j] - fd).abs() < 1e-2 * (1.0 + fd.abs()),
                "grad[{j}] = {} vs fd {fd}",
                g[j]
            );
        }
    }

    #[test]
    fn normalize_hits_target_peak() {
        let mut chans = vec![vec![0.0, 0.25, -0.5], vec![0.1, 0.2, 0.0]];
        normalize_peak(&mut chans, 1.0);
        let peak = chans.iter().flatten().fold(0.0f32, |m, &x| m.max(x.abs()));
        assert!((peak - 1.0).abs() < 1e-6);
    }
}

//! Simple amplitude effects and their backward passes.

/// Multiply a channel in place by a linear gain factor.
pub fn gain(channel: &mut [f32], factor: f32) {
    for x in channel {
        *x *= factor;
    }
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
    use super::{gain, gain_vjp, normalize_peak, normalize_vjp};

    #[test]
    fn gain_scales() {
        let mut c = vec![1.0, -2.0, 0.5];
        gain(&mut c, 2.0);
        assert_eq!(c, vec![2.0, -4.0, 1.0]);
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

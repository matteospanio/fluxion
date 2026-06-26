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

#[cfg(test)]
mod tests {
    use super::{gain, gain_vjp, normalize_peak};

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
    fn normalize_hits_target_peak() {
        let mut chans = vec![vec![0.0, 0.25, -0.5], vec![0.1, 0.2, 0.0]];
        normalize_peak(&mut chans, 1.0);
        let peak = chans.iter().flatten().fold(0.0f32, |m, &x| m.max(x.abs()));
        assert!((peak - 1.0).abs() < 1e-6);
    }
}

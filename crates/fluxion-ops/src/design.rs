//! Design-parameter gradients — the DDSP reparameterisation (PROJECT.md §8.2).
//!
//! `sos_vjp`/`biquad_vjp` give ∂L/∂coeffs. This module back-propagates that one step further, to the
//! **design parameters** (`cutoff`, `Q`, `gain`, `order`, `ripple`, …) that produced the
//! coefficients: ∂L/∂params = Jᵀ · ∂L/∂coeffs, where `J = ∂coeffs/∂params` is the Jacobian of a
//! closed-form filter design. Training the design params instead of raw `a1,a2` keeps optimisation
//! on the always-stable design manifold — you can "learn a cutoff" without the filter blowing up.
//!
//! ponytail: `J` is central finite-difference of the design closure. A filter design is a smooth,
//! well-conditioned closed form in 1–4 parameters, so central FD is accurate to ~1e-7 and needs no
//! hand-derived per-formula Jacobians. Upgrade to analytic / dual-number forward-mode only if FD
//! conditioning ever bites (it won't for these designs).

/// Back-propagate a coefficient gradient to the design parameters that produced the coefficients.
///
/// `design(params) -> flat coeffs` is any closed-form design flattened to `[b0,b1,b2,a1,a2]` per
/// section (e.g. `|p| butterworth_lowpass(4, p[0], fs).iter().flat_map(|b| [b.b0,b.b1,b.b2,b.a1,b.a2]).collect()`).
/// `grad_coeffs` is ∂L/∂coeffs in that **same flat layout** (from `sos_vjp`/`biquad_vjp`). Returns
/// ∂L/∂params.
///
/// # Examples
/// ```
/// use fluxion_ops::{butterworth_lowpass, design_param_grad};
/// let fs = 48_000;
/// let flat = |p: &[f32]| butterworth_lowpass(2, p[0], fs)
///     .iter().flat_map(|b| [b.b0, b.b1, b.b2, b.a1, b.a2]).collect::<Vec<_>>();
/// let n = flat(&[1_000.0]).len();
/// let grad_coeffs = vec![1.0f32; n]; // pretend ∂L/∂coeffs = 1
/// let g = design_param_grad(&[1_000.0], &grad_coeffs, flat);
/// assert_eq!(g.len(), 1); // one gradient per design parameter (here, cutoff)
/// ```
pub fn design_param_grad(
    params: &[f32],
    grad_coeffs: &[f32],
    design: impl Fn(&[f32]) -> Vec<f32>,
) -> Vec<f32> {
    let mut grad = vec![0.0f32; params.len()];
    let mut p = params.to_vec();
    for i in 0..params.len() {
        // Floor h at 1e-3: designs return f32-cast coefficients, so a smaller step (e.g. for a
        // parameter near 0, like an EQ gain initialised at 0 dB) pushes the coefficient delta down
        // to f32 rounding noise and the FD gradient's *sign* becomes garbage. Designs are smooth
        // closed forms, so truncation error at h = 1e-3 is negligible.
        let h = (params[i].abs() * 1e-3).max(1e-3);
        p[i] = params[i] + h;
        let plus = design(&p);
        p[i] = params[i] - h;
        let minus = design(&p);
        p[i] = params[i];
        assert_eq!(
            plus.len(),
            grad_coeffs.len(),
            "design output length ({}) must match grad_coeffs ({})",
            plus.len(),
            grad_coeffs.len()
        );
        // gradᵢ = Σ_k (∂coeff_k/∂paramᵢ) · grad_coeff_k, accumulated in f64 for the small step.
        let mut acc = 0.0f64;
        for k in 0..plus.len() {
            let dck = (plus[k] as f64 - minus[k] as f64) / (2.0 * h as f64);
            acc += dck * grad_coeffs[k] as f64;
        }
        grad[i] = acc as f32;
    }
    grad
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Biquad, butterworth_lowpass, sos_filter, sos_vjp};

    const FS: u32 = 48_000;

    fn design_lp(cutoff: f32) -> Vec<f32> {
        butterworth_lowpass(4, cutoff, FS)
            .iter()
            .flat_map(|b| [b.b0, b.b1, b.b2, b.a1, b.a2])
            .collect()
    }
    fn to_sos(flat: &[f32]) -> Vec<Biquad> {
        flat.chunks_exact(5)
            .map(|c| Biquad {
                b0: c[0],
                b1: c[1],
                b2: c[2],
                a1: c[3],
                a2: c[4],
            })
            .collect()
    }
    fn signal(n: usize) -> Vec<f32> {
        (0..n)
            .map(|k| (k as f32 * 0.1).sin() + 0.5 * (k as f32 * 0.031).cos())
            .collect()
    }

    /// The chained gradient (analytic `sos_vjp` + FD design-Jacobian) must match a fully independent
    /// finite-difference of the whole `cutoff → design → filter → loss` pipeline.
    #[test]
    fn design_grad_matches_full_pipeline_fd() {
        let x = signal(1024);
        let target = sos_filter(&x, &to_sos(&design_lp(3_000.0)));
        let cutoff = 2_000.0f32;

        // ∂L/∂coeffs via the analytic VJP, then ∂L/∂cutoff via the design Jacobian.
        let coeffs = design_lp(cutoff);
        let y = sos_filter(&x, &to_sos(&coeffs));
        let grad_out: Vec<f32> = y.iter().zip(&target).map(|(a, b)| a - b).collect();
        let (_, grad_bq) = sos_vjp(&x, &to_sos(&coeffs), &grad_out);
        let grad_coeffs: Vec<f32> = grad_bq
            .iter()
            .flat_map(|g| [g.b0, g.b1, g.b2, g.a1, g.a2])
            .collect();
        let g = design_param_grad(&[cutoff], &grad_coeffs, |p| design_lp(p[0]));

        // Ground truth: FD of L(cutoff) = ½Σ(filter(design(cutoff)) − target)² through everything.
        let loss = |c: f32| -> f64 {
            let yy = sos_filter(&x, &to_sos(&design_lp(c)));
            0.5 * yy
                .iter()
                .zip(&target)
                .map(|(a, b)| ((a - b) as f64).powi(2))
                .sum::<f64>()
        };
        let h = 1.0; // 1 Hz — independent step, through the full pipeline (not just the design)
        let fd = ((loss(cutoff + h) - loss(cutoff - h)) / (2.0 * h as f64)) as f32;

        assert!(
            (g[0] - fd).abs() <= 1e-2 * fd.abs().max(1e-3),
            "design-param grad {} vs full-pipeline FD {}",
            g[0],
            fd
        );
    }

    /// Gradient descent on `cutoff` alone recovers the target filter's cutoff.
    #[test]
    fn gradient_descent_recovers_cutoff() {
        let x = signal(1024);
        let target = sos_filter(&x, &to_sos(&design_lp(3_000.0)));
        // Rprop: step on the gradient's *sign* with an adaptive step size — the Hz-scale gradient
        // magnitude is tiny and design-dependent, so sign-based descent needs no lr tuning.
        let mut cutoff = 1_500.0f32;
        let (mut step, mut prev) = (300.0f32, 0.0f32);
        for _ in 0..80 {
            let coeffs = design_lp(cutoff);
            let y = sos_filter(&x, &to_sos(&coeffs));
            let grad_out: Vec<f32> = y.iter().zip(&target).map(|(a, b)| a - b).collect();
            let (_, grad_bq) = sos_vjp(&x, &to_sos(&coeffs), &grad_out);
            let grad_coeffs: Vec<f32> = grad_bq
                .iter()
                .flat_map(|g| [g.b0, g.b1, g.b2, g.a1, g.a2])
                .collect();
            let g = design_param_grad(&[cutoff], &grad_coeffs, |p| design_lp(p[0]))[0];
            if g == 0.0 {
                break;
            }
            step = if prev * g < 0.0 {
                step * 0.5
            } else {
                step * 1.2
            }
            .clamp(0.1, 1_000.0);
            cutoff = (cutoff - g.signum() * step).clamp(200.0, 20_000.0);
            prev = g;
        }
        assert!(
            (cutoff - 3_000.0).abs() < 50.0,
            "converged cutoff = {cutoff}, want ~3000"
        );
    }
}

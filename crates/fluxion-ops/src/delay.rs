//! Delay-based effects: a single delayed tap ([`delay`]) and a feedback [`echo`].
//!
//! Both are length-preserving (the feedback tail beyond the input is truncated), which matches the
//! realtime engine's chunk-length-preserving contract.
//!
//! ponytail: integer-sample delay for now; fractional (linear-interpolated) delay is a follow-up
//! when pitch/chorus-style modulation needs it.

/// The wet signal of a feedback delay line: `w[n] = x[n-d] + feedback·w[n-d]`.
fn feedback_delay(input: &[f32], d: usize, feedback: f32) -> Vec<f32> {
    let d = d.max(1);
    let mut w = vec![0.0f32; input.len()];
    for n in d..input.len() {
        w[n] = input[n - d] + feedback * w[n - d];
    }
    w
}

/// Single delayed tap crossfaded with the dry signal: `(1-mix)·x[n] + mix·x[n-d]`.
pub fn delay(input: &[f32], delay_samples: usize, mix: f32) -> Vec<f32> {
    let w = feedback_delay(input, delay_samples, 0.0); // w[n] = x[n-d]
    input
        .iter()
        .zip(&w)
        .map(|(&x, &xd)| (1.0 - mix) * x + mix * xd)
        .collect()
}

/// Feedback echo: the dry signal plus `wet`-scaled repeating echoes spaced `delay_samples` apart.
pub fn echo(input: &[f32], delay_samples: usize, feedback: f32, wet: f32) -> Vec<f32> {
    let w = feedback_delay(input, delay_samples, feedback);
    input.iter().zip(&w).map(|(&x, &e)| x + wet * e).collect()
}

/// Analytic backward for [`delay`] (plan task E3). Returns `(grad_input, grad_mix)`.
///
/// `y[n] = (1-mix)·x[n] + mix·x[n-d]` is a 2-tap FIR; its input adjoint sends `g[n]` back to `x[n]`
/// (weight `1-mix`) and to `x[n+d]`'s source `x[n]` (weight `mix`).
pub fn delay_vjp(
    input: &[f32],
    delay_samples: usize,
    mix: f32,
    grad_out: &[f32],
) -> (Vec<f32>, f32) {
    let n = input.len();
    let d = delay_samples.max(1);
    let mut grad_in = vec![0.0f32; n];
    let mut grad_mix = 0.0f32;
    for i in 0..n {
        grad_in[i] += (1.0 - mix) * grad_out[i];
        if let Some(&g) = grad_out.get(i + d) {
            grad_in[i] += mix * g; // y[i+d]'s delayed tap is x[i]
        }
        let xd = if i >= d { input[i - d] } else { 0.0 };
        grad_mix += grad_out[i] * (xd - input[i]);
    }
    (grad_in, grad_mix)
}

/// Analytic backward for [`echo`] (plan task E5). Returns `(grad_input, grad_feedback, grad_wet)`.
///
/// `y = x + wet·w`, `w[n] = x[n-d] + feedback·w[n-d]`. The input gradient flows the cotangent through
/// the **adjoint** (time-reversed) feedback loop; the coefficient gradients use the forward
/// intermediates `w` and `∂w/∂feedback`.
pub fn echo_vjp(
    input: &[f32],
    delay_samples: usize,
    feedback: f32,
    wet: f32,
    grad_out: &[f32],
) -> (Vec<f32>, f32, f32) {
    let n = input.len();
    let d = delay_samples.max(1);
    let w = feedback_delay(input, d, feedback);

    let grad_wet: f32 = grad_out.iter().zip(&w).map(|(&g, &wv)| g * wv).sum();

    // grad_input[n] = g[n] + wet·v[n], where v is the adjoint of x→w: v[n] = g[n+d] + feedback·v[n+d].
    let mut grad_in = vec![0.0f32; n];
    let mut v = vec![0.0f32; n];
    for i in (0..n).rev() {
        let gp = grad_out.get(i + d).copied().unwrap_or(0.0);
        let vp = v.get(i + d).copied().unwrap_or(0.0);
        v[i] = gp + feedback * vp;
        grad_in[i] = grad_out[i] + wet * v[i];
    }

    // grad_feedback = wet·Σ g[n]·u[n], where u = ∂w/∂feedback: u[n] = w[n-d] + feedback·u[n-d].
    let mut u = vec![0.0f32; n];
    let mut grad_fb = 0.0f32;
    for i in 0..n {
        if i >= d {
            u[i] = w[i - d] + feedback * u[i - d];
        }
        grad_fb += grad_out[i] * u[i];
    }
    (grad_in, grad_fb * wet, grad_wet)
}

#[cfg(test)]
mod tests {
    use super::{delay, delay_vjp, echo, echo_vjp};

    fn impulse(n: usize) -> Vec<f32> {
        let mut x = vec![0.0f32; n];
        x[0] = 1.0;
        x
    }

    #[test]
    fn delay_taps_dry_and_delayed() {
        let y = delay(&impulse(16), 4, 0.5);
        assert!((y[0] - 0.5).abs() < 1e-6); // (1-mix)·dry
        assert!((y[4] - 0.5).abs() < 1e-6); // mix·delayed
        assert!(y[1..4].iter().all(|&v| v == 0.0));
        assert_eq!(y.len(), 16); // length-preserving
    }

    #[test]
    fn echo_repeats_with_feedback() {
        let y = echo(&impulse(16), 3, 0.5, 1.0);
        assert!((y[0] - 1.0).abs() < 1e-6); // dry
        assert!((y[3] - 1.0).abs() < 1e-6); // 1st echo
        assert!((y[6] - 0.5).abs() < 1e-6); // 2nd (×feedback)
        assert!((y[9] - 0.25).abs() < 1e-6); // 3rd (×feedback²)
    }

    fn sig(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (0.3 * i as f32).sin() + 0.2 * (0.11 * i as f32).cos())
            .collect()
    }
    fn seed(n: usize) -> Vec<f32> {
        (0..n).map(|i| (0.2 * i as f32 + 0.7).cos()).collect()
    }

    #[test]
    fn delay_vjp_gradchecks() {
        let (x, s, d, mix) = (sig(40), seed(40), 7usize, 0.6f32);
        let (gx, gmix) = delay_vjp(&x, d, mix, &s);
        let dot = |y: &[f32]| y.iter().zip(&s).map(|(a, b)| a * b).sum::<f32>();
        let eps = 1e-3;
        for i in [0usize, 5, 7, 20, 39] {
            let (mut hi, mut lo) = (x.clone(), x.clone());
            hi[i] += eps;
            lo[i] -= eps;
            let fd = (dot(&delay(&hi, d, mix)) - dot(&delay(&lo, d, mix))) / (2.0 * eps);
            assert!((gx[i] - fd).abs() < 1e-2, "grad_x[{i}] {} vs {fd}", gx[i]);
        }
        let fd = (dot(&delay(&x, d, mix + eps)) - dot(&delay(&x, d, mix - eps))) / (2.0 * eps);
        assert!((gmix - fd).abs() < 1e-2, "grad_mix {gmix} vs {fd}");
    }

    #[test]
    fn echo_vjp_gradchecks() {
        let (x, s, d, fb, wet) = (sig(48), seed(48), 5usize, 0.5f32, 0.8f32);
        let (gx, gfb, gwet) = echo_vjp(&x, d, fb, wet, &s);
        let dot = |y: &[f32]| y.iter().zip(&s).map(|(a, b)| a * b).sum::<f32>();
        let eps = 1e-3;
        for i in [0usize, 5, 10, 30, 47] {
            let (mut hi, mut lo) = (x.clone(), x.clone());
            hi[i] += eps;
            lo[i] -= eps;
            let fd = (dot(&echo(&hi, d, fb, wet)) - dot(&echo(&lo, d, fb, wet))) / (2.0 * eps);
            assert!((gx[i] - fd).abs() < 1e-2, "grad_x[{i}] {} vs {fd}", gx[i]);
        }
        let fd_fb =
            (dot(&echo(&x, d, fb + eps, wet)) - dot(&echo(&x, d, fb - eps, wet))) / (2.0 * eps);
        assert!((gfb - fd_fb).abs() < 1e-2, "grad_fb {gfb} vs {fd_fb}");
        let fd_wet =
            (dot(&echo(&x, d, fb, wet + eps)) - dot(&echo(&x, d, fb, wet - eps))) / (2.0 * eps);
        assert!((gwet - fd_wet).abs() < 1e-2, "grad_wet {gwet} vs {fd_wet}");
    }
}

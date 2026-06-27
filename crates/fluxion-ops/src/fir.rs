//! FIR filtering / convolution (plan task D5) and its analytic backward (E2).
//!
//! [`fir_filter`] is a causal, length-preserving FIR: `y[n] = Σ_k h[k]·x[n-k]` (output length =
//! input length, matching the engine's chunk-length-preserving contract). [`fft_convolve`] computes
//! the same thing via the FFT — `O(N log N)` instead of `O(N·M)`, the path for long kernels
//! (convolution reverb). [`fir_vjp`] is the analytic VJP: the input gradient is correlation by `h`
//! (the adjoint of convolution), the tap gradient is correlation of the cotangent with the input.

use rustfft::FftPlanner;
use rustfft::num_complex::Complex;

/// Causal FIR filter: `y[n] = Σ_{k} h[k]·x[n-k]`, `y` the same length as `x` (`x[i]=0` for `i<0`).
pub fn fir_filter(input: &[f32], taps: &[f32]) -> Vec<f32> {
    let n = input.len();
    let mut out = vec![0.0f32; n];
    for (i, o) in out.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for (k, &h) in taps.iter().enumerate().take(i + 1) {
            acc += h * input[i - k];
        }
        *o = acc;
    }
    out
}

/// Same result as [`fir_filter`], via FFT linear convolution (faster for long `taps`).
pub fn fft_convolve(input: &[f32], taps: &[f32]) -> Vec<f32> {
    let n = input.len();
    if n == 0 || taps.is_empty() {
        return vec![0.0f32; n];
    }
    let full = n + taps.len() - 1;
    let len = full.next_power_of_two();

    let mut planner = FftPlanner::new();
    let fwd = planner.plan_fft_forward(len);
    let inv = planner.plan_fft_inverse(len);

    let mut xs: Vec<Complex<f32>> = input
        .iter()
        .map(|&v| Complex::new(v, 0.0))
        .chain(std::iter::repeat(Complex::new(0.0, 0.0)))
        .take(len)
        .collect();
    let mut hs: Vec<Complex<f32>> = taps
        .iter()
        .map(|&v| Complex::new(v, 0.0))
        .chain(std::iter::repeat(Complex::new(0.0, 0.0)))
        .take(len)
        .collect();
    fwd.process(&mut xs);
    fwd.process(&mut hs);
    for (x, h) in xs.iter_mut().zip(&hs) {
        *x *= *h;
    }
    inv.process(&mut xs);

    // rustfft is unnormalized: divide by `len`. Take the causal first `n` samples.
    let scale = 1.0 / len as f32;
    xs[..n].iter().map(|c| c.re * scale).collect()
}

/// Analytic backward for [`fir_filter`]. Returns `(grad_input, grad_taps)` for `grad_out = ∂L/∂y`.
pub fn fir_vjp(input: &[f32], taps: &[f32], grad_out: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let (n, m) = (input.len(), taps.len());

    // ∂L/∂x[j] = Σ_k h[k]·g[j+k]  (adjoint of convolution = correlation by h).
    let mut grad_in = vec![0.0f32; n];
    for (j, gi) in grad_in.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for (k, &h) in taps.iter().enumerate() {
            if let Some(&g) = grad_out.get(j + k) {
                acc += h * g;
            }
        }
        *gi = acc;
    }

    // ∂L/∂h[k] = Σ_i g[i]·x[i-k].
    let mut grad_taps = vec![0.0f32; m];
    for (k, gt) in grad_taps.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for i in k..n {
            acc += grad_out[i] * input[i - k];
        }
        *gt = acc;
    }

    (grad_in, grad_taps)
}

#[cfg(test)]
mod tests {
    use super::{fft_convolve, fir_filter, fir_vjp};

    #[test]
    fn fir_is_causal_convolution() {
        // h = [1, 0.5] on an impulse → [1, 0.5, 0, ...].
        let mut x = vec![0.0f32; 8];
        x[0] = 1.0;
        let y = fir_filter(&x, &[1.0, 0.5]);
        assert!((y[0] - 1.0).abs() < 1e-6 && (y[1] - 0.5).abs() < 1e-6);
        assert!(y[2..].iter().all(|&v| v.abs() < 1e-6));
        assert_eq!(y.len(), 8);
    }

    #[test]
    fn fft_convolve_matches_direct() {
        let x: Vec<f32> = (0..200).map(|i| (0.1 * i as f32).sin()).collect();
        let h: Vec<f32> = (0..31)
            .map(|k| (-(k as f32) / 10.0).exp() * (0.3 * k as f32).cos())
            .collect();
        let direct = fir_filter(&x, &h);
        let viafft = fft_convolve(&x, &h);
        for (a, b) in direct.iter().zip(&viafft) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn fir_vjp_gradchecks() {
        let x: Vec<f32> = (0..40).map(|i| (0.3 * i as f32).sin()).collect();
        let h = vec![0.5f32, -0.3, 0.2, 0.1];
        let seed: Vec<f32> = (0..40).map(|i| (0.2 * i as f32 + 1.0).cos()).collect();
        let (gx, gh) = fir_vjp(&x, &h, &seed);

        let dot = |y: &[f32]| y.iter().zip(&seed).map(|(a, b)| a * b).sum::<f32>();
        let eps = 1e-3;
        for j in [0usize, 7, 20, 39] {
            let (mut hi, mut lo) = (x.clone(), x.clone());
            hi[j] += eps;
            lo[j] -= eps;
            let fd = (dot(&fir_filter(&hi, &h)) - dot(&fir_filter(&lo, &h))) / (2.0 * eps);
            assert!((gx[j] - fd).abs() < 1e-2, "grad_x[{j}] {} vs {fd}", gx[j]);
        }
        for k in 0..h.len() {
            let (mut hi, mut lo) = (h.clone(), h.clone());
            hi[k] += eps;
            lo[k] -= eps;
            let fd = (dot(&fir_filter(&x, &hi)) - dot(&fir_filter(&x, &lo))) / (2.0 * eps);
            assert!((gh[k] - fd).abs() < 1e-2, "grad_h[{k}] {} vs {fd}", gh[k]);
        }
    }
}

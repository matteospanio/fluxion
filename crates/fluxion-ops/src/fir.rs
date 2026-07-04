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

/// [`fir_filter`] with size-aware routing: direct convolution for short kernels,
/// [`overlap_save`] above [`FIR_FFT_THRESHOLD`] taps. Same causal, length-preserving
/// semantics; the FFT path differs from direct summation only by f32 round-off
/// (different, better-conditioned summation order — not bit-identical).
pub fn fir_filter_auto(input: &[f32], taps: &[f32]) -> Vec<f32> {
    if taps.len() >= FIR_FFT_THRESHOLD {
        overlap_save(input, taps)
    } else {
        fir_filter(input, taps)
    }
}

/// Tap count above which the FFT path wins: direct is `O(N·M)`, overlap-save is
/// `O(N log M)`; the crossover on current x86/ARM cores sits near a few dozen taps —
/// 64 keeps a safety margin for the short-kernel cache-friendly direct loop.
pub const FIR_FFT_THRESHOLD: usize = 64;

/// Causal FIR via **overlap-save**: the signal is processed in fixed FFT blocks
/// (`fft_len = max(4·M, 4096)` rounded up to a power of two, `M` = taps), each block
/// carrying `M−1` samples of history, so memory stays `O(fft_len)` regardless of the
/// input length and the small FFTs stay cache-resident — the standard fast path for
/// long kernels (convolution reverb). Output matches [`fir_filter`] (same length,
/// causal) to f32 round-off.
pub fn overlap_save(input: &[f32], taps: &[f32]) -> Vec<f32> {
    let n = input.len();
    let m = taps.len();
    if n == 0 || m == 0 {
        return vec![0.0f32; n];
    }
    let fft_len = (4 * m).max(4096).next_power_of_two();
    let hop = fft_len - (m - 1); // new samples consumed per block

    let mut planner = FftPlanner::new();
    let fwd = planner.plan_fft_forward(fft_len);
    let inv = planner.plan_fft_inverse(fft_len);

    // Kernel spectrum, computed once.
    let mut hs: Vec<Complex<f32>> = taps
        .iter()
        .map(|&v| Complex::new(v, 0.0))
        .chain(std::iter::repeat(Complex::new(0.0, 0.0)))
        .take(fft_len)
        .collect();
    fwd.process(&mut hs);

    let scale = 1.0 / fft_len as f32;
    let mut out = vec![0.0f32; n];
    let mut buf = vec![Complex::new(0.0f32, 0.0); fft_len];

    let mut pos = 0usize; // start of the new samples for this block
    while pos < n {
        // Block layout: [M-1 samples of history | up to `hop` new samples | zero pad].
        for (i, c) in buf.iter_mut().enumerate() {
            let idx = pos as isize - (m as isize - 1) + i as isize;
            let v = if idx >= 0 && (idx as usize) < n {
                input[idx as usize]
            } else {
                0.0
            };
            *c = Complex::new(v, 0.0);
        }
        fwd.process(&mut buf);
        for (x, h) in buf.iter_mut().zip(&hs) {
            *x *= *h;
        }
        inv.process(&mut buf);
        // The first M-1 outputs are circularly aliased; the rest are valid.
        let valid = hop.min(n - pos);
        for (o, c) in out[pos..pos + valid].iter_mut().zip(&buf[m - 1..m - 1 + valid]) {
            *o = c.re * scale;
        }
        pos += hop;
    }
    out
}

/// Analytic backward for [`fir_filter`]. Returns `(grad_input, grad_taps)` for `grad_out = ∂L/∂y`.
pub fn fir_vjp(input: &[f32], taps: &[f32], grad_out: &[f32]) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(
        grad_out.len(),
        input.len(),
        "fir_vjp: grad_out must match the input length"
    );
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
    use super::overlap_save;

    #[test]
    fn overlap_save_matches_direct() {
        // Awkward sizes: kernel longer than a hop, input shorter than the kernel,
        // non-power-of-two everything.
        for (n, m) in [(4_800usize, 2_047usize), (10_000, 64), (100, 300), (4_096, 65)] {
            let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.137).sin()).collect();
            let h: Vec<f32> = (0..m).map(|k| ((k as f32) * 0.03).cos() / m as f32).collect();
            let want = fir_filter(&x, &h);
            let got = overlap_save(&x, &h);
            assert_eq!(got.len(), want.len());
            let peak = want.iter().fold(0.0f32, |a, &v| a.max(v.abs())).max(1e-9);
            for (i, (a, b)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (a - b).abs() <= 1e-4 * peak.max(1.0),
                    "n={n} m={m} i={i}: {a} vs {b}"
                );
            }
        }
    }

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

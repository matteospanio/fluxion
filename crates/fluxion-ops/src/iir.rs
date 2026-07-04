//! IIR filter design (Butterworth, second-order sections), the cascade filter kernel, and its
//! analytic backward pass (VJP).
//!
//! Design is done in `f64` for precision (analog prototype → bilinear transform with frequency
//! pre-warping), then stored as `f32` biquads. No SciPy at runtime — the closed-form Butterworth
//! poles are computed directly.
//!
//! The backward pass is **analytic**, not autograd-unrolling (the differentiable-IIR approach):
//! the input gradient is the adjoint filter (the same biquad run over the time-reversed cotangent),
//! and the coefficient gradients use the all-pole intermediate `1/A(z)` — see [`biquad_vjp`].

use std::f64::consts::PI;

/// A normalized second-order section (`a0 == 1`): `H(z) = (b0 + b1 z⁻¹ + b2 z⁻²)/(1 + a1 z⁻¹ + a2 z⁻²)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Biquad {
    /// Numerator tap for `x[n]`.
    pub b0: f32,
    /// Numerator tap for `x[n-1]`.
    pub b1: f32,
    /// Numerator tap for `x[n-2]`.
    pub b2: f32,
    /// Denominator tap for `y[n-1]`.
    pub a1: f32,
    /// Denominator tap for `y[n-2]`.
    pub a2: f32,
}

impl Biquad {
    /// Magnitude of the frequency response at digital angular frequency `omega` (rad/sample).
    pub fn magnitude(&self, omega: f32) -> f32 {
        let (s1, c1) = omega.sin_cos();
        let (s2, c2) = (2.0 * omega).sin_cos();
        let nr = self.b0 + self.b1 * c1 + self.b2 * c2;
        let ni = -(self.b1 * s1 + self.b2 * s2);
        let dr = 1.0 + self.a1 * c1 + self.a2 * c2;
        let di = -(self.a1 * s1 + self.a2 * s2);
        ((nr * nr + ni * ni) / (dr * dr + di * di)).sqrt()
    }

    /// True if both poles are strictly inside the unit circle (BIBO stable).
    ///
    /// For `A(z) = 1 + a1 z⁻¹ + a2 z⁻²` the Jury condition is `|a2| < 1` and `|a1| < 1 + a2`.
    /// First-order sections (`a2 == 0`) reduce to `|a1| < 1`. For a graded verdict (with a margin,
    /// and tolerance for the f32-cast boundary), use [`crate::stability::certify_biquad`].
    pub fn is_stable(&self) -> bool {
        self.a2.abs() < 1.0 && self.a1.abs() < 1.0 + self.a2
    }

    /// Spectral radius — the largest pole magnitude of the denominator `1 + a1 z⁻¹ + a2 z⁻²`.
    /// The section is BIBO-stable iff this is `< 1`. Computed on the section's stored `f32`
    /// coefficients (the frozen values that actually ship).
    pub fn spectral_radius(&self) -> f32 {
        let disc = self.a1 * self.a1 - 4.0 * self.a2;
        if disc < 0.0 {
            // Complex-conjugate poles: |z|² = product of roots = a2.
            self.a2.abs().sqrt()
        } else {
            let r = disc.sqrt();
            (((-self.a1 + r) / 2.0).abs()).max(((-self.a1 - r) / 2.0).abs())
        }
    }
}

/// A cascade of second-order sections.
pub type Sos = Vec<Biquad>;

/// Cotangents (gradients) for a biquad's five coefficients.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BiquadGrad {
    /// ∂L/∂b0.
    pub b0: f32,
    /// ∂L/∂b1.
    pub b1: f32,
    /// ∂L/∂b2.
    pub b2: f32,
    /// ∂L/∂a1.
    pub a1: f32,
    /// ∂L/∂a2.
    pub a2: f32,
}

/// Design a Butterworth low-pass filter as second-order sections.
pub fn butterworth_lowpass(order: usize, cutoff_hz: f32, fs: u32) -> Sos {
    design(order, cutoff_hz, fs, false)
}

/// Design a Butterworth high-pass filter as second-order sections.
pub fn butterworth_highpass(order: usize, cutoff_hz: f32, fs: u32) -> Sos {
    design(order, cutoff_hz, fs, true)
}

fn design(order: usize, cutoff_hz: f32, fs: u32, highpass: bool) -> Sos {
    let order = order.max(1);
    let fs = fs as f64;
    let fc = cutoff_hz as f64;
    let k = 2.0 * fs; // bilinear constant 2/T
    let wc = k * (PI * fc / fs).tan(); // pre-warped analog cutoff
    let wc2 = wc * wc;

    let mut sos = Sos::with_capacity(order.div_ceil(2));

    // Conjugate pole pairs → second-order sections.
    for i in 0..order / 2 {
        let theta = PI / 2.0 + PI * (2.0 * i as f64 + 1.0) / (2.0 * order as f64);
        let d1 = -2.0 * wc * theta.cos(); // damping term, > 0 (left-half poles)
        let (b, a) = if highpass {
            ([1.0, 0.0, 0.0], [1.0, d1, wc2]) // s² / (s² + d1·s + wc²)
        } else {
            ([0.0, 0.0, wc2], [1.0, d1, wc2]) // wc² / (s² + d1·s + wc²)
        };
        sos.push(bilinear2(b, a, k));
    }

    // Odd order → one real pole, a first-order section.
    if order % 2 == 1 {
        let (b, a) = if highpass {
            ([1.0, 0.0], [1.0, wc]) // s / (s + wc)
        } else {
            ([0.0, wc], [1.0, wc]) // wc / (s + wc)
        };
        sos.push(bilinear1(b, a, k));
    }

    sos
}

/// Bilinear transform of a 2nd-order analog section. `b`/`a` are `[s², s¹, s⁰]` coefficients.
pub(crate) fn bilinear2(b: [f64; 3], a: [f64; 3], k: f64) -> Biquad {
    let k2 = k * k;
    let nb0 = b[0] * k2 + b[1] * k + b[2];
    let nb1 = 2.0 * (b[2] - b[0] * k2);
    let nb2 = b[0] * k2 - b[1] * k + b[2];
    let na0 = a[0] * k2 + a[1] * k + a[2];
    let na1 = 2.0 * (a[2] - a[0] * k2);
    let na2 = a[0] * k2 - a[1] * k + a[2];
    Biquad {
        b0: (nb0 / na0) as f32,
        b1: (nb1 / na0) as f32,
        b2: (nb2 / na0) as f32,
        a1: (na1 / na0) as f32,
        a2: (na2 / na0) as f32,
    }
}

/// Bilinear transform of a 1st-order analog section. `b`/`a` are `[s¹, s⁰]` coefficients.
pub(crate) fn bilinear1(b: [f64; 2], a: [f64; 2], k: f64) -> Biquad {
    let nb0 = b[0] * k + b[1];
    let nb1 = b[1] - b[0] * k;
    let na0 = a[0] * k + a[1];
    let na1 = a[1] - a[0] * k;
    Biquad {
        b0: (nb0 / na0) as f32,
        b1: (nb1 / na0) as f32,
        b2: 0.0,
        a1: (na1 / na0) as f32,
        a2: 0.0,
    }
}

/// Filter one channel through a single biquad (Direct Form II Transposed), zero initial state.
pub fn biquad_forward(input: &[f32], bq: &Biquad) -> Vec<f32> {
    let mut out = vec![0.0f32; input.len()];
    let (mut s1, mut s2) = (0.0f32, 0.0f32);
    for (o, &x) in out.iter_mut().zip(input) {
        let y = bq.b0 * x + s1;
        s1 = bq.b1 * x - bq.a1 * y + s2;
        s2 = bq.b2 * x - bq.a2 * y;
        *o = y;
    }
    out
}

/// Filter one channel through a cascade of sections.
///
/// Fully fused: one pass over the signal, all `K` section states held in registers
/// (monomorphized per `K ≤ 8`; longer cascades run in fused passes of 8). Per-section
/// passes are latency-bound on the ~12-cycle `y→s1→y` dependency chain of each section;
/// fusing the cascade per sample lets the K independent section chains pipeline in the
/// out-of-order core and reads the signal once instead of K times. Sample-for-sample
/// identical to chaining [`biquad_forward`]: every (section, sample) computation keeps
/// the exact same operands and operation order, only the loop nest is interchanged.
pub fn sos_filter(input: &[f32], sos: &[Biquad]) -> Vec<f32> {
    let mut data = input.to_vec();
    sos_filter_in_place(&mut data, sos);
    data
}

/// In-place [`sos_filter`], dispatching on cascade depth for register-resident state.
///
/// Same fused single pass and bit-identical results; use this on an owned buffer to
/// avoid `sos_filter`'s input copy and fresh output allocation.
pub fn sos_filter_in_place(data: &mut [f32], sos: &[Biquad]) {
    match sos.len() {
        0 => {}
        1 => fused_cascade::<1>(data, sos),
        2 => fused_cascade::<2>(data, sos),
        3 => fused_cascade::<3>(data, sos),
        4 => fused_cascade::<4>(data, sos),
        5 => fused_cascade::<5>(data, sos),
        6 => fused_cascade::<6>(data, sos),
        7 => fused_cascade::<7>(data, sos),
        8 => fused_cascade::<8>(data, sos),
        _ => {
            for chunk in sos.chunks(8) {
                sos_filter_in_place(data, chunk);
            }
        }
    }
}

#[inline(always)]
fn fused_cascade<const K: usize>(data: &mut [f32], sos: &[Biquad]) {
    let bq: [Biquad; K] = core::array::from_fn(|k| sos[k]);
    let mut s1 = [0.0f32; K];
    let mut s2 = [0.0f32; K];
    for x in data.iter_mut() {
        let mut v = *x;
        for k in 0..K {
            let y = bq[k].b0 * v + s1[k];
            s1[k] = bq[k].b1 * v - bq[k].a1 * y + s2[k];
            s2[k] = bq[k].b2 * v - bq[k].a2 * y;
            v = y;
        }
        *x = v;
    }
}

/// Apply an SOS cascade to a **channel-interleaved** batch in place: `data[t * channels + c]` is
/// channel `c` at frame `t`.
///
/// An IIR recurrence is sequential in time (each sample needs the previous), so it can't vectorize
/// over time — but the channels are independent. With this frame-major layout the `channels` values
/// at a frame are contiguous, so the inner per-channel loop is data-parallel and auto-vectorizes
/// (SIMD) across the batch. The per-channel result is identical to [`sos_filter`].
///
/// ISA dispatch is at **runtime** (plan task C3): on x86-64 an AVX2-compiled copy of the loop is
/// selected when the CPU supports it, so a baseline (portable) build still runs 8-wide — no
/// `-C target-cpu=native` needed. (AVX2 only, no FMA contraction, so results stay bit-identical to
/// the scalar path. On aarch64, NEON is baseline and the plain build already vectorizes.)
pub fn sos_filter_interleaved(data: &mut [f32], channels: usize, sos: &[Biquad]) {
    let mut state = vec![0.0f32; sos.len() * 2 * channels];
    sos_filter_interleaved_chunk(data, channels, sos, &mut state);
}

/// Like [`sos_filter_interleaved`], but carrying per-(section, channel) filter state across calls —
/// so a long batch can be processed in cache-sized **time chunks** (the strided transposes then stay
/// in a small hot buffer instead of thrashing power-of-two-strided gigabuffers).
///
/// `state` is `[section-major] 2 × channels` Direct-Form-II-Transposed values,
/// `len == sos.len() * 2 * channels`, zero-initialized for a fresh signal. Feeding consecutive
/// chunks with the same `state` is sample-for-sample identical to one whole-signal call.
pub fn sos_filter_interleaved_chunk(
    data: &mut [f32],
    channels: usize,
    sos: &[Biquad],
    state: &mut [f32],
) {
    assert!(channels > 0 && data.len() % channels == 0);
    assert_eq!(
        state.len(),
        sos.len() * 2 * channels,
        "state must be sos.len()*2*channels"
    );
    if data.is_empty() {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 support was just verified at runtime.
        unsafe { interleaved_avx2(data, channels, sos, state) };
        return;
    }
    interleaved_impl(data, channels, sos, state);
}

/// The interleaved cascade loop, monomorphized per ISA by the wrappers below.
///
/// Loop nest is frame-outer, section-inner: the per-section vector dependency chains
/// (`y→s1→y`, ~12 cycles each) are independent across sections, so fusing the cascade
/// per frame lets them pipeline in the out-of-order core instead of paying the full
/// chain latency per section pass — and the block is read once instead of K times.
/// Per-(section, frame, channel) arithmetic is unchanged, so results stay identical
/// to the section-outer formulation.
#[inline(always)]
fn interleaved_impl(data: &mut [f32], channels: usize, sos: &[Biquad], state: &mut [f32]) {
    for frame in data.chunks_exact_mut(channels) {
        for (bq, st) in sos.iter().zip(state.chunks_exact_mut(2 * channels)) {
            let (s1, s2) = st.split_at_mut(channels);
            let (b0, b1, b2, a1, a2) = (bq.b0, bq.b1, bq.b2, bq.a1, bq.a2);
            // Independent across channels → vectorizes. `s1`/`s2` don't alias `data`.
            for ((x, p1), p2) in frame.iter_mut().zip(s1.iter_mut()).zip(s2.iter_mut()) {
                let xi = *x;
                let y = b0 * xi + *p1;
                *p1 = b1 * xi - a1 * y + *p2;
                *p2 = b2 * xi - a2 * y;
                *x = y;
            }
        }
    }
}

/// AVX2 monomorphization of [`interleaved_impl`] — LLVM re-vectorizes the inlined loop 8-wide.
///
/// # Safety
/// The caller must have verified AVX2 support (`is_x86_feature_detected!("avx2")`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn interleaved_avx2(data: &mut [f32], channels: usize, sos: &[Biquad], state: &mut [f32]) {
    interleaved_impl(data, channels, sos, state);
}

/// All-pole filter `w[n] = x[n] - a1·w[n-1] - a2·w[n-2]` (i.e. `1/A(z)`), zero initial state.
fn allpole(input: &[f32], a1: f32, a2: f32) -> Vec<f32> {
    let mut w = vec![0.0f32; input.len()];
    let (mut w1, mut w2) = (0.0f32, 0.0f32);
    for (o, &x) in w.iter_mut().zip(input) {
        let cur = x - a1 * w1 - a2 * w2;
        *o = cur;
        w2 = w1;
        w1 = cur;
    }
    w
}

/// Analytic backward pass for a single biquad.
///
/// Given the section input and the cotangent of its output (`grad_out = ∂L/∂y`), returns the
/// cotangent of the input (`∂L/∂x`) and the gradients of the five coefficients.
///
/// - **Input gradient** uses the adjoint of an LTI filter: convolution by the time-reversed impulse
///   response, computed as `flip(biquad_forward(flip(grad_out)))` — one extra forward pass, no
///   recursion unrolling.
/// - **Coefficient gradients** use `∂H/∂b_i = z⁻ⁱ/A(z)` and `∂H/∂a_i = −z⁻ⁱ·Y(z)/A(z)`, i.e. the
///   all-pole-filtered input `w = 1/A · x` and output `v = 1/A · y`, correlated with `grad_out`.
pub fn biquad_vjp(input: &[f32], bq: &Biquad, grad_out: &[f32]) -> (Vec<f32>, BiquadGrad) {
    // Input gradient: adjoint filter = flip · forward · flip.
    let mut rev = grad_out.to_vec();
    rev.reverse();
    let mut grad_in = biquad_forward(&rev, bq);
    grad_in.reverse();

    // Coefficient gradients.
    let y = biquad_forward(input, bq);
    let w = allpole(input, bq.a1, bq.a2); // ∂y/∂b_i = w[n-i]
    let v = allpole(&y, bq.a1, bq.a2); // ∂y/∂a_i = -v[n-i]
    let mut g = BiquadGrad::default();
    for n in 0..input.len() {
        let gy = grad_out[n];
        g.b0 += gy * w[n];
        if n >= 1 {
            g.b1 += gy * w[n - 1];
            g.a1 -= gy * v[n - 1];
        }
        if n >= 2 {
            g.b2 += gy * w[n - 2];
            g.a2 -= gy * v[n - 2];
        }
    }
    (grad_in, g)
}

/// Analytic backward pass for a whole SOS cascade.
///
/// Returns the input cotangent and a per-section [`BiquadGrad`]. The forward intermediates are
/// recomputed and the sections are traversed in reverse, each via [`biquad_vjp`].
pub fn sos_vjp(input: &[f32], sos: &[Biquad], grad_out: &[f32]) -> (Vec<f32>, Vec<BiquadGrad>) {
    // Forward intermediates: inter[i] is the input to section i.
    let mut inter = Vec::with_capacity(sos.len() + 1);
    inter.push(input.to_vec());
    for bq in sos {
        let next = biquad_forward(inter.last().unwrap(), bq);
        inter.push(next);
    }

    let mut grads = vec![BiquadGrad::default(); sos.len()];
    let mut g = grad_out.to_vec();
    for i in (0..sos.len()).rev() {
        let (grad_in, grad_c) = biquad_vjp(&inter[i], &sos[i], &g);
        grads[i] = grad_c;
        g = grad_in;
    }
    (g, grads)
}

/// Input gradient (VJP) of an SOS cascade — the adjoint filter, sections applied in reverse.
///
/// Equals `sos_vjp(_, sos, grad_out).0` but needs no forward intermediates and computes no
/// coefficient gradients: `Jᵀ = J₁ᵀ … J_mᵀ`, and each section's adjoint is `flip · filter · flip`.
pub fn sos_input_grad(grad_out: &[f32], sos: &[Biquad]) -> Vec<f32> {
    let mut g = grad_out.to_vec();
    for bq in sos.iter().rev() {
        g.reverse();
        g = biquad_forward(&g, bq);
        g.reverse();
    }
    g
}

/// Magnitude response of a whole cascade at angular frequency `omega` (product of sections).
pub fn sos_magnitude(sos: &[Biquad], omega: f32) -> f32 {
    sos.iter().map(|b| b.magnitude(omega)).product()
}

/// `true` if every section of the cascade is stable.
pub fn sos_is_stable(sos: &[Biquad]) -> bool {
    sos.iter().all(Biquad::is_stable)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_1_SQRT_2, PI as PI_F};

    #[test]
    fn interleaved_matches_per_channel_sos_filter() {
        let sos = butterworth_lowpass(8, 4_000.0, 48_000); // 4 sections
        let (channels, frames) = (5usize, 200usize);
        let rows: Vec<Vec<f32>> = (0..channels)
            .map(|c| {
                (0..frames)
                    .map(|t| (0.05 * (t + c * 7) as f32).sin())
                    .collect()
            })
            .collect();

        // Interleave (frame-major), filter, then compare each channel to sos_filter.
        let mut inter = vec![0.0f32; channels * frames];
        for (c, row) in rows.iter().enumerate() {
            for (t, &x) in row.iter().enumerate() {
                inter[t * channels + c] = x;
            }
        }
        sos_filter_interleaved(&mut inter, channels, &sos);

        for (c, row) in rows.iter().enumerate() {
            let want = sos_filter(row, &sos);
            for t in 0..frames {
                assert!((inter[t * channels + c] - want[t]).abs() < 1e-5);
            }
        }
    }

    fn omega(f: f32, fs: u32) -> f32 {
        2.0 * PI_F * f / fs as f32
    }

    // Deterministic pseudo-signals so tests need no RNG dependency.
    fn ramp_sine(n: usize, w: f32, phase: f32) -> Vec<f32> {
        (0..n).map(|i| (w * i as f32 + phase).sin()).collect()
    }

    // Deterministic broadband pseudo-noise in [-1, 1) (LCG) — well-conditioned for least squares.
    fn pseudo_noise(n: usize) -> Vec<f32> {
        let mut s = 0x1234_5678u32;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 9) as f32 / (1u32 << 22) as f32 - 1.0
            })
            .collect()
    }

    #[test]
    fn lowpass_response_shape() {
        let (fs, fc) = (48_000u32, 1_000.0f32);
        for order in [1, 2, 3, 4, 5, 8] {
            let sos = butterworth_lowpass(order, fc, fs);
            assert!(
                (sos_magnitude(&sos, 0.0) - 1.0).abs() < 1e-3,
                "order {order} DC"
            );
            assert!(sos_magnitude(&sos, PI_F) < 1e-2, "order {order} Nyquist");
            let at_fc = sos_magnitude(&sos, omega(fc, fs));
            assert!(
                (at_fc - FRAC_1_SQRT_2).abs() < 2e-2,
                "order {order} fc mag {at_fc}"
            );
        }
    }

    #[test]
    fn highpass_response_shape() {
        let (fs, fc) = (48_000u32, 1_000.0f32);
        for order in [1, 2, 3, 4, 5, 8] {
            let sos = butterworth_highpass(order, fc, fs);
            assert!(sos_magnitude(&sos, 0.0) < 1e-2, "order {order} DC");
            assert!(
                (sos_magnitude(&sos, PI_F) - 1.0).abs() < 1e-3,
                "order {order} Nyquist"
            );
            let at_fc = sos_magnitude(&sos, omega(fc, fs));
            assert!(
                (at_fc - FRAC_1_SQRT_2).abs() < 2e-2,
                "order {order} fc mag {at_fc}"
            );
        }
    }

    #[test]
    fn lowpass_passes_dc_in_time_domain() {
        let sos = butterworth_lowpass(4, 1_000.0, 48_000);
        let x = vec![1.0f32; 2_000];
        let y = sos_filter(&x, &sos);
        assert!((y[1_999] - 1.0).abs() < 1e-3, "settled DC = {}", y[1_999]);
    }

    #[test]
    fn butterworth_is_stable() {
        for order in [1, 2, 3, 4, 5, 8] {
            assert!(sos_is_stable(&butterworth_lowpass(order, 1_000.0, 48_000)));
            assert!(sos_is_stable(&butterworth_highpass(order, 1_000.0, 48_000)));
        }
        // An obviously unstable section is rejected.
        assert!(
            !Biquad {
                b0: 1.0,
                b1: 0.0,
                b2: 0.0,
                a1: 0.0,
                a2: 1.5
            }
            .is_stable()
        );
    }

    #[test]
    fn biquad_vjp_matches_finite_difference() {
        let x = ramp_sine(64, 0.3, 0.0);
        let gbar = ramp_sine(64, 0.17, 1.0); // arbitrary cotangent
        let bq = butterworth_lowpass(2, 6_000.0, 48_000)[0];

        let (grad_in, grad_c) = biquad_vjp(&x, &bq, &gbar);

        // Scalar objective f(θ) = <gbar, forward_θ(x)>; its derivatives must match the VJP.
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(p, q)| p * q).sum::<f32>();
        let eps = 1e-2;
        let perturbed = |mutate: &dyn Fn(&mut Biquad, f32)| {
            let mut hi = bq;
            mutate(&mut hi, eps);
            let mut lo = bq;
            mutate(&mut lo, -eps);
            (dot(&gbar, &biquad_forward(&x, &hi)) - dot(&gbar, &biquad_forward(&x, &lo)))
                / (2.0 * eps)
        };
        let close = |num: f32, ana: f32, what: &str| {
            assert!(
                (num - ana).abs() < 1e-2 * (1.0 + ana.abs()),
                "{what}: num {num} vs ana {ana}"
            );
        };
        close(perturbed(&|q, d| q.b0 += d), grad_c.b0, "b0");
        close(perturbed(&|q, d| q.b1 += d), grad_c.b1, "b1");
        close(perturbed(&|q, d| q.b2 += d), grad_c.b2, "b2");
        close(perturbed(&|q, d| q.a1 += d), grad_c.a1, "a1");
        close(perturbed(&|q, d| q.a2 += d), grad_c.a2, "a2");

        // Input gradient at a few indices.
        for j in [0usize, 7, 31, 63] {
            let mut hi = x.clone();
            hi[j] += eps;
            let mut lo = x.clone();
            lo[j] -= eps;
            let num = (dot(&gbar, &biquad_forward(&hi, &bq))
                - dot(&gbar, &biquad_forward(&lo, &bq)))
                / (2.0 * eps);
            close(num, grad_in[j], "grad_in");
        }
    }

    #[test]
    fn sos_vjp_chains_through_sections() {
        let x = ramp_sine(80, 0.2, 0.5);
        let gbar = ramp_sine(80, 0.31, 0.0);
        let sos = butterworth_lowpass(4, 5_000.0, 48_000); // two sections

        let (grad_in, grads) = sos_vjp(&x, &sos, &gbar);
        assert_eq!(grads.len(), sos.len());

        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(p, q)| p * q).sum::<f32>();
        let eps = 1e-2;
        for j in [0usize, 13, 79] {
            let mut hi = x.clone();
            hi[j] += eps;
            let mut lo = x.clone();
            lo[j] -= eps;
            let num = (dot(&gbar, &sos_filter(&hi, &sos)) - dot(&gbar, &sos_filter(&lo, &sos)))
                / (2.0 * eps);
            assert!(
                (num - grad_in[j]).abs() < 1e-2 * (1.0 + grad_in[j].abs()),
                "grad_in[{j}]: num {num} vs ana {}",
                grad_in[j]
            );
        }
    }

    #[test]
    fn sos_input_grad_matches_sos_vjp() {
        let x = ramp_sine(80, 0.2, 0.5);
        let gbar = ramp_sine(80, 0.31, 0.0);
        let sos = butterworth_lowpass(4, 5_000.0, 48_000);
        let full = sos_vjp(&x, &sos, &gbar).0; // grad_in from the full VJP
        let light = sos_input_grad(&gbar, &sos);
        for (a, b) in full.iter().zip(&light) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn gradient_descent_fits_numerator_coeffs() {
        // Fit b-coefficients (fixed stable denominator) to match a target: a convex least-squares
        // problem, so GD with the analytic gradient converges. Broadband input keeps the regressors
        // well-conditioned; the step size is derived from the data so the test needs no tuning.
        let x = pseudo_noise(256);
        let denom = butterworth_lowpass(2, 4_000.0, 48_000)[0];
        let target = Biquad {
            b0: 0.5,
            b1: 0.3,
            b2: -0.2,
            ..denom
        };
        let target_y = biquad_forward(&x, &target);

        // λmax(Hessian) ≤ trace = energies of w and its two shifts ≤ 3·‖w‖²  ⇒  lr = 1/(3‖w‖²) is safe.
        let w = allpole(&x, denom.a1, denom.a2);
        let lr = 1.0 / (3.0 * w.iter().map(|v| v * v).sum::<f32>());

        let mut bq = Biquad {
            b0: 0.0,
            b1: 0.0,
            b2: 0.0,
            ..denom
        };
        let loss = |bq: &Biquad| {
            biquad_forward(&x, bq)
                .iter()
                .zip(&target_y)
                .map(|(y, t)| (y - t) * (y - t))
                .sum::<f32>()
        };
        let initial = loss(&bq);
        for _ in 0..5_000 {
            let y = biquad_forward(&x, &bq);
            let resid: Vec<f32> = y.iter().zip(&target_y).map(|(y, t)| y - t).collect();
            let (_, g) = biquad_vjp(&x, &bq, &resid); // grad of 0.5·‖resid‖² wrt coeffs
            bq.b0 -= lr * g.b0;
            bq.b1 -= lr * g.b1;
            bq.b2 -= lr * g.b2;
        }
        let final_loss = loss(&bq);
        assert!(
            final_loss < initial * 1e-3,
            "did not converge: {initial} -> {final_loss}"
        );
        // Coefficients recovered.
        for (got, want) in [(bq.b0, 0.5), (bq.b1, 0.3), (bq.b2, -0.2)] {
            assert!((got - want).abs() < 1e-2, "coeff {got} vs {want}");
        }
    }
}

//! IIR filter design (Butterworth, second-order sections) and the cascade filter kernel.
//!
//! Design is done in `f64` for precision (analog prototype → bilinear transform with frequency
//! pre-warping), then stored as `f32` biquads. No SciPy at runtime — the closed-form Butterworth
//! poles are computed directly.

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
}

/// A cascade of second-order sections.
pub type Sos = Vec<Biquad>;

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
fn bilinear2(b: [f64; 3], a: [f64; 3], k: f64) -> Biquad {
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
fn bilinear1(b: [f64; 2], a: [f64; 2], k: f64) -> Biquad {
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

/// Filter a single channel through a cascade of sections (Direct Form II Transposed).
///
/// State starts at zero and is local to this call (offline, whole-buffer processing).
pub fn sos_filter(input: &[f32], sos: &[Biquad]) -> Vec<f32> {
    let mut data = input.to_vec();
    for bq in sos {
        let (mut s1, mut s2) = (0.0f32, 0.0f32);
        for x in data.iter_mut() {
            let xin = *x;
            let y = bq.b0 * xin + s1;
            s1 = bq.b1 * xin - bq.a1 * y + s2;
            s2 = bq.b2 * xin - bq.a2 * y;
            *x = y;
        }
    }
    data
}

/// Magnitude response of a whole cascade at angular frequency `omega` (product of sections).
pub fn sos_magnitude(sos: &[Biquad], omega: f32) -> f32 {
    sos.iter().map(|b| b.magnitude(omega)).product()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_1_SQRT_2, PI as PI_F};

    fn omega(f: f32, fs: u32) -> f32 {
        2.0 * PI_F * f / fs as f32
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
}

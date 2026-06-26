//! Chebyshev Type I IIR filter design (second-order sections).
//!
//! Type I has equiripple in the passband and a maximally-flat (monotonic) stopband. The analog
//! prototype poles lie on an ellipse (`sinh`/`cosh` of `asinh(1/ε)/N`); they are placed directly as
//! real-coefficient sections — no complex arithmetic — then bilinear-transformed with the same
//! frequency pre-warping as the Butterworth path. High-pass uses the `s → wc/s` transform (poles
//! reciprocated, zeros at the origin).
//!
//! ponytail: Type II (inverse Chebyshev — stopband ripple, finite transmission zeros) is not yet
//! implemented; it needs zero placement + a different gain normalization and is a separate task.

use std::f64::consts::PI;

use crate::iir::{Sos, bilinear1, bilinear2};

/// Design a Chebyshev Type I low-pass with `ripple_db` of passband ripple.
pub fn chebyshev1_lowpass(order: usize, cutoff_hz: f32, ripple_db: f32, fs: u32) -> Sos {
    design1(order, cutoff_hz, ripple_db, fs, false)
}

/// Design a Chebyshev Type I high-pass with `ripple_db` of passband ripple.
pub fn chebyshev1_highpass(order: usize, cutoff_hz: f32, ripple_db: f32, fs: u32) -> Sos {
    design1(order, cutoff_hz, ripple_db, fs, true)
}

fn design1(order: usize, cutoff_hz: f32, ripple_db: f32, fs: u32, highpass: bool) -> Sos {
    let order = order.max(1);
    let fs = fs as f64;
    let fc = cutoff_hz as f64;
    let k = 2.0 * fs;
    let wc = k * (PI * fc / fs).tan();

    let eps = (10f64.powf(ripple_db as f64 / 10.0) - 1.0).sqrt();
    let v0 = (1.0 / eps).asinh() / order as f64;
    let (sh, ch) = (v0.sinh(), v0.cosh());
    let g_even = 1.0 / (1.0 + eps * eps).sqrt(); // passband ripple minimum (= 10^(-Rp/20))

    let mut sos = Sos::with_capacity(order.div_ceil(2));

    for i in 0..order / 2 {
        let theta = PI * (2.0 * i as f64 + 1.0) / (2.0 * order as f64);
        let sp = -sh * theta.sin(); // prototype pole real part (cutoff ω=1)
        let op = ch * theta.cos(); // prototype pole imag part
        let mag2 = sp * sp + op * op;
        let bq = if highpass {
            // High-pass pole = wc / proto_pole: Re = wc·sp/|p|², |p_hp|² = wc²/|p|²; zeros at 0.
            let re = wc * sp / mag2;
            let p2 = wc * wc / mag2;
            bilinear2([1.0, 0.0, 0.0], [1.0, -2.0 * re, p2], k)
        } else {
            let (sg, om) = (wc * sp, wc * op);
            let d0 = sg * sg + om * om;
            bilinear2([0.0, 0.0, d0], [1.0, -2.0 * sg, d0], k)
        };
        sos.push(bq);
    }

    // Odd order → the real prototype pole at -sinh(v0).
    if order % 2 == 1 {
        let bq = if highpass {
            bilinear1([1.0, 0.0], [1.0, wc / sh], k) // s / (s + wc/sh)
        } else {
            bilinear1([0.0, wc * sh], [1.0, wc * sh], k) // wc·sh / (s + wc·sh)
        };
        sos.push(bq);
    }

    // Even order → the ripple extremum is the minimum; scale the cascade to it.
    if order % 2 == 0
        && let Some(first) = sos.first_mut()
    {
        let g = g_even as f32;
        first.b0 *= g;
        first.b1 *= g;
        first.b2 *= g;
    }

    sos
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sos_magnitude;
    use std::f32::consts::PI as PI_F;

    const FS: u32 = 48_000;
    const FC: f32 = 2_000.0;

    fn w(f: f32) -> f32 {
        2.0 * PI_F * f / FS as f32
    }

    #[test]
    fn lowpass_passband_ripple_and_cutoff() {
        for order in [2, 3, 4, 5, 6] {
            let rp = 1.0f32;
            let sos = chebyshev1_lowpass(order, FC, rp, FS);
            let g_even = 10f32.powf(-rp / 20.0); // ~0.891

            // At the cutoff the Chebyshev-I magnitude is exactly 10^(-Rp/20), for any order.
            let at_fc = sos_magnitude(&sos, w(FC));
            assert!(
                (at_fc - g_even).abs() < 3e-2,
                "order {order}: fc mag {at_fc}"
            );

            // Passband stays within the ripple band [g_even, 1].
            for f in [10.0, 200.0, 800.0, 1_500.0] {
                let m = sos_magnitude(&sos, w(f));
                assert!(m <= 1.02 && m >= g_even - 0.05, "order {order} f={f}: {m}");
            }
            // Deep stopband is well below the passband.
            assert!(
                sos_magnitude(&sos, w(8_000.0)) < 0.2,
                "order {order} stopband"
            );
        }
    }

    #[test]
    fn highpass_mirrors_lowpass() {
        let rp = 1.0f32;
        let g_even = 10f32.powf(-rp / 20.0);
        for order in [2, 3, 4, 5] {
            let sos = chebyshev1_highpass(order, FC, rp, FS);
            assert!(
                (sos_magnitude(&sos, w(FC)) - g_even).abs() < 3e-2,
                "order {order} fc"
            );
            assert!(
                sos_magnitude(&sos, w(200.0)) < 0.2,
                "order {order} stopband (DC side)"
            );
            assert!(
                sos_magnitude(&sos, w(16_000.0)) <= 1.02,
                "order {order} passband"
            );
        }
    }
}

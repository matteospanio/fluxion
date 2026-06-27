//! Chebyshev Type I IIR filter design (second-order sections).
//!
//! Type I has equiripple in the passband and a maximally-flat (monotonic) stopband. The analog
//! prototype poles lie on an ellipse (`sinh`/`cosh` of `asinh(1/ε)/N`); they are placed directly as
//! real-coefficient sections — no complex arithmetic — then bilinear-transformed with the same
//! frequency pre-warping as the Butterworth path. High-pass uses the `s → wc/s` transform (poles
//! reciprocated, zeros at the origin).
//!
//! Type II (inverse Chebyshev) has a maximally-flat passband and equiripple stopband with finite
//! transmission zeros. Its poles are the reciprocals of the Type I poles; its zeros sit on the jω
//! axis at `±j/cos(θ_k)`. Each conjugate pole+zero pair forms one analog biquad
//! `(s² + β²)/(s² − 2·Re(p)·s + |p|²)`, normalized to unit DC (low-pass) / unit Nyquist (high-pass)
//! gain, then bilinear-transformed — again with real coefficients only.

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

/// Design a Chebyshev Type II low-pass with `stop_db` of stopband attenuation. `cutoff_hz` is the
/// **stopband edge** (the frequency at which the attenuation reaches `stop_db`); the passband below
/// it is maximally flat.
pub fn chebyshev2_lowpass(order: usize, cutoff_hz: f32, stop_db: f32, fs: u32) -> Sos {
    design2(order, cutoff_hz, stop_db, fs, false)
}

/// Design a Chebyshev Type II high-pass with `stop_db` of stopband attenuation (`cutoff_hz` = the
/// stopband edge).
pub fn chebyshev2_highpass(order: usize, cutoff_hz: f32, stop_db: f32, fs: u32) -> Sos {
    design2(order, cutoff_hz, stop_db, fs, true)
}

fn design2(order: usize, cutoff_hz: f32, stop_db: f32, fs: u32, highpass: bool) -> Sos {
    let order = order.max(1);
    let fs = fs as f64;
    let fc = cutoff_hz as f64;
    let k = 2.0 * fs;
    let wc = k * (PI * fc / fs).tan(); // pre-warped stopband-edge frequency

    let rs = (stop_db as f64).max(0.1);
    let de = 1.0 / (10f64.powf(rs / 10.0) - 1.0).sqrt();
    let a = (1.0 / de).asinh() / order as f64;
    let (sh, ch) = (a.sinh(), a.cosh());

    let mut sos = Sos::with_capacity(order.div_ceil(2));

    for i in 0..order / 2 {
        let theta = PI * (2.0 * i as f64 + 1.0) / (2.0 * order as f64);
        // Type I prototype pole, then reciprocate for Type II.
        let sigma = -sh * theta.sin();
        let omega = ch * theta.cos();
        let mag2 = sigma * sigma + omega * omega;
        let re = sigma / mag2; // Re(Type II prototype pole)  (< 0, left half plane)
        let p2 = 1.0 / mag2; // |Type II prototype pole|²
        let beta2 = 1.0 / (theta.cos() * theta.cos()); // transmission zero at ±j·√beta2

        let bq = if highpass {
            // lp→hp (s → wc/s) on the unit-DC-normalized prototype; unit gain at Nyquist.
            bilinear2(
                [1.0, 0.0, wc * wc / beta2],
                [1.0, -2.0 * re * wc / p2, wc * wc / p2],
                k,
            )
        } else {
            // scaled to the stopband edge wc; unit DC gain.
            bilinear2(
                [p2 / beta2, 0.0, p2 * wc * wc],
                [1.0, -2.0 * re * wc, p2 * wc * wc],
                k,
            )
        };
        sos.push(bq);
    }

    // Odd order → the real Type II pole at -1/sinh(a); no finite zero (it sits at ∞). lp→lp keeps
    // the pole at wc/sinh(a); lp→hp inverts it to wc·sinh(a).
    if order % 2 == 1 {
        let bq = if highpass {
            let wp = wc * sh;
            bilinear1([1.0, 0.0], [1.0, wp], k) // s / (s + wp)
        } else {
            let wp = wc / sh;
            bilinear1([0.0, wp], [1.0, wp], k) // wp / (s + wp)
        };
        sos.push(bq);
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
    fn cheby2_lowpass_stopband_and_passband() {
        for order in [2, 3, 4, 5, 6] {
            let rs = 40.0f32;
            let sos = chebyshev2_lowpass(order, FC, rs, FS);
            let stop = 10f32.powf(-rs / 20.0); // 0.01 (-40 dB)

            // At the stopband edge (= cutoff) the magnitude is the stopband level.
            let at_fc = sos_magnitude(&sos, w(FC));
            assert!(
                (at_fc - stop).abs() < 0.5 * stop + 2e-3,
                "order {order}: fc {at_fc} vs {stop}"
            );
            // Near DC the passband is flat at unity (the transition band is wide at low order).
            for f in [10.0, 100.0] {
                let m = sos_magnitude(&sos, w(f));
                assert!((0.98..=1.02).contains(&m), "order {order} pass f={f}: {m}");
            }
            // Stopband above the edge stays at/below the stopband level (equiripple).
            for f in [4_000.0, 8_000.0, 16_000.0] {
                let m = sos_magnitude(&sos, w(f));
                assert!(m <= stop * 1.2, "order {order} stop f={f}: {m}");
            }
        }
    }

    #[test]
    fn cheby2_highpass_mirrors() {
        let rs = 40.0f32;
        let stop = 10f32.powf(-rs / 20.0);
        for order in [2, 3, 4, 5] {
            let sos = chebyshev2_highpass(order, FC, rs, FS);
            assert!(
                (sos_magnitude(&sos, w(FC)) - stop).abs() < 0.5 * stop + 2e-3,
                "order {order} fc"
            );
            // Passband near Nyquist is flat at unity (wide transition at low order).
            for f in [20_000.0, 23_000.0] {
                let m = sos_magnitude(&sos, w(f));
                assert!((0.98..=1.02).contains(&m), "order {order} pass f={f}: {m}");
            }
            // Stopband below the edge stays low.
            for f in [100.0, 500.0, 1_500.0] {
                assert!(
                    sos_magnitude(&sos, w(f)) <= stop * 1.2,
                    "order {order} stop f={f}"
                );
            }
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

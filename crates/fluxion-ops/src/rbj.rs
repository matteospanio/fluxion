//! RBJ Audio-EQ-Cookbook biquads: peaking, low/high shelf, notch, bandpass, allpass.
//!
//! Each returns a single normalized [`Biquad`] (a one-section cascade). The formulas are the
//! canonical Robert Bristow-Johnson cookbook equations, computed in `f64`. Because these are plain
//! biquads, [`crate::biquad_vjp`] already differentiates them w.r.t. their coefficients and input —
//! no per-op backward is needed.

use std::f64::consts::PI;

use crate::iir::Biquad;

/// Shared intermediates: `(cos w0, sin w0, alpha)` with `alpha = sin(w0)/(2Q)`.
fn pre(freq: f32, q: f32, fs: u32) -> (f64, f64, f64) {
    let w0 = 2.0 * PI * freq as f64 / fs as f64;
    let (sw, cw) = w0.sin_cos();
    let alpha = sw / (2.0 * q as f64);
    (cw, sw, alpha)
}

/// Normalize an unnormalized biquad by `a0` and store as `f32`.
fn norm(b0: f64, b1: f64, b2: f64, a0: f64, a1: f64, a2: f64) -> Biquad {
    Biquad {
        b0: (b0 / a0) as f32,
        b1: (b1 / a0) as f32,
        b2: (b2 / a0) as f32,
        a1: (a1 / a0) as f32,
        a2: (a2 / a0) as f32,
    }
}

/// Peaking EQ: boost/cut of `gain_db` around `freq` with bandwidth set by `q`.
pub fn peaking(freq: f32, gain_db: f32, q: f32, fs: u32) -> Biquad {
    let (cw, _, alpha) = pre(freq, q, fs);
    let a = 10f64.powf(gain_db as f64 / 40.0);
    norm(
        1.0 + alpha * a,
        -2.0 * cw,
        1.0 - alpha * a,
        1.0 + alpha / a,
        -2.0 * cw,
        1.0 - alpha / a,
    )
}

/// Low shelf: `gain_db` applied below `freq`, unity above.
pub fn low_shelf(freq: f32, gain_db: f32, q: f32, fs: u32) -> Biquad {
    let (cw, _, alpha) = pre(freq, q, fs);
    let a = 10f64.powf(gain_db as f64 / 40.0);
    let t = 2.0 * a.sqrt() * alpha;
    norm(
        a * ((a + 1.0) - (a - 1.0) * cw + t),
        2.0 * a * ((a - 1.0) - (a + 1.0) * cw),
        a * ((a + 1.0) - (a - 1.0) * cw - t),
        (a + 1.0) + (a - 1.0) * cw + t,
        -2.0 * ((a - 1.0) + (a + 1.0) * cw),
        (a + 1.0) + (a - 1.0) * cw - t,
    )
}

/// High shelf: `gain_db` applied above `freq`, unity below.
pub fn high_shelf(freq: f32, gain_db: f32, q: f32, fs: u32) -> Biquad {
    let (cw, _, alpha) = pre(freq, q, fs);
    let a = 10f64.powf(gain_db as f64 / 40.0);
    let t = 2.0 * a.sqrt() * alpha;
    norm(
        a * ((a + 1.0) + (a - 1.0) * cw + t),
        -2.0 * a * ((a - 1.0) + (a + 1.0) * cw),
        a * ((a + 1.0) + (a - 1.0) * cw - t),
        (a + 1.0) - (a - 1.0) * cw + t,
        2.0 * ((a - 1.0) - (a + 1.0) * cw),
        (a + 1.0) - (a - 1.0) * cw - t,
    )
}

/// Notch: deep null at `freq`, unity elsewhere.
pub fn notch(freq: f32, q: f32, fs: u32) -> Biquad {
    let (cw, _, alpha) = pre(freq, q, fs);
    norm(1.0, -2.0 * cw, 1.0, 1.0 + alpha, -2.0 * cw, 1.0 - alpha)
}

/// Band-pass with 0 dB peak gain at `freq`.
pub fn bandpass(freq: f32, q: f32, fs: u32) -> Biquad {
    let (cw, _, alpha) = pre(freq, q, fs);
    norm(alpha, 0.0, -alpha, 1.0 + alpha, -2.0 * cw, 1.0 - alpha)
}

/// All-pass: flat magnitude, frequency-dependent phase shift around `freq`.
pub fn allpass(freq: f32, q: f32, fs: u32) -> Biquad {
    let (cw, _, alpha) = pre(freq, q, fs);
    norm(
        1.0 - alpha,
        -2.0 * cw,
        1.0 + alpha,
        1.0 + alpha,
        -2.0 * cw,
        1.0 - alpha,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI as PI_F;

    const FS: u32 = 48_000;
    const F0: f32 = 1_000.0;

    fn w(f: f32) -> f32 {
        2.0 * PI_F * f / FS as f32
    }

    fn linear(db: f32) -> f32 {
        10f32.powf(db / 20.0)
    }

    #[test]
    fn peaking_boosts_at_center() {
        let bq = peaking(F0, 6.0, 1.0, FS);
        assert!((bq.magnitude(w(F0)) - linear(6.0)).abs() < 1e-2);
        assert!((bq.magnitude(0.0) - 1.0).abs() < 1e-2); // unity at DC
        assert!(bq.is_stable());
    }

    #[test]
    fn notch_nulls_at_center() {
        let bq = notch(F0, 5.0, FS);
        assert!(bq.magnitude(w(F0)) < 1e-3);
        assert!((bq.magnitude(0.0) - 1.0).abs() < 1e-2);
    }

    #[test]
    fn shelves_hit_target_gain() {
        let lo = low_shelf(F0, 6.0, 0.707, FS);
        assert!((lo.magnitude(0.0) - linear(6.0)).abs() < 2e-2); // boosted at DC
        assert!((lo.magnitude(PI_F) - 1.0).abs() < 2e-2); // unity at Nyquist

        let hi = high_shelf(F0, 6.0, 0.707, FS);
        assert!((hi.magnitude(PI_F) - linear(6.0)).abs() < 2e-2); // boosted at Nyquist
        assert!((hi.magnitude(0.0) - 1.0).abs() < 2e-2); // unity at DC
    }

    #[test]
    fn bandpass_peaks_at_center_zero_at_dc() {
        let bq = bandpass(F0, 1.0, FS);
        assert!((bq.magnitude(w(F0)) - 1.0).abs() < 1e-2);
        assert!(bq.magnitude(0.0) < 1e-2);
    }

    #[test]
    fn allpass_is_flat() {
        let bq = allpass(F0, 0.707, FS);
        for f in [50.0, F0, 5_000.0, 18_000.0] {
            assert!((bq.magnitude(w(f)) - 1.0).abs() < 1e-3, "f={f}");
        }
        assert!(bq.is_stable());
    }
}

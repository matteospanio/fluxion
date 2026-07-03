//! Golden-vector oracle tests (plan task D12): pin fluxion's filter designs against SciPy and the
//! RBJ Audio-EQ-Cookbook as independent references.
//!
//! The reference vectors live in the generated `golden_data.rs` (produced by
//! `scripts/gen_golden.py`, which needs SciPy) — this test itself has **no** Python/SciPy
//! dependency; it replays fluxion's own design + `sos_filter` on a unit impulse and compares the
//! impulse response to the embedded golden vector.
//!
//! ## Why impulse responses, not raw coefficients
//! An SOS cascade's transfer function is invariant to the ordering and the pole/zero *pairing* of
//! its sections, but SciPy and fluxion legitimately order/pair sections differently (SciPy sorts by
//! pole proximity to the unit circle; fluxion emits prototype-pole order). Raw coefficient rows
//! therefore need not match section-for-section. The **impulse response** is a property of the whole
//! cascade, so it is the invariant we assert on across every family. (The reference `sos` rows are
//! embedded for documentation only.)
//!
//! ## Convention notes (fluxion vs. the reference)
//! - Butterworth / Chebyshev: fluxion and SciPy share the bilinear-transform-with-prewarping design,
//!   so the digital filters coincide to f64 precision; the only gap is fluxion's f32 coefficient
//!   cast + f32 filtering vs. SciPy's f64.
//! - Chebyshev II `cutoff_hz` is the **stopband edge** (where attenuation reaches `stop_db`), matching
//!   SciPy's `cheby2` `Wn` semantics.
//! - RBJ biquads: fluxion uses the cookbook Q-form (`alpha = sin(w0)/(2Q)`); `bandpass` is the
//!   constant-0-dB-peak variant. `gen_golden.py` evaluates the identical cookbook formulas as the
//!   oracle (SciPy has no RBJ designer).
//!
//! ## Tolerance
//! Comparison is the standard mixed form `|a-b| <= ATOL + RTOL*|b|` (numpy `allclose` semantics).
//! `RTOL` is sized for f32 accumulation over 64 samples through cascades whose poles sit close to
//! the unit circle (the narrowband high-order cases are the stress test); `ATOL` floors it for the
//! near-zero leading samples. If a design genuinely disagreed beyond this it would be a real bug,
//! reported rather than papered over.

#[path = "golden_data.rs"]
mod golden_data;

use fluxion_ops::{
    Biquad, allpass, bandpass, butterworth_highpass, butterworth_lowpass, chebyshev1_highpass,
    chebyshev1_lowpass, chebyshev2_highpass, chebyshev2_lowpass, high_shelf, low_shelf, notch,
    peaking, sos_filter,
};
use golden_data::CASES;

const FS: u32 = 48_000;
const IR_LEN: usize = 64;

/// Mixed absolute/relative tolerance for the f32 impulse response vs. the f64 reference.
const RTOL: f32 = 1e-4;
const ATOL: f32 = 1e-6;

/// The fluxion side of the oracle: build each design with the same name `gen_golden.py` used. The
/// name is the join key; every golden case must match exactly one entry here and vice-versa.
fn fluxion_cases() -> Vec<(String, Vec<Biquad>)> {
    let mut v: Vec<(String, Vec<Biquad>)> = Vec::new();

    for &order in &[2usize, 4, 6] {
        for &fc in &[200u32, 1000, 8000] {
            v.push((
                format!("butter_lp_o{order}_fc{fc}"),
                butterworth_lowpass(order, fc as f32, FS),
            ));
            v.push((
                format!("butter_hp_o{order}_fc{fc}"),
                butterworth_highpass(order, fc as f32, FS),
            ));
        }
    }

    for &fc in &[1000u32, 8000] {
        v.push((
            format!("cheby1_lp_o4_rp1_fc{fc}"),
            chebyshev1_lowpass(4, fc as f32, 1.0, FS),
        ));
        v.push((
            format!("cheby1_hp_o4_rp1_fc{fc}"),
            chebyshev1_highpass(4, fc as f32, 1.0, FS),
        ));
    }

    for &fc in &[2000u32, 8000] {
        v.push((
            format!("cheby2_lp_o4_rs40_fc{fc}"),
            chebyshev2_lowpass(4, fc as f32, 40.0, FS),
        ));
        v.push((
            format!("cheby2_hp_o4_rs40_fc{fc}"),
            chebyshev2_highpass(4, fc as f32, 40.0, FS),
        ));
    }

    v.push((
        "rbj_peaking_f1000_g6_q1".into(),
        vec![peaking(1000.0, 6.0, 1.0, FS)],
    ));
    v.push((
        "rbj_lowshelf_f1000_g6_q0707".into(),
        vec![low_shelf(1000.0, 6.0, 0.707, FS)],
    ));
    v.push((
        "rbj_highshelf_f1000_g6_q0707".into(),
        vec![high_shelf(1000.0, 6.0, 0.707, FS)],
    ));
    v.push(("rbj_notch_f1000_q5".into(), vec![notch(1000.0, 5.0, FS)]));
    v.push((
        "rbj_bandpass_f1000_q1".into(),
        vec![bandpass(1000.0, 1.0, FS)],
    ));
    v.push((
        "rbj_allpass_f1000_q0707".into(),
        vec![allpass(1000.0, 0.707, FS)],
    ));

    v
}

/// Unit-impulse response of `sos` over `IR_LEN` samples — the same signal `gen_golden.py` feeds
/// `scipy.signal.sosfilt`.
fn impulse_response(sos: &[Biquad]) -> Vec<f32> {
    let mut x = vec![0.0f32; IR_LEN];
    x[0] = 1.0;
    sos_filter(&x, sos)
}

#[test]
fn golden_designs_match_reference_impulse_responses() {
    let fluxion = fluxion_cases();

    // The name sets must be identical: catches a matrix drift between this file and the generator.
    assert_eq!(
        fluxion.len(),
        CASES.len(),
        "case-count drift: fluxion {} vs golden {}",
        fluxion.len(),
        CASES.len()
    );

    let mut worst_ratio = 0.0f32; // worst diff / tolerance across every sample (< 1.0 ⇒ pass)
    let mut worst_case = "";

    for case in CASES {
        let (_, sos) = fluxion
            .iter()
            .find(|(name, _)| name == case.name)
            .unwrap_or_else(|| panic!("golden case {} has no fluxion design", case.name));

        let got = impulse_response(sos);
        assert_eq!(got.len(), case.ir.len());

        for (n, (&a, &b)) in got.iter().zip(case.ir.iter()).enumerate() {
            let tol = ATOL + RTOL * b.abs();
            let diff = (a - b).abs();
            let ratio = diff / tol;
            if ratio > worst_ratio {
                worst_ratio = ratio;
                worst_case = case.name;
            }
            assert!(
                diff <= tol,
                "{}: sample {n} fluxion {a} vs golden {b} (|Δ|={diff}, tol={tol})",
                case.name,
            );
        }
    }

    // A passing-margin readout (visible with `--nocapture`) so the tolerance can be tightened
    // deliberately rather than guessed.
    eprintln!(
        "golden oracle: {} cases, worst margin {:.3} of tolerance in `{}`",
        CASES.len(),
        worst_ratio,
        worst_case
    );
    assert!(worst_ratio <= 1.0);
}

//! Stability certification for filters and feedback loops.
//!
//! This upgrades the old per-section Jury check ([`crate::Biquad::is_stable`]) into a graded
//! **verdict ladder** with a numerical margin, plus a loop-aware **small-gain** certificate for
//! feedback graphs (the `~`/FDN case a per-section pole check cannot cover).
//!
//! Two principles, both lessons from the differentiable-audio deployment literature:
//! - Certify the **frozen `f32` coefficients** (the values that actually ship after the
//!   `f64`-design → `f32`-cast), not the pristine design — a lossless prototype can read back as
//!   `spectral_radius = 1.0000002`, which must be `marginally-stable`, not `unstable`.
//! - For feedback loops, stability is a property of the **whole loop** (small gain), not of any
//!   single section.

use std::f32::consts::PI;
use std::fmt;

use crate::iir::Biquad;

/// Tolerance band around the unit circle / unity loop gain for a "marginal" verdict — wide enough
/// to absorb `f32` design+cast noise, narrow enough to still flag a genuine instability.
const MARGIN_EPS: f32 = 1e-4;

/// Stability verdict, from best to worst: `certified-stable`, `marginally-stable`, `indeterminate`,
/// `not-certified`, `unstable`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Strictly inside the stable region with a comfortable margin.
    CertifiedStable,
    /// On / within `f32`-tolerance of the boundary (e.g. a lossless prototype).
    MarginallyStable,
    /// Could not be evaluated (non-finite coefficients).
    Indeterminate,
    /// No certificate is available for this construct.
    NotCertified,
    /// Provably outside the stable region.
    Unstable,
}

impl Verdict {
    fn severity(self) -> u8 {
        match self {
            Verdict::Unstable => 0,
            Verdict::Indeterminate => 1,
            Verdict::NotCertified => 2,
            Verdict::MarginallyStable => 3,
            Verdict::CertifiedStable => 4,
        }
    }

    /// Whether an artifact with this verdict is safe to ship (certified or marginal).
    pub fn is_shippable(self) -> bool {
        matches!(self, Verdict::CertifiedStable | Verdict::MarginallyStable)
    }
}

impl fmt::Display for Verdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Verdict::CertifiedStable => "certified-stable",
            Verdict::MarginallyStable => "marginally-stable",
            Verdict::Indeterminate => "indeterminate",
            Verdict::NotCertified => "not-certified",
            Verdict::Unstable => "unstable",
        })
    }
}

/// A stability verdict plus its numerical margin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Certificate {
    /// The verdict.
    pub verdict: Verdict,
    /// `1 - spectral_radius` (poles) or `1 - sup loop gain` (small-gain): positive inside, ~0 on the
    /// boundary, negative outside; `NaN` if indeterminate.
    pub margin: f32,
}

impl Certificate {
    /// A trivially-stable certificate (e.g. for pass-through / feedforward nodes).
    pub fn certified() -> Self {
        Self {
            verdict: Verdict::CertifiedStable,
            margin: 1.0,
        }
    }

    /// A verdict from a spectral radius / sup loop gain `r` (the stable region is `r < 1`).
    fn from_radius(r: f32) -> Self {
        if !r.is_finite() {
            return Self {
                verdict: Verdict::Indeterminate,
                margin: f32::NAN,
            };
        }
        let verdict = if r <= 1.0 - MARGIN_EPS {
            Verdict::CertifiedStable
        } else if r <= 1.0 + MARGIN_EPS {
            Verdict::MarginallyStable
        } else {
            Verdict::Unstable
        };
        Self {
            verdict,
            margin: 1.0 - r,
        }
    }

    /// The more severe of two certificates (used to aggregate a cascade or graph): the worse
    /// verdict wins, and on a tie the smaller (tighter) margin wins so the aggregate reflects the
    /// least-stable element.
    pub fn worst(self, other: Self) -> Self {
        let (s, o) = (self.verdict.severity(), other.verdict.severity());
        if o < s || (o == s && other.margin < self.margin) {
            other
        } else {
            self
        }
    }
}

impl fmt::Display for Certificate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (margin {:.2e})", self.verdict, self.margin)
    }
}

/// Certify a single biquad from its pole locations (on the frozen `f32` coefficients).
pub fn certify_biquad(bq: &Biquad) -> Certificate {
    Certificate::from_radius(bq.spectral_radius())
}

/// Certify an SOS cascade — the worst section verdict.
pub fn certify_sos(sos: &[Biquad]) -> Certificate {
    sos.iter().fold(Certificate::certified(), |acc, bq| {
        acc.worst(certify_biquad(bq))
    })
}

/// Small-gain certificate for a feedback loop: the loop is stable if its gain stays below 1 at
/// every frequency. `loop_gain(omega)` returns the loop magnitude at normalized frequency
/// `omega ∈ [0, π]`; the certificate is the supremum over an `n+1`-point grid.
pub fn small_gain_certify(loop_gain: impl Fn(f32) -> f32, n: usize) -> Certificate {
    let n = n.max(1);
    let mut sup = 0.0f32;
    for k in 0..=n {
        let g = loop_gain(PI * k as f32 / n as f32);
        if !g.is_finite() {
            return Certificate {
                verdict: Verdict::Indeterminate,
                margin: f32::NAN,
            };
        }
        sup = sup.max(g);
    }
    Certificate::from_radius(sup)
}

/// Project a biquad's denominator `(a1, a2)` into the **Jury stability triangle** so every pole sits
/// strictly inside the unit circle; the numerator is left untouched.
///
/// This is the *in-loop* stability projection for training **raw** SOS coefficients — free `a1,a2`
/// can drift outside the unit circle mid-optimization and the filter blows up. Apply it after each
/// optimizer step. (Training design parameters instead — `cutoff`/`Q` via
/// [`crate::design_param_grad`] — stays stable by construction and needs no projection.)
///
/// The strict-stability region for `1 + a1·z⁻¹ + a2·z⁻²` is `|a2| < 1` and `|a1| < 1 + a2`.
/// `margin ∈ [0, 1)` shrinks it to hold a safety buffer off the boundary.
///
/// ponytail: a coordinate clamp into the (shrunk) triangle, not the nearest-point Euclidean
/// projection — it guarantees stability and keeps a training loop bounded, which is all it's for.
/// Upgrade to the true projection only if minimal-displacement matters for convergence.
///
/// # Examples
/// ```
/// use fluxion_ops::{project_stable, Biquad};
/// let unstable = Biquad { b0: 1.0, b1: 0.0, b2: 0.0, a1: 0.0, a2: 1.5 }; // poles at |z| ≈ 1.22
/// let stable = project_stable(unstable, 1e-2);
/// assert!(stable.spectral_radius() < 1.0);
/// ```
pub fn project_stable(bq: Biquad, margin: f32) -> Biquad {
    let m = margin.clamp(0.0, 0.999);
    let a2 = bq.a2.clamp(-(1.0 - m), 1.0 - m);
    let lim = (1.0 + a2 - m).max(0.0); // |a1| < 1 + a2
    let a1 = bq.a1.clamp(-lim, lim);
    Biquad { a1, a2, ..bq }
}

/// Project every biquad's `(a1, a2)` in a flat `[b0,b1,b2,a1,a2]·n` coefficient vector into the Jury
/// stability triangle in place ([`project_stable`]) — the convenient in-loop call for a trained SOS
/// coefficient tensor.
pub fn project_stable_flat(coeffs: &mut [f32], margin: f32) {
    for c in coeffs.chunks_exact_mut(5) {
        let p = project_stable(
            Biquad {
                b0: c[0],
                b1: c[1],
                b2: c[2],
                a1: c[3],
                a2: c[4],
            },
            margin,
        );
        c[3] = p.a1;
        c[4] = p.a2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{butterworth_lowpass, sos_filter};

    fn bq(a1: f32, a2: f32) -> Biquad {
        Biquad {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1,
            a2,
        }
    }

    #[test]
    fn stable_filter_is_certified() {
        let c = certify_sos(&butterworth_lowpass(6, 1_000.0, 48_000));
        assert_eq!(c.verdict, Verdict::CertifiedStable);
        assert!(c.margin > 0.0);
    }

    #[test]
    fn poles_on_unit_circle_are_marginal() {
        // a1=0, a2=1 -> poles at ±j, |z| = 1.
        assert_eq!(
            certify_biquad(&bq(0.0, 1.0)).verdict,
            Verdict::MarginallyStable
        );
    }

    #[test]
    fn f32_cast_just_over_unity_is_marginal_not_unstable() {
        // The lossless-prototype lesson: 1.0000002 after the f32 cast must not read as unstable.
        assert_eq!(
            certify_biquad(&bq(0.0, 1.000_000_2)).verdict,
            Verdict::MarginallyStable
        );
    }

    #[test]
    fn clearly_unstable_is_detected() {
        let c = certify_biquad(&bq(0.0, 1.5));
        assert_eq!(c.verdict, Verdict::Unstable);
        assert!(c.margin < 0.0);
    }

    #[test]
    fn non_finite_is_indeterminate() {
        assert_eq!(
            certify_biquad(&bq(f32::NAN, 0.5)).verdict,
            Verdict::Indeterminate
        );
    }

    #[test]
    fn small_gain_loop_verdicts() {
        assert_eq!(
            small_gain_certify(|_| 0.5, 16).verdict,
            Verdict::CertifiedStable
        );
        assert_eq!(
            small_gain_certify(|_| 1.0, 16).verdict,
            Verdict::MarginallyStable
        );
        assert_eq!(small_gain_certify(|_| 1.5, 16).verdict, Verdict::Unstable);
    }

    #[test]
    fn worst_aggregates_severity() {
        let stable = Certificate::certified();
        let unstable = certify_biquad(&bq(0.0, 1.5));
        assert_eq!(stable.worst(unstable).verdict, Verdict::Unstable);
    }

    #[test]
    fn projection_lands_in_the_stable_region() {
        // Every (a1, a2) — stable or not — projects to a strictly-stable, shippable biquad.
        for &a1 in &[-3.0, -1.9, -1.0, 0.0, 1.0, 1.9, 3.0] {
            for &a2 in &[-2.0, -0.9, 0.0, 0.9, 1.5, 2.0] {
                let p = project_stable(bq(a1, a2), 1e-2);
                let r = p.spectral_radius();
                assert!(r < 1.0, "a1={a1} a2={a2} -> radius {r}");
                assert!(certify_biquad(&p).verdict.is_shippable());
            }
        }
    }

    #[test]
    fn projection_preserves_a_stable_interior_and_the_numerator() {
        let inside = Biquad {
            b0: 0.5,
            b1: -0.2,
            b2: 0.1,
            a1: 0.3,
            a2: 0.2,
        };
        let p = project_stable(inside, 1e-2);
        assert_eq!((p.a1, p.a2), (inside.a1, inside.a2)); // well inside → unchanged
        assert_eq!((p.b0, p.b1, p.b2), (inside.b0, inside.b1, inside.b2)); // numerator untouched
    }

    #[test]
    fn projection_keeps_the_impulse_response_bounded() {
        // An unstable denominator blows the impulse response up; projection tames it to bounded.
        let mut impulse = vec![0.0f32; 256];
        impulse[0] = 1.0;

        let unstable = bq(0.0, 1.5); // |z| ≈ 1.22 → 1.5^128 overflows f32
        assert_eq!(certify_biquad(&unstable).verdict, Verdict::Unstable);
        let blown = sos_filter(&impulse, &[unstable]);
        assert!(blown.iter().any(|&x| !x.is_finite() || x.abs() > 1e3));

        let tamed = sos_filter(&impulse, &[project_stable(unstable, 1e-2)]);
        assert!(tamed.iter().all(|&x| x.is_finite() && x.abs() <= 1.01));
    }
}

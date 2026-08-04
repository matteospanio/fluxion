//! The envelope follower against SciPy (ROADMAP S2).
//!
//! With attack and release equal the follower is a plain one-pole, which is something `lfilter`
//! can state exactly — so this is a sample-by-sample comparison rather than a statistical one, and
//! the tolerance is float noise rather than a judgement call.
//!
//! The asymmetric case, which is the one a gate actually uses, has no LTI reference to compare to;
//! it is checked against the closed-form attack and release curves in `follower.rs`.
//!
//! Regenerate after changing the follower or the case set:
//!
//! ```text
//! python scripts/gen_follower_golden.py
//! ```

mod follower_golden_data;

use fluxion_ops::follower::{Detector, envelope};
use follower_golden_data::{FOLLOWER_CASES, FRAMES, FS, STRIDE};

/// Must match `noise()` in `scripts/gen_follower_golden.py`.
fn lcg(n: usize) -> Vec<f32> {
    let mut state: u32 = 0x1234_5678;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / 16_777_216.0 * 2.0 - 1.0) * 0.5
        })
        .collect()
}

/// Two one-poles running the same recurrence in the same order, one in f64 and one in f32. What
/// separates them is f32 rounding accumulating over 4800 samples, and nothing else — worst measured
/// is 1.2e-6. The bound is an order of magnitude above that, which is still far below any
/// difference a wrong convention would make: `1 - exp(-2.2/(t·fs))` instead of `exp(-1/(t·fs))`
/// misses by 0.2, and smoothing `|x|` instead of `x²` for RMS misses by 0.05.
const TOLERANCE: f32 = 1e-5;

#[test]
fn the_follower_is_scipys_one_pole() {
    assert!(FOLLOWER_CASES.len() >= 4, "the oracle set shrank");
    let x = lcg(FRAMES);

    let mut worst = 0.0f32;
    for case in FOLLOWER_CASES {
        let detector = match case.detector {
            "peak" => Detector::Peak,
            "rms" => Detector::Rms,
            other => panic!("case '{}': unknown detector '{other}'", case.name),
        };
        let ours = envelope(&x, case.seconds, case.seconds, detector, FS);
        assert_eq!(ours.len(), x.len());

        for (i, want) in case.expected.iter().enumerate() {
            let got = ours[i * STRIDE];
            let off = (got - want).abs();
            worst = worst.max(off);
            assert!(
                off < TOLERANCE,
                "case '{}': frame {} is {got}, SciPy says {want}",
                case.name,
                i * STRIDE
            );
        }
    }
    println!("worst disagreement with SciPy: {worst:e}");
}

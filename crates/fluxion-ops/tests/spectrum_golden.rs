//! The spectrum tap against SciPy (ROADMAP A2).
//!
//! A known multitone, four combinations of FFT size and overlap, compared bin by bin against an
//! independently written Hann-windowed rfft. What matters most is the **scaling**: a spectrum that
//! is merely proportional to the right one looks convincing until somebody reads a number off it,
//! so a 0.4-amplitude partial has to come back as 0.4 whatever the size and overlap are.
//!
//! Regenerate after changing the tap or the case set:
//!
//! ```text
//! python scripts/gen_spectrum_golden.py
//! ```

mod spectrum_golden_data;

use fluxion_core::{TapData, TapKind};
use fluxion_ops::analysis::measure;
use spectrum_golden_data::{FS, SECONDS, SPECTRUM_CASES, STRIDE, TONES};

/// Must match `signal()` in `scripts/gen_spectrum_golden.py`.
fn multitone() -> Vec<f32> {
    let n = (SECONDS * FS as f32) as usize;
    (0..n)
        .map(|i| {
            TONES
                .iter()
                .map(|(freq, amp)| {
                    (std::f64::consts::TAU * f64::from(*freq) * i as f64 / f64::from(FS)).sin()
                        * f64::from(*amp)
                })
                .sum::<f64>() as f32
        })
        .collect()
}

/// Two FFTs of the same windowed samples, one in f64 and one in f32, averaged over up to 187
/// frames. What separates them is f32 rounding — worst measured is 1.8e-7 against partials of
/// amplitude 0.1 to 0.4, so this is set well above the noise and still far below
/// anything a scaling mistake would produce (the usual ones are out by 2x, by the window's coherent
/// gain of 0.5, or by the FFT size).
const TOLERANCE: f32 = 5e-6;

#[test]
fn the_spectrum_tap_is_scipys_spectrum() {
    assert!(SPECTRUM_CASES.len() >= 4, "the oracle set shrank");
    let x = vec![multitone()];

    let mut worst = 0.0f32;
    for case in SPECTRUM_CASES {
        let kind = TapKind::Spectrum {
            size: case.size,
            overlap: case.overlap,
        };
        let TapData::Spectrum { bin_hz, magnitude } = measure(&x, kind, FS) else {
            panic!("case '{}': expected a spectrum", case.name);
        };
        assert_eq!(bin_hz, FS as f32 / case.size as f32);
        assert_eq!(magnitude.len(), case.size / 2 + 1);

        for (i, want) in case.expected.iter().enumerate() {
            let got = magnitude[i * STRIDE];
            let off = (got - want).abs();
            worst = worst.max(off);
            assert!(
                off < TOLERANCE,
                "case '{}': bin {} is {got}, SciPy says {want}",
                case.name,
                i * STRIDE
            );
        }

        // And the thing the whole scaling exists for: each partial reads its own amplitude.
        for (freq, amp) in TONES {
            let bin = (freq / bin_hz).round() as usize;
            assert!(
                (magnitude[bin] - amp).abs() < 0.005,
                "case '{}': the {freq} Hz partial reads {}, not {amp}",
                case.name,
                magnitude[bin]
            );
        }
    }
    println!("worst disagreement with SciPy: {worst:e}");
}

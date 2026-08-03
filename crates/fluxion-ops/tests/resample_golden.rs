//! Our streaming resampler against `scipy.signal.resample_poly` (ROADMAP R1).
//!
//! Two different filter designs — a Kaiser-windowed polyphase FIR against our Blackman-windowed
//! sinc — so this is not a bit-comparison. What is compared is the **spectrum**: whether the two
//! converters agree about what the signal contains, band by band.
//!
//! Comparing samples was the first attempt and it measured the wrong thing. Two converters can
//! both be right and still differ by a fraction of an output sample in delay; at 15 kHz a sixth of
//! a sample is already a third of full scale, so a sample-wise tolerance wide enough to pass would
//! have been far too wide to catch a real fault. A spectrum is immune to delay and sensitive to
//! exactly what distinguishes two filter designs.
//!
//! Regenerate after changing the resampler or the signal set:
//!
//! ```text
//! python scripts/gen_resample_golden.py
//! ```

mod resample_golden_data;

use fluxion_ops::resample::{Quality, Resampler};
use resample_golden_data::{
    ALIAS_REJECTION_DB, BAND_HI, BAND_LO, BANDS, KEEP, RESAMPLE_CASES, ResampleCase, SECONDS,
    SKIP_OUT,
};
use rustfft::{FftPlanner, num_complex::Complex};

const FROM_FS: u32 = 48_000;
const TO_FS: u32 = 44_100;

/// How far apart two independently designed converters may be, per band, in dB.
///
/// Measured rather than chosen: the worst band across this set is printed by the test. The
/// difference is dominated by the two windows' stopband shapes at the very top of the compared
/// range, not by either converter being wrong.
const TOLERANCE_DB: f32 = 1.0;

/// How far below a case's loudest band still counts as signal. Below it, the two are describing
/// their own windows' leakage, where ours reads 26 dB quieter than `resample_poly` on a pure tone —
/// a difference that says nothing about conversion.
const SIGNAL_RANGE_DB: f32 = 60.0;

fn stream(input: &[f32], block: usize) -> Vec<f32> {
    let mut r = Resampler::new(FROM_FS, TO_FS, Quality::Hq, block);
    let mut out = Vec::with_capacity(r.max_output(input.len()));
    let mut scratch = vec![0.0f32; r.max_output(block)];
    for chunk in input.chunks(block) {
        let n = r.process(chunk, &mut scratch);
        out.extend_from_slice(&scratch[..n]);
    }
    out
}

/// Magnitude spectrum in dB, averaged into `BANDS` geometric bins. Must match `spectrum()` in
/// `scripts/gen_resample_golden.py`.
fn spectrum(x: &[f32]) -> Vec<f32> {
    let n = x.len();
    let mut buffer: Vec<Complex<f32>> = x
        .iter()
        .enumerate()
        .map(|(i, v)| {
            // numpy's `hanning` is the symmetric window, divisor n-1.
            let w = 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / (n - 1) as f32).cos();
            Complex::new(v * w, 0.0)
        })
        .collect();
    FftPlanner::new().plan_fft_forward(n).process(&mut buffer);

    let bin_hz = TO_FS as f32 / n as f32;
    let ratio = (BAND_HI / BAND_LO).powf(1.0 / BANDS as f32);
    (0..BANDS)
        .map(|band| {
            let lo = BAND_LO * ratio.powi(band as i32);
            let hi = BAND_LO * ratio.powi(band as i32 + 1);
            let (mut sum, mut count) = (0.0f32, 0usize);
            for (bin, value) in buffer.iter().take(n / 2 + 1).enumerate() {
                let f = bin as f32 * bin_hz;
                if f >= lo && f < hi {
                    sum += value.norm_sqr();
                    count += 1;
                }
            }
            let rms = if count > 0 {
                (sum / count as f32).sqrt()
            } else {
                0.0
            };
            20.0 * rms.max(1e-9).log10()
        })
        .collect()
}

/// Our filter delay in output frames. `resample_poly` compensates its own, so ours has to come off
/// before the two windows describe the same slice of the signal.
///
/// It does not matter for a steady tone, but it matters a great deal for a sweep: analysing 8192
/// frames starting 32 frames later is analysing a different band, and the spectra then differ by
/// two decibels for reasons that have nothing to do with either converter.
fn delay_out_frames() -> usize {
    let r = Resampler::new(FROM_FS, TO_FS, Quality::Hq, 256);
    (r.latency() as f64 * f64::from(TO_FS) / f64::from(FROM_FS)).round() as usize
}

fn lcg(n: usize) -> Vec<f32> {
    let mut state: u32 = 0x1234_5678;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / 16_777_216.0 * 2.0 - 1.0
        })
        .collect()
}

/// Rebuild one case's signal. Must match `signal()` in `scripts/gen_resample_golden.py`.
fn signal(case: &ResampleCase) -> Vec<f32> {
    let n = (SECONDS * FROM_FS as f32) as usize;
    let fs = f64::from(FROM_FS);
    match case.kind {
        "tone" => (0..n)
            .map(|i| (std::f64::consts::TAU * f64::from(case.freq) * i as f64 / fs).sin() as f32)
            .collect(),
        "sweep" => {
            let (f0, f1, secs) = (20.0f64, 15_000.0f64, f64::from(SECONDS));
            (0..n)
                .map(|i| {
                    let t = i as f64 / fs;
                    (std::f64::consts::TAU * (f0 * t + 0.5 * (f1 - f0) / secs * t * t)).sin() as f32
                })
                .collect()
        }
        "noise" => {
            let sos = fluxion_ops::iir::butterworth_lowpass(8, 15_000.0, FROM_FS);
            fluxion_ops::sos_filter(&lcg(n), &sos)
                .into_iter()
                .map(|v| v * 0.5)
                .collect()
        }
        other => panic!("case '{}': unknown signal kind '{other}'", case.name),
    }
}

/// Pre-condition: the generated spectra are what `resample_poly` produced.
/// Post-condition: ours agree band by band, within `TOLERANCE_DB`.
#[test]
fn matches_scipy_resample_poly() {
    assert!(RESAMPLE_CASES.len() >= 4, "the oracle set shrank");

    let mut worst_overall = 0.0f32;
    let mut worst_where = String::new();
    for case in RESAMPLE_CASES {
        let ours = stream(&signal(case), 256);
        let start = SKIP_OUT + delay_out_frames();
        assert!(
            ours.len() >= start + KEEP,
            "case '{}': only {} output frames, need {}",
            case.name,
            ours.len(),
            start + KEEP
        );
        let mine = spectrum(&ours[start..start + KEEP]);
        assert_eq!(mine.len(), case.expected_db.len());

        // Where the reference has signal, ours must match. Where it does not, ours must not be
        // *louder* — being quieter there is a converter with a cleaner floor, which is the
        // direction nobody needs protecting from.
        let reference_peak = case.expected_db.iter().copied().fold(f32::MIN, f32::max);
        for (band, (a, b)) in mine.iter().zip(case.expected_db).enumerate() {
            let floor = reference_peak - SIGNAL_RANGE_DB;
            if *b < floor {
                // The reference has nothing here, so the only fault worth reporting is us putting
                // something *audible* there. Comparing two noise floors to each other would be
                // comparing two windows' leakage, which is not a property of either converter.
                assert!(
                    *a <= floor + TOLERANCE_DB,
                    "case '{}', band {band}: we put {a:.2} dB where resample_poly has \
                     {b:.2} dB of nothing (floor {floor:.2})",
                    case.name
                );
                continue;
            }
            let error = (a - b).abs();
            assert!(
                error <= TOLERANCE_DB,
                "case '{}', band {band}: {a:.2} dB vs resample_poly {b:.2} dB \
                 — off by {error:.2} dB",
                case.name
            );
            if error > worst_overall {
                worst_overall = error;
                worst_where = format!("{} band {band}", case.name);
            }
        }
    }
    println!(
        "resample oracle: {} cases x {BANDS} bands, worst difference {worst_overall:.2} dB \
         of {TOLERANCE_DB} in `{worst_where}`",
        RESAMPLE_CASES.len()
    );
}

/// The failure that matters: content above the output Nyquist has nowhere to fold but into the
/// band. Ours must reject a 23 kHz tone at least as well as the reference does.
#[test]
fn rejects_out_of_band_content_at_least_as_well_as_scipy() {
    let n = (SECONDS * FROM_FS as f32) as usize;
    let tone: Vec<f32> = (0..n)
        .map(|i| (std::f64::consts::TAU * 23_000.0 * i as f64 / f64::from(FROM_FS)).sin() as f32)
        .collect();

    let out = stream(&tone, 256);
    let steady = &out[SKIP_OUT..out.len() - SKIP_OUT];
    let rms = (steady.iter().map(|v| v * v).sum::<f32>() / steady.len() as f32).sqrt();
    let rejection_db = 20.0 * (rms.max(1e-12) / std::f32::consts::FRAC_1_SQRT_2).log10();

    assert!(
        rejection_db <= ALIAS_REJECTION_DB,
        "we reject a 23 kHz tone by {rejection_db:.1} dB; resample_poly manages \
         {ALIAS_REJECTION_DB:.1} dB"
    );
    println!(
        "alias rejection at 23 kHz: {rejection_db:.1} dB (resample_poly {ALIAS_REJECTION_DB:.1} dB)"
    );
}

/// `Fast` trades stopband for speed, so it is allowed to be worse — but it still has to be a
/// resampler, not a decoration. Its passband must be flat where it matters.
#[test]
fn fast_quality_still_has_a_flat_passband() {
    for freq in [100.0, 1000.0, 5000.0] {
        let n = (SECONDS * FROM_FS as f32) as usize;
        let tone: Vec<f32> = (0..n)
            .map(|i| (std::f64::consts::TAU * freq * i as f64 / f64::from(FROM_FS)).sin() as f32)
            .collect();

        let mut r = Resampler::new(FROM_FS, TO_FS, Quality::Fast, 256);
        let mut out = Vec::new();
        let mut scratch = vec![0.0f32; r.max_output(256)];
        for chunk in tone.chunks(256) {
            let k = r.process(chunk, &mut scratch);
            out.extend_from_slice(&scratch[..k]);
        }

        let steady = &out[SKIP_OUT..out.len() - SKIP_OUT];
        let peak = steady.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(
            (peak - 1.0).abs() < 0.02,
            "Fast at {freq} Hz: peak {peak:.4}, expected 1.0"
        );
    }
}

//! Our time-stretch against Rubber Band (ROADMAP R3).
//!
//! Rubber Band, through `ffmpeg -af rubberband`, is the independent reference. It is a different
//! algorithm, so this is not a sample comparison — what is compared is the **spectrum**, band by
//! band, which is what a listener hears and the one thing two stretchers must agree about.
//!
//! The ground truth is neither converter: it is the **source**. A stretcher changes how long the
//! material lasts and nothing else, so the spectrum of its output should be the spectrum of its
//! input. Both are scored against that, and ours has to do at least as well.
//!
//! Scoring against Rubber Band's output directly was the first attempt and it measured the wrong
//! thing: on a pure 440 Hz tone it puts -41 dB of sideband at 350 Hz where we put -88 dB, and a
//! test built that way fails us for being 46 dB cleaner than the reference.
//!
//! Regenerate after changing the stretcher or the signal set:
//!
//! ```text
//! python scripts/gen_stretch_golden.py
//! ```

mod stretch_golden_data;

use fluxion_ops::stretch::{pitch_shift, time_stretch};
use rustfft::{FftPlanner, num_complex::Complex};
use std::f32::consts::TAU;
use stretch_golden_data::{BAND_HI, BAND_LO, BANDS, FS, SECONDS, STRETCH_CASES, StretchCase};

fn lcg(n: usize) -> Vec<f32> {
    let mut state: u32 = 0x1234_5678;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / 16_777_216.0 * 2.0 - 1.0
        })
        .collect()
}

/// Rebuild one case's signal. Must match `signal()` in `scripts/gen_stretch_golden.py`.
fn signal(case: &StretchCase) -> Vec<f32> {
    let n = (SECONDS * FS as f32) as usize;
    let fs = f64::from(FS);
    let sine = |freq: f64, i: usize| (std::f64::consts::TAU * freq * i as f64 / fs).sin();
    match case.kind {
        "tone" => (0..n)
            .map(|i| (sine(f64::from(case.freq), i) * 0.5) as f32)
            .collect(),
        "chord" => (0..n)
            .map(|i| ((sine(220.0, i) + sine(277.183, i) + sine(329.628, i)) * 0.2) as f32)
            .collect(),
        "sweep" => {
            let (f0, f1, secs) = (100.0f64, 8000.0f64, f64::from(SECONDS));
            (0..n)
                .map(|i| {
                    let t = i as f64 / fs;
                    ((std::f64::consts::TAU * (f0 * t + 0.5 * (f1 - f0) / secs * t * t)).sin()
                        * 0.5) as f32
                })
                .collect()
        }
        "noise" => {
            let sos = fluxion_ops::iir::butterworth_lowpass(8, 12_000.0, FS);
            fluxion_ops::sos_filter(&lcg(n), &sos)
                .into_iter()
                .map(|v| v * 0.5)
                .collect()
        }
        other => panic!("case '{}': unknown signal kind '{other}'", case.name),
    }
}

/// Band spectrum of the whole signal in dB, normalized so the loudest band reads 0.
/// Must match `spectrum()` in `scripts/gen_stretch_golden.py`.
fn spectrum(x: &[f32]) -> Vec<f32> {
    let len = x.len();
    let n = len.next_power_of_two();
    let mut buffer = vec![Complex::new(0.0f32, 0.0); n];
    for (i, slot) in buffer.iter_mut().take(len).enumerate() {
        // numpy's `hanning` is the symmetric window, divisor len-1.
        let w = 0.5 - 0.5 * (TAU * i as f32 / (len - 1) as f32).cos();
        *slot = Complex::new(x[i] * w, 0.0);
    }
    FftPlanner::new().plan_fft_forward(n).process(&mut buffer);

    let bin_hz = FS as f32 / n as f32;
    let ratio = (BAND_HI / BAND_LO).powf(1.0 / BANDS as f32);
    let mut out: Vec<f32> = (0..BANDS)
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
            20.0 * rms.max(1e-12).log10()
        })
        .collect();
    let peak = out.iter().copied().fold(f32::MIN, f32::max);
    for v in &mut out {
        *v -= peak;
    }
    out
}

/// Mean deviation from the source spectrum, in dB, over the bands where the source has content.
///
/// Measured, not chosen; the test prints the whole table. Ours runs 0.00 dB on a sustained chord,
/// 0.85–1.69 dB on a sweep, and 1.20–2.35 dB on noise — the last is the worst case and sets this.
const MEAN_TOLERANCE_DB: f32 = 3.0;

/// The same, for the single worst band rather than the average. Worst measured is 8.25 dB, in the
/// sweep around 350 Hz where the instantaneous frequency crosses a band edge.
const BAND_TOLERANCE_DB: f32 = 10.0;

/// How much worse than Rubber Band ours is allowed to be, on the mean.
///
/// It is not zero, and the reason is written down rather than tuned away: this stretcher has no
/// transient or noise handling (`docs/time-stretch.md`), so on band-limited noise Rubber Band tracks
/// the source about 0.7 dB better than we do. On tonal material — the chord and the sweep — ours is
/// three to ten times closer, which is what peak-locked phases buy. The margin covers the noise
/// cases and nothing more; if it ever needs raising, the fix is transient handling, not the number.
const BEHIND_REFERENCE_DB: f32 = 1.5;

/// How far below the loudest band still counts as source content.
const SIGNAL_RANGE_DB: f32 = 45.0;

/// Pre-condition: `expected_db` is the spectrum Rubber Band produced for the same stretch.
/// Post-condition: ours tracks the source spectrum at least as closely, within a stated margin.
#[test]
fn tracks_the_source_at_least_as_well_as_rubberband() {
    assert!(STRETCH_CASES.len() >= 12, "the oracle set shrank");

    println!(
        "{:12} {:>8} {:>12} {:>14}",
        "case", "ours", "rubberband", "worst band"
    );
    for case in STRETCH_CASES {
        let x = signal(case);
        let source = spectrum(&x);
        let mine = spectrum(&time_stretch(&x, FS, case.ratio));
        assert_eq!(mine.len(), case.expected_db.len());

        let (mut ours, mut theirs, mut count) = (0.0f32, 0.0f32, 0usize);
        let (mut worst, mut worst_band) = (0.0f32, 0usize);
        for band in 0..BANDS {
            // Only where the *source* has something. Elsewhere both are describing their own
            // window's leakage, which is a property of the analysis and not of either stretcher.
            if source[band] < -SIGNAL_RANGE_DB {
                continue;
            }
            let mine_off = (mine[band] - source[band]).abs();
            ours += mine_off;
            theirs += (case.expected_db[band] - source[band]).abs();
            count += 1;
            if mine_off > worst {
                worst = mine_off;
                worst_band = band;
            }
        }
        assert!(count > 0, "case '{}': the source has no content", case.name);
        let (ours, theirs) = (ours / count as f32, theirs / count as f32);
        println!(
            "{:12} {ours:8.2} {theirs:12.2} {worst:11.2} @{worst_band}",
            case.name
        );

        assert!(
            worst <= BAND_TOLERANCE_DB,
            "case '{}': band {worst_band} is {worst:.2} dB off the source",
            case.name
        );
        assert!(
            ours <= MEAN_TOLERANCE_DB,
            "case '{}': {ours:.2} dB from the source on average",
            case.name
        );
        assert!(
            ours <= theirs + BEHIND_REFERENCE_DB,
            "case '{}': we are {:.2} dB from the source where Rubber Band is {theirs:.2}",
            case.name,
            ours
        );
    }
}

/// The roadmap asks for exact duration, and it is worth being explicit that this is *not* something
/// the reference provides: Rubber Band lands on 93566 frames where 96000 was asked for. A host
/// laying a stretched clip on a timeline needs the number it asked for, so ours is exact.
#[test]
fn duration_is_exact() {
    for case in STRETCH_CASES {
        let x = signal(case);
        let y = time_stretch(&x, FS, case.ratio);
        let want = (x.len() as f64 * f64::from(case.ratio)).round() as usize;
        assert_eq!(y.len(), want, "case '{}'", case.name);
    }
}

/// Peak frequency, refined below the bin spacing by fitting a parabola to the log magnitudes.
fn peak_hz(x: &[f32], fs: u32) -> f32 {
    let n = x.len().next_power_of_two() / 2;
    let mut buf: Vec<Complex<f32>> = (0..n)
        .map(|i| {
            let w = 0.5 - 0.5 * (TAU * i as f32 / n as f32).cos();
            Complex::new(x[i] * w, 0.0)
        })
        .collect();
    FftPlanner::new().plan_fft_forward(n).process(&mut buf);

    let mag: Vec<f32> = buf[..n / 2].iter().map(|c| c.norm().max(1e-12)).collect();
    let k = (1..mag.len() - 1)
        .max_by(|a, b| mag[*a].partial_cmp(&mag[*b]).unwrap())
        .unwrap();
    let (a, b, c) = (mag[k - 1].ln(), mag[k].ln(), mag[k + 1].ln());
    let delta = 0.5 * (a - c) / (a - 2.0 * b + c);
    (k as f32 + delta) * fs as f32 / n as f32
}

/// ROADMAP R4's own check, written out: a 440 Hz sine shifted +1200 cents peaks at 880 Hz ± 1 Hz,
/// duration unchanged. Extended over the range a musician would actually reach for.
#[test]
fn pitch_shift_lands_on_the_interval() {
    let fs = 48_000;
    let x: Vec<f32> = (0..fs as usize)
        .map(|i| (TAU * 440.0 * i as f32 / fs as f32).sin() * 0.5)
        .collect();

    for semitones in -12..=12 {
        let cents = semitones as f32 * 100.0;
        let want = 440.0 * 2f32.powf(cents / 1200.0);
        let y = pitch_shift(&x, fs, cents);

        assert_eq!(y.len(), x.len(), "{cents} cents changed the duration");
        // Skip the first quarter: both the vocoder and the resampler are still filling there.
        let hz = peak_hz(&y[fs as usize / 4..], fs);
        assert!(
            (hz - want).abs() < 1.0,
            "{semitones:+} semitones: peak at {hz:.2} Hz, expected {want:.2}"
        );
    }
}

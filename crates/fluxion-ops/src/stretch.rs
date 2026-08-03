//! Time-stretch and pitch-shift (ROADMAP R3, R4).
//!
//! Two knobs that people expect to turn independently: make it longer without making it lower, and
//! make it higher without making it longer. `transform::speed` does neither — it resamples, so
//! tempo and pitch move together, the way a tape machine does.
//!
//! [`time_stretch`] is a phase vocoder with **peak-locked phases**. The short version: cut the
//! signal into overlapping windows, work out from the phase change between windows what frequency
//! each bin *actually* holds, then lay the windows back down at a different spacing and advance
//! each phase by what that frequency would have done over the new spacing. The locking is the part
//! that matters — see below.
//!
//! [`pitch_shift`] is [`time_stretch`] followed by [`resample`](crate::resample): stretch by the
//! pitch ratio, then play the result back that much faster. The duration change cancels; the pitch
//! change does not.
//!
//! `docs/time-stretch.md` records why this is written rather than bound or ported, and what is
//! deliberately missing (transient handling, above all).

use std::f32::consts::TAU;

use rustfft::{FftPlanner, num_complex::Complex};

use crate::resample::{Quality, Resampler};

/// Windows overlapping at any point. Four is the usual choice for a phase vocoder: enough that the
/// Hann window sums flat, few enough that the FFT count stays sane.
const OVERLAP: usize = 4;

/// How far the stretch may go. Beyond this the fixed window is the wrong length for the job and the
/// result stops being worth defending, so the ratio is clamped rather than quietly degrading.
const MIN_RATIO: f32 = 0.1;
const MAX_RATIO: f32 = 10.0;

/// FFT size, in samples. About 85 ms, rounded up to a power of two.
///
/// The window sets the trade the vocoder cannot avoid: long enough to resolve the bass (at 48 kHz
/// this puts the first bin at 12 Hz), short enough that a transient is not smeared across a whole
/// syllable. 85 ms is the middle ground the literature and every shipping stretcher land near.
fn window_len(fs: u32) -> usize {
    ((fs as f64 * 0.085) as usize)
        .next_power_of_two()
        .clamp(256, 8192)
}

/// Wrap to `(-PI, PI]`.
fn princarg(x: f32) -> f32 {
    x - TAU * (x / TAU).round()
}

/// Time-stretch by `ratio`: output duration over input duration.
///
/// `2.0` is twice as long at the same pitch, `0.5` is half. `fs` is only used to size the window.
/// The output is **exactly** `round(x.len() * ratio)` samples — the roadmap asks for exact duration
/// and a vocoder's frame count does not naturally land there, so the last partial frame is written
/// and then the buffer is cut to length.
///
/// A ratio of 1 returns a copy rather than a round trip through the vocoder, which would only add
/// its own colour to a signal nobody asked to change.
pub fn time_stretch(x: &[f32], fs: u32, ratio: f32) -> Vec<f32> {
    let ratio = ratio.clamp(MIN_RATIO, MAX_RATIO);
    let out_len = (x.len() as f64 * f64::from(ratio)).round() as usize;
    if x.is_empty() || out_len == 0 {
        return vec![0.0; out_len];
    }
    if (ratio - 1.0).abs() < 1e-6 {
        let mut out = x.to_vec();
        out.resize(out_len, 0.0);
        return out;
    }

    let n = window_len(fs);
    let hop_out = n / OVERLAP;
    let hop_in = f64::from(hop_out as u32) / f64::from(ratio);
    let bins = n / 2 + 1;

    let window: Vec<f32> = (0..n)
        .map(|i| 0.5 - 0.5 * (TAU * i as f32 / n as f32).cos())
        .collect();

    // `OVERLAP - 1` lead-in frames read from before the start of the signal, so that by the time
    // the window reaches real audio the overlap is already at full depth. Without them the first
    // half-window is normalized by a partial sum, which is a fade-in nobody asked for.
    let lead = OVERLAP - 1;
    let frames = out_len.div_ceil(hop_out) + OVERLAP;

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);

    let mut buf = vec![Complex::new(0.0f32, 0.0); n];
    let mut mag = vec![0.0f32; bins];
    let mut phase = vec![0.0f32; bins];
    let mut prev_phase = vec![0.0f32; bins];
    let mut sum_phase = vec![0.0f32; bins];
    let mut peak_of = vec![0usize; bins];
    let mut peaks: Vec<usize> = Vec::with_capacity(bins / 2);

    // Frame `f` is laid down at `f * hop_out`, so index `lead * hop_out + n / 2` is input time zero.
    let origin = lead * hop_out + n / 2;
    let total = (frames - 1) * hop_out + n;
    let mut acc = vec![0.0f32; total];
    let mut norm = vec![0.0f32; total];

    let mut prev_pos: isize = 0;
    // Set whenever the last frame was silent, so the next frame with content starts from the
    // signal's own phases instead of from an accumulator that has been free-running through
    // nothing. This doubles as the only transient handling here: a gap resets the vocoder.
    let mut restart = true;

    for f in 0..frames {
        let pos = ((f as f64 - lead as f64) * hop_in).round() as isize;
        let start = pos - (n as isize) / 2;

        // Read the analysis window, zero outside the signal.
        let mut loudest = 0.0f32;
        for (i, slot) in buf.iter_mut().enumerate() {
            let j = start + i as isize;
            let v = if j >= 0 && (j as usize) < x.len() {
                x[j as usize]
            } else {
                0.0
            };
            loudest = loudest.max(v.abs());
            *slot = Complex::new(v * window[i], 0.0);
        }
        if loudest == 0.0 {
            // Nothing to transform and nothing to add; `acc` is already zero here.
            restart = true;
            continue;
        }

        fft.process(&mut buf);
        for k in 0..bins {
            mag[k] = buf[k].norm();
            phase[k] = buf[k].arg();
        }

        if restart {
            sum_phase.copy_from_slice(&phase);
            restart = false;
        } else {
            // Frames are placed at rounded sample positions, so the true analysis hop varies by a
            // sample. Using the actual gap rather than the nominal one keeps the frequency estimate
            // honest — at Nyquist a one-sample error is half a cycle.
            let ha = (pos - prev_pos).max(1) as f32;

            // Peaks first: a bin louder than its four neighbours. Everything else follows one.
            peaks.clear();
            for k in 2..bins - 2 {
                let m = mag[k];
                if m > mag[k - 1] && m > mag[k + 1] && m > mag[k - 2] && m > mag[k + 2] {
                    peaks.push(k);
                }
            }

            if peaks.is_empty() {
                for k in 0..bins {
                    sum_phase[k] = advance(
                        k,
                        n,
                        ha,
                        hop_out as f32,
                        phase[k],
                        prev_phase[k],
                        sum_phase[k],
                    );
                }
            } else {
                for &p in &peaks {
                    sum_phase[p] = advance(
                        p,
                        n,
                        ha,
                        hop_out as f32,
                        phase[p],
                        prev_phase[p],
                        sum_phase[p],
                    );
                }
                // Every bin belongs to the nearest peak.
                let mut i = 0;
                for (k, slot) in peak_of.iter_mut().enumerate() {
                    while i + 1 < peaks.len() && k.abs_diff(peaks[i]) > k.abs_diff(peaks[i + 1]) {
                        i += 1;
                    }
                    *slot = peaks[i];
                }
                // Identity phase locking: a bin keeps the phase relationship to its peak that it
                // has in *this* analysis frame. That relationship is what makes a partial sound
                // like one partial; advancing every bin independently is what makes a phase
                // vocoder sound smeared.
                for k in 0..bins {
                    let p = peak_of[k];
                    if k != p {
                        sum_phase[k] = sum_phase[p] + phase[k] - phase[p];
                    }
                }
            }
        }
        prev_phase.copy_from_slice(&phase);
        prev_pos = pos;

        // Rebuild the frame from the original magnitudes and the advanced phases.
        for k in 0..bins {
            let (sin, cos) = sum_phase[k].sin_cos();
            buf[k] = Complex::new(mag[k] * cos, mag[k] * sin);
        }
        for k in 1..n / 2 {
            buf[n - k] = buf[k].conj();
        }
        ifft.process(&mut buf);

        let scale = 1.0 / n as f32;
        let at = f * hop_out;
        for i in 0..n {
            acc[at + i] += buf[i].re * scale * window[i];
            norm[at + i] += window[i] * window[i];
        }
    }

    // Divide out the overlapped window energy. At 75 % overlap the steady-state sum is 1.5; the
    // floor keeps the ends, where fewer windows land, from being amplified into noise.
    let floor = 0.1 * OVERLAP as f32 / 2.0;
    (0..out_len)
        .map(|i| {
            let j = origin + i;
            if j < total {
                acc[j] / norm[j].max(floor)
            } else {
                0.0
            }
        })
        .collect()
}

/// One bin's new synthesis phase: measure what frequency it really holds, then advance the
/// accumulated output phase by what that frequency does over the synthesis hop.
#[inline]
fn advance(k: usize, n: usize, ha: f32, hs: f32, phase: f32, prev: f32, accumulated: f32) -> f32 {
    // The bin's nominal angular frequency, per sample.
    let nominal = TAU * k as f32 / n as f32;
    // Whatever the phase did beyond the nominal advance says where in the bin the partial sits.
    let deviation = princarg(phase - prev - nominal * ha);
    let true_freq = nominal + deviation / ha;
    princarg(accumulated + true_freq * hs)
}

/// Pitch-shift by `cents`, keeping the duration.
///
/// 1200 cents is an octave up, -1200 an octave down. Built the way R4 says: [`time_stretch`] by the
/// pitch ratio, then [`Resampler`] the result back to the original length. The stretch moves the
/// tempo and leaves the pitch; the resample moves both; what survives is pitch alone.
///
/// The output is exactly `x.len()` samples.
pub fn pitch_shift(x: &[f32], fs: u32, cents: f32) -> Vec<f32> {
    if x.is_empty() || cents.abs() < 1e-6 {
        return x.to_vec();
    }
    // Ratio the frequencies move by. Up an octave is 2.0.
    let ratio = 2f32.powf(cents / 1200.0);
    let ratio = ratio.clamp(MIN_RATIO, MAX_RATIO);

    // Stretch by `ratio`, then read back `ratio` times faster.
    let stretched = time_stretch(x, fs, ratio);

    // `Resampler` takes rates, not a ratio; a millionth is far finer than any pitch that can be
    // heard, and far finer than R4's 1 Hz at 880.
    let from = 1_000_000u32;
    let to = ((f64::from(from) / f64::from(ratio)).round() as u32).max(1);

    let block = 4096;
    let mut r = Resampler::new(from, to, Quality::Hq, block);
    // The filter is centred, so its output lags the input by half its length. Flushing that many
    // extra zeros through pushes the real tail out; dropping that many output frames lines the
    // start back up.
    let drop = (r.latency() as f64 * f64::from(to) / f64::from(from)).round() as usize;
    let mut fed = stretched;
    fed.resize(fed.len() + r.latency() + block, 0.0);

    let mut out = Vec::with_capacity(r.max_output(fed.len()));
    let mut scratch = vec![0.0f32; r.max_output(block)];
    for chunk in fed.chunks(block) {
        let k = r.process(chunk, &mut scratch);
        out.extend_from_slice(&scratch[..k]);
    }

    let mut out = if drop < out.len() {
        out.split_off(drop)
    } else {
        Vec::new()
    };
    // Exact duration: the roadmap's check, and the reason this is safe to expose as an op rather
    // than a geometry stage.
    out.resize(x.len(), 0.0);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(freq: f32, fs: u32, frames: usize) -> Vec<f32> {
        (0..frames)
            .map(|i| (TAU * freq * i as f32 / fs as f32).sin() * 0.5)
            .collect()
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

    #[test]
    fn duration_is_exact() {
        let x = tone(440.0, 48_000, 48_000);
        for ratio in [0.5, 0.75, 1.0, 1.5, 2.0, 1.0 / 3.0] {
            let y = time_stretch(&x, 48_000, ratio);
            let want = (x.len() as f64 * f64::from(ratio)).round() as usize;
            assert_eq!(y.len(), want, "ratio {ratio}");
        }
    }

    #[test]
    fn stretching_does_not_move_the_pitch() {
        let fs = 48_000;
        let x = tone(440.0, fs, fs as usize);
        for ratio in [0.5, 0.75, 1.5, 2.0] {
            let y = time_stretch(&x, fs, ratio);
            let hz = peak_hz(&y[fs as usize / 4..], fs);
            assert!(
                (hz - 440.0).abs() < 1.0,
                "ratio {ratio}: peak at {hz:.2} Hz, expected 440"
            );
        }
    }

    #[test]
    fn a_stretched_tone_keeps_its_level() {
        let fs = 48_000;
        let x = tone(440.0, fs, fs as usize);
        for ratio in [0.5, 1.5, 2.0] {
            let y = time_stretch(&x, fs, ratio);
            // Measure well inside, away from the ends where the windows are still filling.
            let mid = &y[y.len() / 4..y.len() * 3 / 4];
            let rms = (mid.iter().map(|v| v * v).sum::<f32>() / mid.len() as f32).sqrt();
            let want = 0.5 / std::f32::consts::SQRT_2;
            assert!(
                (rms / want - 1.0).abs() < 0.1,
                "ratio {ratio}: rms {rms:.4}, expected about {want:.4}"
            );
        }
    }

    #[test]
    fn ratio_one_is_a_copy() {
        let x = tone(1000.0, 48_000, 4800);
        assert_eq!(time_stretch(&x, 48_000, 1.0), x);
    }

    #[test]
    fn silence_stays_silent_and_the_right_length() {
        let y = time_stretch(&vec![0.0; 48_000], 48_000, 1.7);
        assert_eq!(y.len(), 81_600);
        assert!(y.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn empty_input_is_empty_output() {
        assert!(time_stretch(&[], 48_000, 2.0).is_empty());
    }

    #[test]
    fn pitch_shift_moves_the_pitch_and_not_the_length() {
        let fs = 48_000;
        let x = tone(440.0, fs, fs as usize);
        for (cents, want) in [(1200.0, 880.0), (-1200.0, 220.0), (700.0, 659.26)] {
            let y = pitch_shift(&x, fs, cents);
            assert_eq!(y.len(), x.len(), "{cents} cents changed the duration");
            let hz = peak_hz(&y[fs as usize / 4..], fs);
            assert!(
                (hz - want).abs() < 1.0,
                "{cents} cents: peak at {hz:.2} Hz, expected {want}"
            );
        }
    }

    #[test]
    fn zero_cents_is_a_copy() {
        let x = tone(1000.0, 48_000, 4800);
        assert_eq!(pitch_shift(&x, 48_000, 0.0), x);
    }
}

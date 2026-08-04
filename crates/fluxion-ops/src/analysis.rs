//! What an observer tap measures (ROADMAP A2, A3).
//!
//! These are pure measurements: signal in, numbers out, nothing touched. The chain-level plumbing
//! that carries them is [`fluxion_core::tap`]; this is the arithmetic underneath.

use fluxion_core::{TapData, TapKind};
use rustfft::{FftPlanner, num_complex::Complex};

/// Measure `channels` as `kind` asks.
pub fn measure(channels: &[Vec<f32>], kind: TapKind, fs: u32) -> TapData {
    match kind {
        TapKind::Spectrum { size, overlap } => spectrum(channels, size, overlap, fs),
        TapKind::Meter => meter(channels, fs),
    }
}

/// Mean magnitude spectrum over every window the signal is long enough for.
///
/// Averaged rather than a single frame, because one frame of an analyser view is mostly noise: the
/// average is what makes a stationary signal's spectrum a stable, checkable quantity. A Hann window
/// tapers each frame, and the result is scaled so a full-scale sine reads its own amplitude in the
/// bin it lands on rather than a number that depends on the window and the FFT size.
///
/// Channels are averaged together: an analyser view is of the programme, not of one side of it.
fn spectrum(channels: &[Vec<f32>], size: usize, overlap: f32, fs: u32) -> TapData {
    let size = size.max(2).next_power_of_two();
    let bins = size / 2 + 1;
    let bin_hz = fs as f32 / size as f32;
    let frames = channels.iter().map(|c| c.len()).max().unwrap_or(0);

    let mut magnitude = vec![0.0f32; bins];
    if channels.is_empty() || frames < size {
        // Not one whole window: an empty spectrum, rather than a shape invented from a padded
        // fragment.
        return TapData::Spectrum {
            bin_hz,
            magnitude: vec![0.0; bins],
        };
    }

    // Hann, and the coherent gain (0.5) it costs, so a sine reads its amplitude.
    let window: Vec<f32> = (0..size)
        .map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / size as f32).cos())
        .collect();
    let scale = 2.0 / (size as f32 * 0.5);

    let hop = (((1.0 - overlap.clamp(0.0, 0.95)) * size as f32).round() as usize).max(1);
    let fft = FftPlanner::new().plan_fft_forward(size);
    let mut buffer = vec![Complex::new(0.0f32, 0.0); size];
    let mut windows = 0usize;

    let mut start = 0;
    while start + size <= frames {
        for slot in buffer.iter_mut() {
            *slot = Complex::new(0.0, 0.0);
        }
        // Sum the channels into one frame, then normalize: the average programme, not the sum.
        for channel in channels {
            for (i, slot) in buffer.iter_mut().enumerate() {
                slot.re += channel.get(start + i).copied().unwrap_or(0.0);
            }
        }
        for (i, slot) in buffer.iter_mut().enumerate() {
            slot.re = slot.re / channels.len() as f32 * window[i];
        }
        fft.process(&mut buffer);
        for (bin, slot) in magnitude.iter_mut().enumerate() {
            *slot += buffer[bin].norm() * scale;
        }
        windows += 1;
        start += hop;
    }

    if windows > 0 {
        for slot in magnitude.iter_mut() {
            *slot /= windows as f32;
        }
        // DC and Nyquist appear once in the transform, not twice, so the factor of two that makes
        // the other bins read a sine's amplitude over-counts them.
        magnitude[0] /= 2.0;
        if let Some(last) = magnitude.last_mut() {
            *last /= 2.0;
        }
    }
    TapData::Spectrum { bin_hz, magnitude }
}

/// Peak, RMS and the loudest short-term window.
fn meter(channels: &[Vec<f32>], fs: u32) -> TapData {
    // `sample_peak` is already dBFS, and `-inf` for silence.
    let peak_db = crate::loudness::sample_peak(channels);

    let (mut sum, mut count) = (0.0f64, 0usize);
    for channel in channels {
        for s in channel {
            sum += f64::from(*s) * f64::from(*s);
            count += 1;
        }
    }
    // dBFS, to match `peak` and the loudness below. A silent signal has no level at all, and
    // `log10(0)` says so.
    let rms_db = if count > 0 {
        20.0 * (sum / count as f64).sqrt().log10() as f32
    } else {
        f32::NEG_INFINITY
    };

    // The loudest 3 s window is what a meter's short-term reading peaks at. Material too short for
    // one window has no short-term loudness, and says so rather than reporting a number.
    let short_term = crate::loudness::short_term_loudness(channels, fs);
    let short_term_lufs = short_term
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);

    TapData::Meter {
        peak_db,
        rms_db,
        short_term_lufs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    const FS: u32 = 48_000;

    fn tone(freq: f32, amp: f32, frames: usize) -> Vec<f32> {
        (0..frames)
            .map(|i| (TAU * freq * i as f32 / FS as f32).sin() * amp)
            .collect()
    }

    /// A sine has to read its own amplitude in its own bin — the scaling that makes a spectrum
    /// mean something rather than being proportional to one.
    #[test]
    fn a_sine_reads_its_amplitude_in_its_own_bin() {
        // 1500 Hz at a 1024-point FFT and 48 kHz is bin 32 exactly, so no leakage muddies it.
        let x = vec![tone(1500.0, 0.5, 48_000)];
        let TapData::Spectrum { bin_hz, magnitude } = measure(
            &x,
            TapKind::Spectrum {
                size: 1024,
                overlap: 0.5,
            },
            FS,
        ) else {
            panic!("expected a spectrum");
        };
        assert_eq!(magnitude.len(), 513);
        let bin = (1500.0 / bin_hz).round() as usize;
        assert_eq!(bin, 32);
        assert!(
            (magnitude[bin] - 0.5).abs() < 0.01,
            "bin {bin} read {}, expected 0.5",
            magnitude[bin]
        );
    }

    /// Too short for one window is an empty spectrum, not a shape invented from a fragment.
    #[test]
    fn a_signal_shorter_than_one_window_has_no_spectrum() {
        let x = vec![tone(1000.0, 0.5, 100)];
        let TapData::Spectrum { magnitude, .. } = measure(
            &x,
            TapKind::Spectrum {
                size: 1024,
                overlap: 0.5,
            },
            FS,
        ) else {
            panic!("expected a spectrum");
        };
        assert!(magnitude.iter().all(|m| *m == 0.0));
    }

    /// The meter's three numbers, against what a sine's are by definition.
    #[test]
    fn the_meter_reads_a_sine() {
        let x = vec![tone(1000.0, 0.5, FS as usize * 4)];
        let TapData::Meter {
            peak_db,
            rms_db,
            short_term_lufs,
        } = measure(&x, TapKind::Meter, FS)
        else {
            panic!("expected a meter");
        };
        // 0.5 is -6.02 dBFS, and a sine's RMS is 3.01 dB below its peak.
        assert!((peak_db + 6.02).abs() < 0.05, "peak {peak_db} dBFS");
        assert!((rms_db + 9.03).abs() < 0.05, "rms {rms_db} dBFS");
        // A 1 kHz sine at -6 dBFS: K-weighting is ~0 dB at 1 kHz, so this lands near -9 LUFS
        // (-6 dBFS peak is -9 dBFS RMS) — the same number `integrated_loudness` gives it.
        let integrated = crate::loudness::integrated_loudness(&x, FS);
        assert!(
            (short_term_lufs - integrated).abs() < 0.5,
            "short-term {short_term_lufs} against integrated {integrated}"
        );
    }

    /// Silence measures as silence, and says "no loudness" rather than 0 LUFS — which would mean
    /// full scale.
    #[test]
    fn silence_has_no_loudness() {
        let x = vec![vec![0.0f32; FS as usize * 4]];
        let TapData::Meter {
            peak_db,
            rms_db,
            short_term_lufs,
        } = measure(&x, TapKind::Meter, FS)
        else {
            panic!("expected a meter");
        };
        assert_eq!(peak_db, f32::NEG_INFINITY);
        assert_eq!(rms_db, f32::NEG_INFINITY);
        assert!(
            short_term_lufs < -100.0,
            "silence read {short_term_lufs} LUFS"
        );
    }
}

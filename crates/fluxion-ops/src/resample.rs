//! Streaming sample-rate conversion (ROADMAP R1).
//!
//! [`transform::resample`](crate::transform::resample) needs the whole signal up front, which is
//! exactly what a realtime path and an AudioWorklet cannot give it. This is the same windowed-sinc
//! conversion arranged so blocks go in and blocks come out, with every buffer allocated when the
//! [`Resampler`] is built and none afterwards.
//!
//! Two qualities, because the two uses want different things. [`Quality::Hq`] is the offline
//! filter, for import and export. [`Quality::Fast`] is a quarter of the taps, for scrubbing and
//! varispeed where the rate is moving and nobody is listening closely.
//!
//! # Why there is a table
//!
//! The offline version evaluates `sin()` per tap per output sample. That is fine once; in an audio
//! callback it is not. The filter is precomputed here into a polyphase table at construction, and
//! the hot loop is a dot product with a linear blend between adjacent phases.

use std::f32::consts::PI;

/// How good, and how expensive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quality {
    /// A quarter of the taps: for scrubbing and varispeed, where the rate moves and the point is
    /// to keep up.
    Fast,
    /// The same filter [`transform::resample`](crate::transform::resample) uses, for anything that
    /// will be listened to or written to a file.
    Hq,
}

impl Quality {
    /// Zero crossings of the sinc on each side. Taps per phase is about twice this, divided by the
    /// cutoff when downsampling.
    fn zeros(self) -> f32 {
        match self {
            Quality::Fast => 8.0,
            Quality::Hq => 32.0,
        }
    }
}

/// Fractional positions the filter is precomputed at. Between them the two nearest phases are
/// blended, which costs one extra multiply-add and removes the stepping a coarse table would
/// otherwise put into a slow sweep.
const PHASES: usize = 512;

/// A streaming windowed-sinc sample-rate converter.
///
/// ```
/// use fluxion_ops::resample::{Quality, Resampler};
/// // 48 kHz in, 44.1 kHz out, in blocks of 128.
/// let mut r = Resampler::new(48_000, 44_100, Quality::Hq, 128);
/// let block = vec![0.0f32; 128];
/// let mut out = vec![0.0f32; r.max_output(128)];
/// let written = r.process(&block, &mut out);
/// assert!(written <= out.len());
/// ```
pub struct Resampler {
    /// Input samples consumed per output sample.
    step: f64,
    /// Output frames per input frame — the reciprocal, kept to size buffers.
    ratio: f64,
    /// Taps per phase.
    taps: usize,
    /// `PHASES + 1` rows of `taps`; the extra row is phase 1.0, so blending never wraps.
    table: Vec<f32>,
    /// The tail of the input seen so far, so a block boundary is not a discontinuity.
    history: Vec<f32>,
    /// Input frames received before the current block, and output frames produced so far.
    ///
    /// The read position is *computed* from these rather than accumulated, because accumulating
    /// `pos += step` makes the answer depend on where the block boundaries fell: converting 12000
    /// frames from 48k to 44.1k lands the last output exactly on the boundary, and a few
    /// thousandths of an ULP of drift decides whether it is produced. Counters give the same
    /// result for every block size, which is the property the caller actually needs.
    frames_in: u64,
    frames_out: u64,
    /// Where output frame 0 reads, in input frames from the first sample fed.
    ///
    /// Streaming, this is `-latency`: the converter starts half a kernel before the signal, so the
    /// output lags and the host compensates. [`Resampler::align_to_input`] moves it to 0, which is
    /// what an offline conversion wants — see [`convert_ratio`].
    origin: f64,
    quality: Quality,
}

impl Resampler {
    /// Build a converter from `from_fs` to `to_fs`, fed blocks of at most `max_block` frames.
    ///
    /// Everything is allocated here. `max_block` only sizes the history, so passing a larger block
    /// later is safe but will re-borrow from a shorter tail than intended; pass what you will use.
    pub fn new(from_fs: u32, to_fs: u32, quality: Quality, max_block: usize) -> Resampler {
        Resampler::with_ratio(f64::from(to_fs) / f64::from(from_fs), quality, max_block)
    }

    /// The same, given the ratio directly: output frames per input frame.
    ///
    /// Two rates are the readable way to say this and [`Resampler::new`] is the one to reach for.
    /// A ratio is for the conversions that are not between two rates at all — a speed factor, a
    /// pitch interval — where rounding the request into a pair of integer rates would quietly
    /// change what was asked for.
    pub fn with_ratio(ratio: f64, quality: Quality, max_block: usize) -> Resampler {
        let ratio = ratio.max(1e-9);
        let step = 1.0 / ratio;

        // Downsampling has to lowpass to the *output* Nyquist, which widens the kernel by the same
        // factor. Upsampling keeps the input Nyquist and the width is just the lobe count.
        let cutoff = ratio.min(1.0) as f32;
        let half = quality.zeros() / cutoff.max(1e-6);
        // Forced odd: a symmetric FIR wants an odd length, and it puts the centre tap on a whole
        // sample so the alignment arithmetic below is exact rather than truncated.
        let taps = ((2.0 * half).ceil() as usize) | 1;

        // The polyphase table: row `p` is the filter for a fractional offset of `p / PHASES`.
        let mut table = vec![0.0f32; (PHASES + 1) * taps];
        for phase in 0..=PHASES {
            let frac = phase as f32 / PHASES as f32;
            let row = &mut table[phase * taps..(phase + 1) * taps];
            let mut sum = 0.0f32;
            for (k, weight) in row.iter_mut().enumerate() {
                // Tap k sits at input offset `k - (taps-1)/2` from the centre; `frac` moves the
                // centre between samples.
                let dx = k as f32 - (taps - 1) as f32 / 2.0 - frac;
                *weight = cutoff * sinc(cutoff * dx) * blackman(dx / half);
                sum += *weight;
            }
            // Normalize each phase to unit DC gain, so a constant comes through as itself and the
            // conversion cannot drift in level with the fractional position.
            if sum.abs() > 1e-12 {
                for weight in row.iter_mut() {
                    *weight /= sum;
                }
            }
        }

        Resampler {
            step,
            ratio,
            taps,
            table,
            // History holds the taps the next block will reach back into.
            history: vec![0.0; taps + max_block],
            frames_in: 0,
            frames_out: 0,
            origin: -(((taps - 1) / 2) as f64),
            quality,
        }
    }

    /// Line output frame 0 up with input frame 0: no start-up delay, and the kernel's leading half
    /// reads the silence in front of the signal.
    ///
    /// This is for converting something whole, where the delay is a nuisance rather than a fact
    /// about the stream — [`latency`](Resampler::latency) reads 0 afterwards. Do not use it on a
    /// live stream: it throws away the first half-kernel of context that a stream genuinely has.
    pub fn align_to_input(&mut self) {
        self.origin = 0.0;
    }

    /// The quality this was built with.
    pub fn quality(&self) -> Quality {
        self.quality
    }

    /// Taps per phase — the filter's length, and the reason `Fast` is faster.
    pub fn taps(&self) -> usize {
        self.taps
    }

    /// Filter delay, in input frames. The output lags the input by this much.
    pub fn latency(&self) -> usize {
        (-self.origin).max(0.0) as usize
    }

    /// The most output frames `in_frames` of input can produce. Size the output buffer with this;
    /// a smaller one truncates.
    pub fn max_output(&self, in_frames: usize) -> usize {
        (in_frames as f64 * self.ratio).ceil() as usize + 2
    }

    /// Convert one block. Returns how many output frames were written.
    ///
    /// Allocation-free: everything it touches was sized in [`Resampler::new`].
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) -> usize {
        let history_len = self.history.len();
        // Odd tap count, so the centre is a whole sample and this is exact.
        let half = ((self.taps - 1) / 2) as i64;

        // Sample from the virtual concatenation of history and input: index 0 is the first sample
        // of `input`, negative indices reach back into what came before.
        let at = |history: &[f32], input: &[f32], index: i64| -> f32 {
            if index >= 0 {
                input.get(index as usize).copied().unwrap_or(0.0)
            } else {
                let back = (-index) as usize;
                if back <= history.len() {
                    history[history.len() - back]
                } else {
                    0.0
                }
            }
        };

        let mut written = 0;
        // The horizon: every sample received so far. An output can be produced once its whole
        // kernel is behind it.
        let horizon = (self.frames_in + input.len() as u64) as f64;
        while written < output.len() {
            // Output `n` reads at `origin + n * step`; by default `origin` is half a kernel before
            // the signal, so the first outputs are made partly from the silence in front of it.
            let centre_abs = self.frames_out as f64 * self.step + self.origin;
            if centre_abs + half as f64 >= horizon {
                break;
            }
            // Relative to the start of this block; negative reaches into history.
            let centre = centre_abs - self.frames_in as f64;
            let base = centre.floor();
            let frac = (centre - base) as f32;

            // Blend the two nearest phases rather than snapping to one.
            let phase = frac * PHASES as f32;
            let low = phase.floor() as usize;
            let mix = phase - low as f32;
            let (row_a, row_b) = (low.min(PHASES), (low + 1).min(PHASES));
            let a = &self.table[row_a * self.taps..(row_a + 1) * self.taps];
            let b = &self.table[row_b * self.taps..(row_b + 1) * self.taps];

            let start = base as i64 - half;
            let mut acc = 0.0f32;
            for k in 0..self.taps {
                let weight = a[k] + (b[k] - a[k]) * mix;
                acc += weight * at(&self.history, input, start + k as i64);
            }

            output[written] = acc;
            written += 1;
            self.frames_out += 1;
        }

        // Remember enough tail for the next block to reach back into.
        self.frames_in += input.len() as u64;
        if input.len() >= history_len {
            self.history
                .copy_from_slice(&input[input.len() - history_len..]);
        } else {
            self.history.copy_within(input.len().., 0);
            let keep = history_len - input.len();
            self.history[keep..].copy_from_slice(input);
        }

        written
    }

    /// Forget the signal, keep the filter. The next block starts as if from silence.
    pub fn reset(&mut self) {
        self.history.fill(0.0);
        self.frames_in = 0;
        self.frames_out = 0;
    }
}

/// Blocks used to run a whole signal through the streaming converter. Big enough that the
/// per-block bookkeeping disappears, small enough to stay in cache.
const OFFLINE_BLOCK: usize = 1024;

/// Convert a whole channel from `from_fs` to `to_fs`, returning exactly `round(len · to/from)`
/// samples aligned with the input in time.
///
/// This is [`Resampler`] run over a signal that happens to be complete, rather than a second
/// converter for offline use: a file imported whole and the same file streamed through a callback
/// come out as the same samples, which is the property a pinned project rate exists to give.
pub fn convert(input: &[f32], from_fs: u32, to_fs: u32, quality: Quality) -> Vec<f32> {
    if from_fs == to_fs {
        return input.to_vec();
    }
    convert_ratio(input, f64::from(to_fs) / f64::from(from_fs), quality)
}

/// The same, given output frames per input frame — see [`Resampler::with_ratio`].
pub fn convert_ratio(input: &[f32], ratio: f64, quality: Quality) -> Vec<f32> {
    let want = (input.len() as f64 * ratio).round() as usize;
    if input.is_empty() || want == 0 {
        return vec![0.0; want];
    }

    let mut r = Resampler::with_ratio(ratio, quality, OFFLINE_BLOCK);
    // Offline there is no such thing as "before the signal": output frame 0 is input frame 0. This
    // is what makes the alignment exact rather than rounded to the nearest output frame.
    let half = r.taps() / 2 + 1;
    r.align_to_input();

    let mut out = Vec::with_capacity(want + 2);
    let mut scratch = vec![0.0f32; r.max_output(OFFLINE_BLOCK)];
    for chunk in input.chunks(OFFLINE_BLOCK) {
        let n = r.process(chunk, &mut scratch);
        out.extend_from_slice(&scratch[..n]);
    }
    // A frame is only produced once its whole kernel has arrived, so the last half-kernel of the
    // signal is still inside the filter. Silence pushes it out.
    let flush = vec![0.0f32; half];
    for chunk in flush.chunks(OFFLINE_BLOCK) {
        let n = r.process(chunk, &mut scratch);
        out.extend_from_slice(&scratch[..n]);
    }

    // Exact, not approximate: a host laying this on a timeline needs the length it computed.
    out.resize(want, 0.0);
    out
}

/// A symmetric Blackman window over `u ∈ [-1, 1]` (0 outside), for tapering the sinc.
fn blackman(u: f32) -> f32 {
    if u.abs() > 1.0 {
        0.0
    } else {
        0.42 + 0.5 * (PI * u).cos() + 0.08 * (2.0 * PI * u).cos()
    }
}

/// Normalized sinc `sin(πx)/(πx)`, `sinc(0) = 1`.
fn sinc(x: f32) -> f32 {
    if x.abs() < 1e-8 {
        1.0
    } else {
        let px = PI * x;
        px.sin() / px
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run a whole signal through in blocks, the way a host would.
    fn stream(input: &[f32], from: u32, to: u32, quality: Quality, block: usize) -> Vec<f32> {
        let mut r = Resampler::new(from, to, quality, block);
        let mut out = Vec::with_capacity(r.max_output(input.len()));
        let mut scratch = vec![0.0f32; r.max_output(block)];
        for chunk in input.chunks(block) {
            let n = r.process(chunk, &mut scratch);
            out.extend_from_slice(&scratch[..n]);
        }
        out
    }

    fn tone(freq: f64, fs: u32, secs: f64) -> Vec<f32> {
        let n = (secs * f64::from(fs)) as usize;
        (0..n)
            .map(|i| (std::f64::consts::TAU * freq * i as f64 / f64::from(fs)).sin() as f32)
            .collect()
    }

    /// The output length has to track the rate ratio, whatever the block size.
    #[test]
    fn output_length_follows_the_ratio() {
        for (from, to) in [
            (48_000, 44_100),
            (44_100, 48_000),
            (48_000, 96_000),
            (96_000, 8_000),
        ] {
            let input = tone(440.0, from, 1.0);
            let out = stream(&input, from, to, Quality::Hq, 128);
            let expected = input.len() as f64 * f64::from(to) / f64::from(from);
            assert!(
                (out.len() as f64 - expected).abs() <= 2.0,
                "{from} -> {to}: got {} frames, expected about {expected:.0}",
                out.len()
            );
        }
    }

    /// Block size is an artefact of how audio is delivered, not part of the answer: the same input
    /// must convert identically however it is chopped up.
    #[test]
    fn the_result_does_not_depend_on_the_block_size() {
        let input = tone(1000.0, 48_000, 0.25);
        let reference = stream(&input, 48_000, 44_100, Quality::Hq, 128);
        for block in [1, 7, 64, 333, 4096] {
            let other = stream(&input, 48_000, 44_100, Quality::Hq, block);
            assert_eq!(
                other.len(),
                reference.len(),
                "block {block} changed the length"
            );
            for (i, (a, b)) in reference.iter().zip(&other).enumerate() {
                assert!(
                    (a - b).abs() < 1e-6,
                    "block {block}: sample {i} differs, {a} vs {b}"
                );
            }
        }
    }

    /// A tone comes out at the same frequency and the same level — the two things a converter must
    /// not change.
    #[test]
    fn a_tone_keeps_its_frequency_and_level() {
        let out = stream(&tone(1000.0, 48_000, 0.5), 48_000, 44_100, Quality::Hq, 256);
        // Skip the filter's start-up, then measure.
        let steady = &out[2000..out.len() - 2000];
        let peak = steady.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!((peak - 1.0).abs() < 0.02, "level changed: peak {peak}");

        // Count zero crossings to get the frequency without an FFT.
        let crossings = steady
            .windows(2)
            .filter(|w| w[0] <= 0.0 && w[1] > 0.0)
            .count();
        let seconds = steady.len() as f64 / 44_100.0;
        let freq = crossings as f64 / seconds;
        assert!(
            (freq - 1000.0).abs() < 2.0,
            "frequency changed: {freq:.1} Hz"
        );
    }

    /// Converting up and back down returns the original, delayed by the two filters. The round
    /// trip is the cheapest end-to-end check there is — and lining it up is what pins the meaning
    /// of `latency()`, which a host has to compensate for to keep parallel paths in phase.
    #[test]
    fn up_then_back_down_returns_the_signal() {
        let input = tone(500.0, 48_000, 0.3);
        let up = stream(&input, 48_000, 96_000, Quality::Hq, 128);
        let back = stream(&up, 96_000, 48_000, Quality::Hq, 128);

        // `latency()` is in input frames. The first stage's is already in 48 kHz frames; the
        // second's is in 96 kHz frames, so it is half as long in the output's own rate.
        let delay = Resampler::new(48_000, 96_000, Quality::Hq, 128).latency()
            + Resampler::new(96_000, 48_000, Quality::Hq, 128).latency() / 2;

        let n = (input.len() - delay).min(back.len() - delay);
        // Skip the filter starting up and stopping at the ends.
        let worst = (4000..n - 4000)
            .map(|i| (input[i] - back[i + delay]).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst < 0.02,
            "round trip lost the signal at a delay of {delay}: worst {worst}"
        );
    }

    /// The whole point of `Fast`: fewer taps, still correct, just a wider transition band.
    #[test]
    fn fast_is_cheaper_than_hq_and_still_works() {
        let fast = Resampler::new(48_000, 44_100, Quality::Fast, 128);
        let hq = Resampler::new(48_000, 44_100, Quality::Hq, 128);
        assert!(
            fast.taps() * 3 < hq.taps(),
            "Fast should be much shorter: {} vs {}",
            fast.taps(),
            hq.taps()
        );

        let out = stream(
            &tone(1000.0, 48_000, 0.5),
            48_000,
            44_100,
            Quality::Fast,
            128,
        );
        let steady = &out[1000..out.len() - 1000];
        let peak = steady.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(
            (peak - 1.0).abs() < 0.05,
            "Fast changed the level: peak {peak}"
        );
    }

    /// A constant must come out as the same constant, at any ratio: this is what the per-phase
    /// normalization is for, and it fails loudly if a phase is left unnormalized.
    #[test]
    fn a_constant_survives_at_unit_gain() {
        for (from, to) in [(48_000, 44_100), (44_100, 48_000), (22_050, 48_000)] {
            let out = stream(&vec![0.5f32; 20_000], from, to, Quality::Hq, 512);
            let steady = &out[2000..out.len() - 2000];
            for (i, v) in steady.iter().enumerate() {
                assert!(
                    (v - 0.5).abs() < 1e-3,
                    "{from} -> {to}: sample {i} came out {v}, not 0.5"
                );
            }
        }
    }

    /// Same rate in and out is still a filter: it delays by `latency()` and must not otherwise
    /// change the signal.
    #[test]
    fn a_matched_rate_passes_the_signal_through() {
        let input = tone(1000.0, 48_000, 0.2);
        let out = stream(&input, 48_000, 48_000, Quality::Hq, 128);
        assert_eq!(out.len(), input.len());

        let delay = Resampler::new(48_000, 48_000, Quality::Hq, 128).latency();
        let worst = (3000..input.len() - 3000 - delay)
            .map(|i| (input[i] - out[i + delay]).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst < 1e-3,
            "a matched rate changed the signal: worst {worst}"
        );
    }

    /// R2's own check, at the channel level: whatever goes in, the frame count that comes out is
    /// the one a host computed from the two rates.
    #[test]
    fn convert_lands_on_the_exact_length() {
        for (from, to) in [
            (48_000, 44_100),
            (44_100, 48_000),
            (22_050, 48_000),
            (96_000, 8_000),
            (48_000, 48_000),
            (8_000, 192_000),
        ] {
            for frames in [0, 1, 37, 4_800] {
                let out = convert(&tone(440.0, from, 1.0)[..frames], from, to, Quality::Hq);
                let want = (frames as f64 * f64::from(to) / f64::from(from)).round() as usize;
                assert_eq!(out.len(), want, "{from} -> {to}, {frames} frames");
            }
        }
    }

    /// Alignment, against the only reference that cannot itself be misaligned: the tone the
    /// converter is supposed to be producing. Output frame `i` must be the source at time
    /// `i / to_fs` — one frame of slip at 1 kHz is 0.14 rad of phase, and even the half-frame a
    /// rounded alignment would leave shows up as 0.02.
    ///
    /// Worst measured is 1.8e-6, which is f32 noise: the conversion is not approximately aligned,
    /// it is aligned. The bound is set an order of magnitude above that and nothing more.
    #[test]
    fn convert_puts_the_signal_where_the_output_rate_says_it_is() {
        for (from, to) in [(48_000u32, 44_100u32), (44_100, 48_000), (48_000, 96_000)] {
            let out = convert(&tone(1000.0, from, 0.5), from, to, Quality::Hq);
            // Skip the ends, where the kernel runs off the signal and the taper is real.
            let worst = (2000..out.len() - 2000)
                .map(|i| {
                    let want =
                        (std::f64::consts::TAU * 1000.0 * i as f64 / f64::from(to)).sin() as f32;
                    (out[i] - want).abs()
                })
                .fold(0.0f32, f32::max);
            assert!(worst < 1e-5, "{from} -> {to}: off by {worst}");
        }
    }

    /// The same rate in and out is not a conversion, and must not behave like one: no filter, no
    /// delay, the samples themselves.
    #[test]
    fn convert_at_a_matched_rate_is_the_input() {
        let x = tone(1000.0, 48_000, 0.1);
        assert_eq!(convert(&x, 48_000, 48_000, Quality::Hq), x);
    }

    #[test]
    fn reset_forgets_the_signal_but_keeps_the_filter() {
        let mut r = Resampler::new(48_000, 44_100, Quality::Hq, 128);
        let mut out = vec![0.0f32; r.max_output(128)];
        let loud = vec![1.0f32; 128];
        for _ in 0..20 {
            r.process(&loud, &mut out);
        }
        r.reset();
        let taps = r.taps();
        let n = r.process(&vec![0.0f32; 128], &mut out);
        assert_eq!(r.taps(), taps);
        assert!(
            out[..n].iter().all(|v| v.abs() < 1e-6),
            "silence after reset should be silent"
        );
    }
}

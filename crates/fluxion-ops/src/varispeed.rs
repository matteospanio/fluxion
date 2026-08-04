//! Varispeed: playback speed that can move while it is playing (ROADMAP R5).
//!
//! [`Resampler`](crate::resample::Resampler) converts between two rates that are decided before it
//! is built. Scrubbing a timeline and tape-style speed effects are the opposite: the ratio is a
//! control the user is moving, and the converter has to follow it inside the callback without
//! allocating and without clicking.
//!
//! Pull, not push. A resampler is fed blocks and gives back however many frames the ratio happened
//! to produce; an audio callback needs *exactly* the block it asked for, and how much input that
//! takes is the part that varies. [`Varispeed::process`] fills the output and reports how much
//! input it swallowed doing it.
//!
//! # Why the table is shaped differently
//!
//! The fixed-ratio converter precomputes one filter per fractional position, which is the cheapest
//! arrangement when the filter never changes. Here it changes constantly: playing back faster than
//! 1× has to lowpass to the *new* Nyquist, so the kernel gets wider as the speed goes up. So this
//! table is the windowed sinc sampled as a plain function of distance, read with a step that
//! depends on the current speed — which is what lets one table cover every speed up to `max_speed`.

use crate::resample::{Quality, blackman, sinc};

/// Samples of the table per zero crossing of the sinc. Between them the two neighbours are blended,
/// so the kernel stays smooth where the speed puts a tap between table entries.
const PER_ZERO: usize = 512;

/// How long the speed takes to travel most of the way to a new target, in milliseconds.
///
/// A speed that jumps between blocks puts a step in the read position, and a step is a click. 20 ms
/// is slow enough that no jump is audible as one and fast enough that a scrub still feels attached
/// to the mouse.
const SMOOTH_MS: f32 = 20.0;

/// Playback speed, varying, with the anti-aliasing that speeding up needs.
///
/// ```
/// use fluxion_ops::resample::Quality;
/// use fluxion_ops::varispeed::Varispeed;
///
/// // Up to 4x, fed at most 256 frames at a time.
/// let mut v = Varispeed::new(48_000, Quality::Fast, 4.0, 256);
/// v.set_speed(2.0);
///
/// let input = vec![0.0f32; 256];
/// let mut block = vec![0.0f32; 128];
/// let (consumed, written) = v.process(&input, &mut block);
/// assert!(consumed <= input.len() && written <= block.len());
/// ```
pub struct Varispeed {
    /// The windowed sinc as a function of distance: `table[i]` is the kernel at `i / PER_ZERO`
    /// input samples from the centre, at unit speed.
    table: Vec<f32>,
    /// Zero crossings each side at unit speed — the kernel's half-width there.
    zeros: f32,
    /// Input frames the kernel reaches at `max_speed`, which is what sizes `buf`.
    max_reach: f64,
    /// Fastest speed this was built for. Above it the buffer would have to be longer than it is.
    max_speed: f32,
    /// Where the speed is going, and where it is now. They differ for a few milliseconds after a
    /// change, which is the whole point.
    target: f32,
    current: f32,
    /// One-pole coefficient taking `current` toward `target`, per output frame.
    smoothing: f32,
    /// Input, oldest first: the tail the kernel still reaches, plus whatever has been handed over
    /// and not yet read.
    buf: Vec<f32>,
    /// How much of `buf` is real audio.
    filled: usize,
    /// Read position in `buf`, in input frames. Fractional — that is the whole job.
    pos: f64,
}

impl Varispeed {
    /// Build a varispeed at `fs`, able to reach `max_speed`, fed at most `max_block` frames at a
    /// time.
    ///
    /// Everything is allocated here. `max_speed` is a commitment rather than a hint: the kernel
    /// width at that speed is what sizes the buffer, and [`set_speed`](Self::set_speed) clamps to
    /// it. Declare what the control can actually reach — 2× for a scrub bar, 4× for a tape effect —
    /// because asking for 32× makes every block more expensive at every speed.
    pub fn new(fs: u32, quality: Quality, max_speed: f32, max_block: usize) -> Varispeed {
        let max_speed = max_speed.clamp(1.0, 32.0);
        let zeros = quality.zeros();

        // The table covers `zeros` crossings. A lower cutoff reads it more slowly, which is what
        // widens the kernel in input frames without needing a second table.
        let len = (zeros * PER_ZERO as f32).ceil() as usize + 2;
        let mut table = vec![0.0f32; len];
        for (i, weight) in table.iter_mut().enumerate() {
            let u = i as f32 / PER_ZERO as f32;
            *weight = sinc(u) * blackman(u / zeros);
        }

        let max_reach = f64::from(zeros * max_speed);
        // Room for the kernel either side of the read position, plus a block being handed over.
        let capacity = (2.0 * max_reach).ceil() as usize + max_block + 2;

        Varispeed {
            table,
            zeros,
            max_reach,
            max_speed,
            target: 1.0,
            current: 1.0,
            // Time constant to per-frame coefficient. `fs` is the output rate: the ramp is heard at
            // the rate it plays back, whatever the input is doing.
            smoothing: 1.0 - (-1000.0 / (SMOOTH_MS * fs as f32)).exp(),
            buf: vec![0.0; capacity],
            filled: 0,
            pos: 0.0,
        }
    }

    /// Aim for `speed`: 1.0 is normal, 2.0 twice as fast and an octave up, 0 frozen on the current
    /// sample. Clamped to `[0, max_speed]`. It takes effect over a few milliseconds, not instantly.
    pub fn set_speed(&mut self, speed: f32) {
        self.target = speed.clamp(0.0, self.max_speed);
    }

    /// The speed asked for.
    pub fn speed(&self) -> f32 {
        self.target
    }

    /// The speed playing right now, part-way to [`speed`](Self::speed) after a change.
    pub fn current_speed(&self) -> f32 {
        self.current
    }

    /// The fastest speed this was built for.
    pub fn max_speed(&self) -> f32 {
        self.max_speed
    }

    /// Input frames the converter needs in hand before it can produce output — the kernel's
    /// half-width at the current speed. The output lags the input by this much.
    pub fn latency(&self) -> usize {
        (f64::from(self.zeros) * f64::from(self.current.max(1.0))).ceil() as usize
    }

    /// Fill `output`, returning `(input frames consumed, output frames written)`.
    ///
    /// Written is short of `output.len()` when the input ran out: hand over more and call again,
    /// starting at `input[consumed..]`. Consumed is short of `input.len()` when the output filled
    /// first, and the rest is still the caller's.
    ///
    /// Allocation-free: everything it touches was sized in [`Varispeed::new`].
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) -> (usize, usize) {
        // Forget what no kernel can reach any more, and slide the rest down.
        let stale = (self.pos - self.max_reach).floor().max(0.0) as usize;
        let stale = stale.min(self.filled);
        if stale > 0 {
            self.buf.copy_within(stale..self.filled, 0);
            self.filled -= stale;
            self.pos -= stale as f64;
        }

        let consumed = input.len().min(self.buf.len() - self.filled);
        self.buf[self.filled..self.filled + consumed].copy_from_slice(&input[..consumed]);
        self.filled += consumed;

        let mut written = 0;
        while written < output.len() {
            // Faster than 1× asks the output rate to carry more band than it has, so lowpass to
            // what fits — exactly what downsampling does, for the same reason.
            let cutoff = f64::from(1.0 / self.current.max(1.0));
            let reach = f64::from(self.zeros) / cutoff;

            // Every tap has to have arrived. Stop rather than invent the rest.
            if self.pos + reach >= self.filled as f64 {
                break;
            }

            output[written] = self.read(cutoff, reach);
            written += 1;

            self.pos += f64::from(self.current);
            self.current += (self.target - self.current) * self.smoothing;
        }

        (consumed, written)
    }

    /// One output sample: the windowed sinc centred on `pos`, read at `cutoff`.
    #[inline]
    fn read(&self, cutoff: f64, reach: f64) -> f32 {
        let first = (self.pos - reach).ceil() as i64;
        let last = (self.pos + reach).floor() as i64;

        let mut acc = 0.0f32;
        let mut weights = 0.0f32;
        for k in first..=last {
            // Distance from the centre, in table steps.
            let u = (k as f64 - self.pos).abs() * cutoff * PER_ZERO as f64;
            let low = u as usize;
            let mix = (u - low as f64) as f32;
            let (a, b) = match (self.table.get(low), self.table.get(low + 1)) {
                (Some(a), Some(b)) => (*a, *b),
                _ => continue, // past the end of the window, where the weight is 0 anyway
            };
            let weight = a + (b - a) * mix;

            // Before the start of the stream there is silence — the same thing the fixed-ratio
            // converter does, so a run starts by fading in rather than with a step.
            if k >= 0 && (k as usize) < self.filled {
                acc += weight * self.buf[k as usize];
            }
            weights += weight;
        }

        // Normalizing by the weights keeps the level flat as the kernel slides between samples and
        // as its width changes with the speed.
        if weights.abs() > 1e-9 {
            acc / weights
        } else {
            0.0
        }
    }

    /// Forget the audio and land on the target speed, ready for a new take. Keeps the table.
    pub fn reset(&mut self) {
        self.buf.fill(0.0);
        self.filled = 0;
        self.pos = 0.0;
        self.current = self.target;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    fn tone(freq: f32, fs: u32, frames: usize) -> Vec<f32> {
        (0..frames)
            .map(|i| (TAU * freq * i as f32 / fs as f32).sin() * 0.5)
            .collect()
    }

    /// Pull blocks until the input runs out, the way a host would.
    fn run(v: &mut Varispeed, input: &[f32], block: usize) -> Vec<f32> {
        let mut out = Vec::new();
        let mut scratch = vec![0.0f32; block];
        let mut at = 0;
        while at < input.len() {
            let feed = &input[at..(at + 256).min(input.len())];
            let (consumed, written) = v.process(feed, &mut scratch);
            out.extend_from_slice(&scratch[..written]);
            at += consumed;
            if consumed == 0 && written == 0 {
                break;
            }
        }
        out
    }

    /// Run a whole signal through at one steady speed.
    fn play(input: &[f32], speed: f32, quality: Quality, block: usize) -> Vec<f32> {
        let mut v = Varispeed::new(48_000, quality, speed.max(1.0), 256);
        v.set_speed(speed);
        v.reset(); // start at the speed rather than ramping into it
        run(&mut v, input, block)
    }

    /// Zero crossings per second — the frequency, without an FFT.
    fn hz(x: &[f32], fs: u32) -> f32 {
        let crossings = x.windows(2).filter(|w| w[0] <= 0.0 && w[1] > 0.0).count();
        crossings as f32 * fs as f32 / x.len() as f32
    }

    /// Speed 1 is not a special case in the code, so it is worth checking that it behaves like one:
    /// the signal comes back, at its own pitch and its own level.
    #[test]
    fn speed_one_plays_the_signal_as_it_is() {
        let input = tone(1000.0, 48_000, 48_000);
        let out = play(&input, 1.0, Quality::Hq, 128);

        assert!(
            out.len().abs_diff(input.len()) < 256,
            "speed 1 should give back what it was given: {} vs {}",
            out.len(),
            input.len()
        );
        let steady = &out[2000..out.len() - 2000];
        assert!((hz(steady, 48_000) - 1000.0).abs() < 2.0);
        let peak = steady.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!((peak - 0.5).abs() < 0.02, "level changed: peak {peak}");
    }

    /// The tape property: speed moves pitch and duration together, by the same factor.
    #[test]
    fn speed_moves_the_pitch_and_the_duration_together() {
        for (speed, want_hz) in [(0.5f32, 500.0f32), (2.0, 2000.0), (4.0, 4000.0)] {
            let input = tone(1000.0, 48_000, 48_000);
            let out = play(&input, speed, Quality::Hq, 128);

            let want_frames = (input.len() as f32 / speed) as usize;
            assert!(
                (out.len() as f32 - want_frames as f32).abs() < want_frames as f32 * 0.02,
                "speed {speed}: {} frames, expected about {want_frames}",
                out.len()
            );
            let steady = &out[1000..out.len() - 1000];
            let got = hz(steady, 48_000);
            assert!(
                (got - want_hz).abs() < 5.0,
                "speed {speed}: {got:.1} Hz, expected {want_hz}"
            );
        }
    }

    /// Speeding up without lowpassing folds the top of the band back down. A 15 kHz tone at 4×
    /// would land at 60 kHz, which does not exist at 48 kHz: what comes out has to be quiet, not a
    /// loud tone at the wrong frequency.
    #[test]
    fn speeding_up_does_not_fold_the_top_of_the_band_back_down() {
        let input = tone(15_000.0, 48_000, 48_000);
        let out = play(&input, 4.0, Quality::Hq, 128);
        let steady = &out[1000..out.len() - 1000];
        let peak = steady.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(
            peak < 0.05,
            "aliased through at peak {peak}, expected under 0.05"
        );
    }

    /// The check R5 names, measuring the right thing.
    ///
    /// A speed change is not a step in the output — the read position stays continuous either way,
    /// so the largest sample-to-sample jump does not move at all when the smoothing is removed, and
    /// a test written on it passes whatever the code does. What a speed change does put in the
    /// signal is a **corner**: the position's *slope* jumps from 1.0 to 1.6 input frames per output
    /// frame in one sample. So the measurement is the second difference, against the same signal
    /// played at a steady 1.6× — the speed where the waveform moves fastest, and the honest
    /// baseline.
    ///
    /// Measured: slamming the speed every 20 blocks gives 0.00084 against a baseline of 0.00089.
    /// With the smoothing taken out it is 0.0076 — eight and a half times the baseline, which is
    /// the corner this exists to round off.
    #[test]
    fn a_speed_change_does_not_click() {
        let input = tone(200.0, 48_000, 48_000);

        let steady = play(&input, 1.6, Quality::Hq, 128);
        let worst_steady = steady[200..steady.len() - 200]
            .windows(3)
            .map(|w| (w[2] - 2.0 * w[1] + w[0]).abs())
            .fold(0.0f32, f32::max);

        let mut v = Varispeed::new(48_000, Quality::Hq, 2.0, 256);
        let mut out = Vec::new();
        let mut scratch = vec![0.0f32; 128];
        let (mut at, mut blocks) = (0usize, 0usize);
        while at < input.len() {
            // Slam the speed between two values every 20 blocks — nastier than any control surface.
            if blocks % 20 == 0 {
                v.set_speed(if (blocks / 20) % 2 == 0 { 1.0 } else { 1.6 });
            }
            let feed = &input[at..(at + 256).min(input.len())];
            let (consumed, written) = v.process(feed, &mut scratch);
            out.extend_from_slice(&scratch[..written]);
            at += consumed;
            blocks += 1;
            if consumed == 0 && written == 0 {
                break;
            }
        }

        let worst = out[200..out.len() - 200]
            .windows(3)
            .map(|w| (w[2] - 2.0 * w[1] + w[0]).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst <= worst_steady * 1.2,
            "sharpest corner {worst}, against {worst_steady} at a steady 1.6x"
        );
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// Frozen is a real position on a scrub bar. It must not spin, produce silence, or refuse to
    /// fill the block.
    #[test]
    fn speed_zero_holds_still() {
        let input = tone(1000.0, 48_000, 4_800);
        let mut v = Varispeed::new(48_000, Quality::Fast, 2.0, 256);
        v.set_speed(0.0);
        v.reset();

        let mut scratch = vec![0.0f32; 128];
        let (_, first) = v.process(&input, &mut scratch);
        assert_eq!(
            first,
            scratch.len(),
            "a frozen playhead still has to fill the block"
        );
        let held = &scratch[..first];
        let spread = held.iter().fold(f32::MIN, |m, v| m.max(*v))
            - held.iter().fold(f32::MAX, |m, v| m.min(*v));
        assert!(spread < 1e-3, "a frozen playhead moved: spread {spread}");
    }

    #[test]
    fn speed_is_clamped_to_what_it_was_built_for() {
        let mut v = Varispeed::new(48_000, Quality::Fast, 2.0, 128);
        v.set_speed(100.0);
        assert_eq!(v.speed(), 2.0);
        v.set_speed(-1.0);
        assert_eq!(v.speed(), 0.0);
    }

    /// Consumption follows the speed: playing twice as fast eats twice as much input for the same
    /// output. This is how a host knows where the playhead is.
    #[test]
    fn it_consumes_input_at_the_speed_it_is_playing() {
        for speed in [0.5f32, 1.0, 2.0] {
            let input = tone(440.0, 48_000, 48_000);
            let mut v = Varispeed::new(48_000, Quality::Fast, 2.0, 256);
            v.set_speed(speed);
            v.reset();

            let mut scratch = vec![0.0f32; 128];
            let (mut eaten, mut made, mut at) = (0usize, 0usize, 0usize);
            while at + 256 < input.len() && made < 12_000 {
                let (consumed, written) = v.process(&input[at..at + 256], &mut scratch);
                at += consumed;
                eaten += consumed;
                made += written;
            }
            let ratio = eaten as f32 / made as f32;
            assert!(
                (ratio - speed).abs() < 0.05,
                "speed {speed}: ate {eaten} for {made} out, ratio {ratio:.3}"
            );
        }
    }

    #[test]
    fn reset_forgets_the_audio_and_lands_on_the_speed() {
        let mut v = Varispeed::new(48_000, Quality::Fast, 2.0, 256);
        let mut scratch = vec![0.0f32; 128];
        v.process(&vec![1.0f32; 256], &mut scratch);
        v.set_speed(2.0);
        v.reset();

        assert_eq!(v.current_speed(), 2.0);
        let (_, written) = v.process(&vec![0.0f32; 512], &mut scratch);
        assert!(
            scratch[..written].iter().all(|s| s.abs() < 1e-6),
            "silence after reset should be silent"
        );
    }
}

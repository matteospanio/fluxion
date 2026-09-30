//! Modulated delay/phase effects: [`chorus`], [`flanger`], and [`phaser`].
//!
//! Each is a single-voice, length-preserving, fs-preserving kernel (kept minimally correct rather
//! than feature-maximal — one LFO, one voice). Chorus and flanger use a linearly-interpolated
//! fractional delay whose length is swept by a low-frequency oscillator; the phaser sweeps a cascade
//! of first-order all-pass stages. Flanger and phaser add a feedback path (bounded `< 1` for BIBO
//! stability); chorus is purely feed-forward.
//!
//! # One step, two engines
//!
//! Each effect is a small state machine — [`Chorus`], [`Flanger`], [`Phaser`] — designed once in
//! `new` and advanced one frame at a time by `step`. The offline functions run it over the whole
//! signal and the realtime `RtGraph` nodes run it over each block, the same arrangement as
//! [`CompandCoeffs`](crate::CompandCoeffs). That is all ROADMAP L5's parity rests on: there is one
//! computation, so a stream cannot disagree with the render however its blocks fall.
//!
//! Everything a block boundary could break lives in the state — the frame the LFO is at, the delay
//! line and its cursor, the all-pass memories, the feedback sample — and the delay line is sized in
//! `new` for the longest delay the settings can reach, so `step` never allocates.
//!
//! ponytail: single-voice, raised-cosine LFO only — multi-voice/stereo-spread chorus and
//! triangle/adjustable LFO shapes are a later refinement.

use std::f32::consts::TAU;

/// The LFO the modulated effects share: a raised cosine that is 0 on frame 0, reaches 1 half a
/// cycle later and comes back, `rate_hz` times a second.
///
/// It is the curve of [`Curve::lfo`](fluxion_core::automation::Curve::lfo) but not its time base.
/// A compiled automation curve rounds its period to whole frames, which is what lets a seek land on
/// exactly the phase a render had there — and which would retune tremolo, whose rate runs to 20 kHz:
/// a 2.4-frame period at 48 kHz rounds to 2. Here the phase is computed from the absolute frame
/// number in `f64`, so the rate stays exact and a frame still has one value however the frames
/// were batched.
///
/// The `f64` is not decoration. The phase used to be `w·n` in `f32`, where the product loses the
/// fractional part of the cycle as `n` grows: by 2^24 frames (5.8 minutes at 48 kHz) a 1 kHz
/// tremolo's phase was only good to an eighth of a radian, and past that `n` itself is no longer
/// representable. A live session runs for hours, so the realtime lowering could not inherit it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lfo {
    cycles_per_frame: f64,
}

impl Lfo {
    /// An LFO at `rate_hz` for sample rate `fs`. A rate of 0 holds at 0.
    pub fn new(rate_hz: f32, fs: u32) -> Lfo {
        Lfo {
            cycles_per_frame: f64::from(rate_hz) / f64::from(fs.max(1)),
        }
    }

    /// The value at absolute frame `n`, in `[0, 1]`.
    #[inline]
    pub fn at(&self, n: u64) -> f32 {
        let cycle = (n as f64 * self.cycles_per_frame).fract() as f32;
        0.5 - 0.5 * (TAU * cycle).cos()
    }
}

/// A delay line read at a delay the LFO sweeps between `base` and `base + depth` frames, linearly
/// interpolated — what [`Chorus`] and [`Flanger`] have in common.
#[derive(Clone, Debug)]
struct SweptDelay {
    lfo: Lfo,
    base: f32,
    depth: f32,
    /// The history, newest at `ring[w]`. Long enough for the longest delay the sweep reaches plus
    /// the one extra frame the interpolation reads behind it.
    ring: Vec<f32>,
    w: usize,
    /// The absolute frame being computed — the LFO's position.
    n: u64,
}

impl SweptDelay {
    fn new(rate_hz: f32, depth_s: f32, delay_s: f32, fs: u32) -> SweptDelay {
        let base = delay_s.max(0.0) * fs as f32;
        let depth = depth_s.max(0.0) * fs as f32;
        // `tap` reads `floor(d)` and `floor(d) + 1` frames back, and `d` never exceeds
        // `base + depth` — the LFO tops out at exactly 1, and f32 rounding is monotone.
        SweptDelay {
            lfo: Lfo::new(rate_hz, fs),
            base,
            depth,
            ring: vec![0.0; (base + depth) as usize + 2],
            w: 0,
            n: 0,
        }
    }

    /// The history `d` frames back from the newest entry, for this frame's `d`:
    /// `(1 − frac)·h[n − i] + frac·h[n − i − 1]` with `d = i + frac`. Frames before the start
    /// read as silence, because that is what the ring was filled with.
    ///
    /// ponytail: `d` is an `f32` count of frames, so its fraction resolves to one ULP of the delay —
    /// 1/65536 of a frame at 5 ms, 1/256 at the 1 s maximum (48 kHz). Carry the integer and the
    /// fraction separately if a long swept delay ever needs finer.
    #[inline]
    fn tap(&self) -> f32 {
        let d = self.base + self.depth * self.lfo.at(self.n);
        let i = d as usize; // d >= 0, so this is floor(d)
        let frac = d - i as f32;
        (1.0 - frac) * self.back(i) + frac * self.back(i + 1)
    }

    #[inline]
    fn back(&self, k: usize) -> f32 {
        let len = self.ring.len();
        self.ring[if self.w >= k {
            self.w - k
        } else {
            self.w + len - k
        }]
    }

    #[inline]
    fn advance(&mut self) {
        self.w += 1;
        if self.w == self.ring.len() {
            self.w = 0;
        }
        self.n += 1;
    }

    fn reset(&mut self) {
        self.ring.fill(0.0);
        self.w = 0;
        self.n = 0;
    }
}

/// [`chorus`], one frame at a time — the state the realtime chorus node carries between blocks.
#[derive(Clone, Debug)]
pub struct Chorus {
    line: SweptDelay,
    mix: f32,
}

impl Chorus {
    /// Design a chorus and allocate its delay line (`delay + depth` seconds of it).
    pub fn new(rate_hz: f32, depth_s: f32, delay_s: f32, mix: f32, fs: u32) -> Chorus {
        Chorus {
            line: SweptDelay::new(rate_hz, depth_s, delay_s, fs),
            mix,
        }
    }

    /// Advance one frame.
    #[inline]
    pub fn step(&mut self, x: f32) -> f32 {
        self.line.ring[self.line.w] = x;
        let wet = self.line.tap();
        self.line.advance();
        (1.0 - self.mix) * x + self.mix * wet
    }

    /// Back to frame 0 and an empty delay line.
    pub fn reset(&mut self) {
        self.line.reset();
    }
}

/// Chorus: a single LFO-modulated fractional-delay voice blended with the dry signal.
///
/// The delay sweeps in `[delay, delay + depth]` seconds at `rate` Hz; `mix` (0..1) is the wet/dry
/// blend. Feed-forward (no feedback), so it is unconditionally stable. Length-preserving.
pub fn chorus(
    input: &[f32],
    rate_hz: f32,
    depth_s: f32,
    delay_s: f32,
    mix: f32,
    fs: u32,
) -> Vec<f32> {
    let mut c = Chorus::new(rate_hz, depth_s, delay_s, mix, fs);
    input.iter().map(|&x| c.step(x)).collect()
}

/// [`flanger`], one frame at a time — the state the realtime flanger node carries between blocks.
#[derive(Clone, Debug)]
pub struct Flanger {
    /// Holds the loop state `x + feedback·delayed`, not the input.
    line: SweptDelay,
    feedback: f32,
    mix: f32,
}

impl Flanger {
    /// Design a flanger and allocate its delay line (`delay + depth` seconds of it).
    pub fn new(
        rate_hz: f32,
        depth_s: f32,
        delay_s: f32,
        feedback: f32,
        mix: f32,
        fs: u32,
    ) -> Flanger {
        Flanger {
            line: SweptDelay::new(rate_hz, depth_s, delay_s, fs),
            feedback,
            mix,
        }
    }

    /// Advance one frame.
    #[inline]
    pub fn step(&mut self, x: f32) -> f32 {
        // A delay under one frame reads the loop state this frame is computing, which does not
        // exist yet. A loop with no delay in it has no answer, so that frame reads as silence.
        self.line.ring[self.line.w] = 0.0;
        let delayed = self.line.tap();
        self.line.ring[self.line.w] = x + self.feedback * delayed;
        self.line.advance();
        (1.0 - self.mix) * x + self.mix * delayed
    }

    /// Back to frame 0 and an empty delay line.
    pub fn reset(&mut self) {
        self.line.reset();
    }
}

/// Flanger: a short LFO-modulated delay with `feedback`, blended with the dry signal.
///
/// The delay sweeps in `[delay, delay + depth]` seconds at `rate` Hz; `feedback` (`|f| < 1`) recirculates
/// the delayed signal for the characteristic resonant sweep; `mix` (0..1) is the wet/dry blend.
/// Length-preserving.
#[allow(clippy::too_many_arguments)]
pub fn flanger(
    input: &[f32],
    rate_hz: f32,
    depth_s: f32,
    delay_s: f32,
    feedback: f32,
    mix: f32,
    fs: u32,
) -> Vec<f32> {
    let mut f = Flanger::new(rate_hz, depth_s, delay_s, feedback, mix, fs);
    input.iter().map(|&x| f.step(x)).collect()
}

/// All-pass stages in the [`phaser`] cascade.
const STAGES: usize = 4;

/// [`phaser`], one frame at a time — the state the realtime phaser node carries between blocks.
#[derive(Clone, Debug)]
pub struct Phaser {
    lfo: Lfo,
    depth: f32,
    feedback: f32,
    mix: f32,
    /// Per-stage previous input and output.
    ap_x: [f32; STAGES],
    ap_y: [f32; STAGES],
    /// The cascade's last output, recirculated.
    fb: f32,
    n: u64,
}

impl Phaser {
    /// Design a phaser. `depth` is clamped to `0..1`.
    pub fn new(rate_hz: f32, depth: f32, feedback: f32, mix: f32, fs: u32) -> Phaser {
        Phaser {
            lfo: Lfo::new(rate_hz, fs),
            depth: depth.clamp(0.0, 1.0),
            feedback,
            mix,
            ap_x: [0.0; STAGES],
            ap_y: [0.0; STAGES],
            fb: 0.0,
            n: 0,
        }
    }

    /// Advance one frame.
    #[inline]
    pub fn step(&mut self, x: f32) -> f32 {
        // -0.9·depth .. 0.9·depth, so |a| < 1 and every stage stays stable.
        let a = 0.9 * self.depth * (2.0 * self.lfo.at(self.n) - 1.0);
        let mut s = x + self.feedback * self.fb;
        for k in 0..STAGES {
            let y = a * s + self.ap_x[k] - a * self.ap_y[k];
            self.ap_x[k] = s;
            self.ap_y[k] = y;
            s = y;
        }
        self.fb = s;
        self.n += 1;
        (1.0 - self.mix) * x + self.mix * s
    }

    /// Back to frame 0 and silent all-pass memories.
    pub fn reset(&mut self) {
        self.ap_x = [0.0; STAGES];
        self.ap_y = [0.0; STAGES];
        self.fb = 0.0;
        self.n = 0;
    }
}

/// Phaser: an LFO-swept cascade of first-order all-pass stages with `feedback`, blended with the dry
/// signal.
///
/// Four all-pass stages share one swept coefficient `a ∈ [−0.9·depth, 0.9·depth]` at `rate` Hz; each
/// stage is `y[n] = a·x[n] + x[n−1] − a·y[n−1]` (unity magnitude, so the moving phase notches are the
/// audible effect). `feedback` (`|f| < 1`) recirculates the cascade output; `mix` (0..1) is the
/// wet/dry blend. Length-preserving.
pub fn phaser(
    input: &[f32],
    rate_hz: f32,
    depth: f32,
    feedback: f32,
    mix: f32,
    fs: u32,
) -> Vec<f32> {
    let mut p = Phaser::new(rate_hz, depth, feedback, mix, fs);
    input.iter().map(|&x| p.step(x)).collect()
}

#[cfg(test)]
mod tests {
    use super::{Lfo, chorus, flanger, phaser};

    fn sig(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (0.05 * i as f32).sin() + 0.3 * (0.21 * i as f32).sin())
            .collect()
    }

    #[test]
    fn chorus_is_dry_with_zero_delay_and_depth() {
        // delay 0, depth 0 -> the wet tap reads the current sample, so wet == dry for any mix.
        let x = sig(200);
        let y = chorus(&x, 1.5, 0.0, 0.0, 1.0, 48_000);
        assert_eq!(y.len(), x.len());
        for (a, b) in y.iter().zip(&x) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn chorus_taps_a_fixed_delay_when_depth_zero() {
        // depth 0, delay = 10 samples (integer), mix 1 -> pure 10-sample delay.
        let x = sig(300);
        let d = 10usize;
        let y = chorus(&x, 1.5, 0.0, d as f32 / 48_000.0, 1.0, 48_000);
        for i in d..x.len() {
            assert!(
                (y[i] - x[i - d]).abs() < 1e-4,
                "at {i}: {} vs {}",
                y[i],
                x[i - d]
            );
        }
    }

    /// A delay of 10.25 frames is three quarters of the frame 10 back and a quarter of the one
    /// before it — and where the older of the two is before the start, a quarter of silence. The
    /// read is relative to the write cursor, so this holds however far into a stream the frame is.
    /// It used to be computed as `n − d` in `f32`, which rounded the fraction to halves by 2^22
    /// frames (87 s at 48 kHz) and dropped it entirely at 2^23.
    #[test]
    fn a_fractional_delay_interpolates_between_its_neighbours() {
        let x = sig(300);
        let y = chorus(&x, 1.5, 0.0, 10.25 / 48_000.0, 1.0, 48_000);
        assert!(y[..10].iter().all(|&v| v == 0.0));
        assert!(
            (y[10] - 0.75 * x[0]).abs() < 1e-7,
            "{} vs {}",
            y[10],
            0.75 * x[0]
        );
        for n in 11..x.len() {
            let want = 0.75 * x[n - 10] + 0.25 * x[n - 11];
            assert!((y[n] - want).abs() < 1e-6, "at {n}: {} vs {want}", y[n]);
        }
    }

    /// The LFO is exact at any frame, not only near the start. The reference is built from integer
    /// arithmetic — `n·rate mod fs` is the phase as an exact fraction — so it shares no rounding
    /// with the code under test. The frames are 2^24 + 1 (the first one `f32` cannot name) and two
    /// days of 48 kHz audio; the old `f32` phase was out by an eighth of a radian at the first.
    #[test]
    fn the_lfo_is_exact_hours_into_a_stream() {
        let (rate, fs) = (1000u64, 48_000u64);
        let lfo = Lfo::new(rate as f32, fs as u32);
        for n in [0, 1, 12_345, (1 << 24) + 1, 2 * 86_400 * fs + 7] {
            let cycle = ((n * rate) % fs) as f64 / fs as f64;
            let want = 0.5 - 0.5 * (std::f64::consts::TAU * cycle).cos();
            let got = f64::from(lfo.at(n));
            assert!((got - want).abs() < 1e-6, "frame {n}: {got} vs {want}");
        }
    }

    #[test]
    fn flanger_matches_delay_when_feedback_zero() {
        // feedback 0, depth 0, integer delay -> (1-mix)·x + mix·x[n-D].
        let x = sig(400);
        let (d, mix) = (7usize, 0.6f32);
        let y = flanger(&x, 0.5, 0.0, d as f32 / 48_000.0, 0.0, mix, 48_000);
        for i in d..x.len() {
            let want = (1.0 - mix) * x[i] + mix * x[i - d];
            assert!((y[i] - want).abs() < 1e-4, "at {i}: {} vs {want}", y[i]);
        }
    }

    #[test]
    fn flanger_feedback_stays_bounded() {
        let x = sig(4_000);
        let y = flanger(&x, 0.5, 0.002, 0.001, 0.9, 0.5, 48_000);
        assert_eq!(y.len(), x.len());
        assert!(
            y.iter().all(|v| v.is_finite() && v.abs() < 20.0),
            "flanger blew up"
        );
    }

    #[test]
    fn phaser_is_dry_when_mix_zero_and_bounded_otherwise() {
        let x = sig(2_000);
        // mix 0 -> exactly the dry signal.
        let dry = phaser(&x, 0.5, 0.5, 0.5, 0.0, 48_000);
        for (a, b) in dry.iter().zip(&x) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
        // wet output stays finite/bounded (all-pass stages are unity-gain, feedback < 1).
        let wet = phaser(&x, 0.5, 1.0, 0.9, 1.0, 48_000);
        assert!(
            wet.iter().all(|v| v.is_finite() && v.abs() < 20.0),
            "phaser blew up"
        );
    }
}

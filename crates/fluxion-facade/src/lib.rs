//! `fluxion` — differentiable, cross-vendor, framework-agnostic audio DSP with a functional
//! graph API.
//!
//! This is the **facade** crate: it re-exports the public surface and a [`prelude`] of ergonomic
//! node constructors.
//!
//! The Rust quickstart. It is a doctest, so `cargo test` runs it — see `docs/interfaces.md` for
//! the ten-line rule it is held to.
//!
//! <!-- quickstart:start -->
//! ```
//! use fluxion::prelude::*;
//!
//! let chain = highpass(80.0, 4) | gain(0.708);  // `|` = series, `+` = parallel (summed)
//! let dry = Signal::new(48_000, vec![vec![0.0; 480]]);
//! let wet = process(&chain, &dry);
//! assert_eq!(wet.frames(), 480);
//!
//! // The same chain in the text syntax the CLI, Python, C and JS all read.
//! assert_eq!("highpass(80, 4) | gain(0.708)".parse::<Graph>().unwrap(), chain);
//! ```
//! <!-- quickstart:end -->

pub use fluxion_backend::{
    Backend, Certificate, Cpu, Verdict, certify_graph, eval, graph_to_sos, is_differentiable,
    process, process_batch, sos_filter_batch,
};
pub use fluxion_core::{
    FORMAT_VERSION, Graph, Group, LoadError, Op, OpError, OpKind, ParamSpec, ParseError, Signal,
    Unit, fxg,
};

/// The shared chain text syntax — `"highpass(80, 4) | gain(-3dB)".parse::<Graph>()`, and the
/// [`ParseError`] it reports, whose `render` draws a caret under the offending token. The same
/// grammar the CLI's `--chain`, Python's `fluxion.chain()`, C's `fx_chain_from_text` and the
/// browser read. See `docs/chain-syntax.md`.
pub use fluxion_core::parse;

/// "Did you mean …?" — the edit-distance helper behind every name error, exposed so a host built
/// on fluxion can give the same suggestion for its own names.
pub use fluxion_core::suggest;

/// Geometry transforms on a whole [`Signal`] — trim / pad / repeat / silence-trim / resample / speed
/// / remix / channels and the multi-input concat / mix. Deliberately **not** graph ops: they change
/// frame count, channel count, or sample rate, so they run before/after a graph rather than inside it.
pub use fluxion_ops::transform;

/// Sample-rate conversion, one converter reached two ways: [`resample::Resampler`] for blocks
/// (realtime, the worklet) and [`resample::convert`] for a signal you already hold whole.
/// [`transform::ensure_fs`] is the one to reach for on an input path — it skips the work when the
/// signal is already at the project rate.
pub use fluxion_ops::resample;

/// The Jury-triangle stability projection for flat `[b0,b1,b2,a1,a2]·K` coefficient vectors (plan
/// task E8) — clamps every section's `(a1, a2)` strictly inside the stable region. Used by
/// `fluxion import --project-stable` to tame an unstable trained checkpoint before certification.
pub use fluxion_ops::project_stable_flat;

/// Whole-graph differentiation through Burn's autograd (feature `autodiff`): [`diff_process`] lowers
/// a [`Graph`] onto Burn so `loss.backward()` flows a gradient through an entire effect chain, via
/// the same [`eval`]/[`Backend`] surface the CPU executor uses (plan tasks E12 + C1).
#[cfg(feature = "autodiff")]
pub use fluxion_autodiff::graph::diff_process;

/// The real-time freeze / lowering surface (feature `realtime`): [`freeze`] a pure-filter graph to a
/// [`FrozenSos`] plan and [`to_rt_graph`] to lower any realtime-safe graph to an [`RtGraph`].
#[cfg(feature = "realtime")]
pub use fluxion_backend::{freeze, to_rt_graph};
/// The frozen realtime plan (feature `realtime`) — designed coefficients + sample rate, ready for
/// [`SosStream`] with no design at load time.
#[cfg(feature = "realtime")]
pub use fluxion_core::FrozenSos;
/// The real-time engine building blocks (feature `realtime`): the allocation-free block executors
/// [`RtGraph`] / [`RtEngine`] / [`SosStream`] and the click-free [`SmoothedValue`] parameter ramp.
#[cfg(feature = "realtime")]
pub use fluxion_rt::{RtEngine, RtGraph, SmoothedValue, SosStream};

/// Ergonomic node constructors plus the core types.
pub mod prelude {
    pub use fluxion_backend::process;
    pub use fluxion_core::{Graph, Op, OpKind, Signal};

    /// A linear gain node (`y = x * g`).
    pub fn gain(g: f32) -> Graph {
        Graph::op(OpKind::Gain, [g])
    }

    /// A Butterworth low-pass of the given cutoff (Hz) and order.
    pub fn lowpass(cutoff_hz: f32, order: u32) -> Graph {
        Graph::op(OpKind::Lowpass, [cutoff_hz, order as f32])
    }

    /// A Butterworth high-pass of the given cutoff (Hz) and order.
    pub fn highpass(cutoff_hz: f32, order: u32) -> Graph {
        Graph::op(OpKind::Highpass, [cutoff_hz, order as f32])
    }

    /// Peak-normalize to a target linear amplitude.
    pub fn normalize(peak: f32) -> Graph {
        Graph::op(OpKind::Normalize, [peak])
    }

    /// RBJ peaking EQ: `gain` dB around `frequency` Hz with bandwidth `q`.
    pub fn peaking(frequency: f32, gain_db: f32, q: f32) -> Graph {
        Graph::op(OpKind::Peaking, [frequency, gain_db, q])
    }

    /// RBJ low shelf: `gain` dB below `frequency` Hz (bandwidth `q`).
    pub fn lowshelf(frequency: f32, gain_db: f32, q: f32) -> Graph {
        Graph::op(OpKind::LowShelf, [frequency, gain_db, q])
    }

    /// RBJ high shelf: `gain` dB above `frequency` Hz (bandwidth `q`).
    pub fn highshelf(frequency: f32, gain_db: f32, q: f32) -> Graph {
        Graph::op(OpKind::HighShelf, [frequency, gain_db, q])
    }

    /// RBJ notch at `frequency` Hz with bandwidth `q`.
    pub fn notch(frequency: f32, q: f32) -> Graph {
        Graph::op(OpKind::Notch, [frequency, q])
    }

    /// RBJ band-pass (0 dB peak) at `frequency` Hz with bandwidth `q`.
    pub fn bandpass(frequency: f32, q: f32) -> Graph {
        Graph::op(OpKind::Bandpass, [frequency, q])
    }

    /// RBJ all-pass at `frequency` Hz with bandwidth `q`.
    pub fn allpass(frequency: f32, q: f32) -> Graph {
        Graph::op(OpKind::Allpass, [frequency, q])
    }

    /// Single delayed tap: `time` seconds, crossfaded by `mix` (0 = dry, 1 = fully delayed).
    pub fn delay(time: f32, mix: f32) -> Graph {
        Graph::op(OpKind::Delay, [time, mix])
    }

    /// Feedback echo: `time` seconds between repeats, `feedback` decay, `wet` echo level.
    pub fn echo(time: f32, feedback: f32, wet: f32) -> Graph {
        Graph::op(OpKind::Echo, [time, feedback, wet])
    }

    /// A direct-form FIR filter from its tap vector: `y[n] = Σ_k taps[k]·x[n-k]` — the graph form of
    /// a trained/frozen FIR. Composable, freezable (`.fxg`), and realtime-playable (lowers to an
    /// `RtGraph::Fir` node). Panics if `taps` is empty.
    pub fn fir(taps: impl Into<Vec<f32>>) -> Graph {
        Graph::op(OpKind::Fir, taps)
    }

    /// Chebyshev Type I low-pass: `cutoff` Hz, `order`, passband `ripple` dB.
    pub fn cheby1_lowpass(cutoff_hz: f32, order: u32, ripple_db: f32) -> Graph {
        Graph::op(OpKind::Cheby1Lowpass, [cutoff_hz, order as f32, ripple_db])
    }

    /// Chebyshev Type I high-pass: `cutoff` Hz, `order`, passband `ripple` dB.
    pub fn cheby1_highpass(cutoff_hz: f32, order: u32, ripple_db: f32) -> Graph {
        Graph::op(OpKind::Cheby1Highpass, [cutoff_hz, order as f32, ripple_db])
    }

    /// Chebyshev Type II low-pass: `cutoff` = stopband edge Hz, `order`, stopband `atten` dB.
    pub fn cheby2_lowpass(cutoff_hz: f32, order: u32, atten_db: f32) -> Graph {
        Graph::op(OpKind::Cheby2Lowpass, [cutoff_hz, order as f32, atten_db])
    }

    /// Chebyshev Type II high-pass: `cutoff` = stopband edge Hz, `order`, stopband `atten` dB.
    pub fn cheby2_highpass(cutoff_hz: f32, order: u32, atten_db: f32) -> Graph {
        Graph::op(OpKind::Cheby2Highpass, [cutoff_hz, order as f32, atten_db])
    }

    /// Schroeder–Moorer reverb: `room` size, `damping`, wet/dry `mix` (all 0..1).
    pub fn reverb(room: f32, damping: f32, mix: f32) -> Graph {
        Graph::op(OpKind::Reverb, [room, damping, mix])
    }

    /// Amplitude fade: `fadein`/`fadeout` seconds with a `shape` curve (0 = linear, 1 = quarter-sine
    /// [the SoX default], 2 = half-sine).
    pub fn fade(fadein: f32, fadeout: f32, shape: f32) -> Graph {
        Graph::op(OpKind::Fade, [fadein, fadeout, shape])
    }

    /// Tremolo: amplitude LFO at `rate` Hz dipping by `depth` (0..1).
    pub fn tremolo(rate_hz: f32, depth: f32) -> Graph {
        Graph::op(OpKind::Tremolo, [rate_hz, depth])
    }

    /// Overdrive: `gain` dB of drive through a `tanh` soft-clipper with a `colour` (0..1) bias.
    pub fn overdrive(gain_db: f32, colour: f32) -> Graph {
        Graph::op(OpKind::Overdrive, [gain_db, colour])
    }

    /// Feed-forward compressor (compand): `attack`/`release` seconds, `threshold` dBFS, `ratio`,
    /// `knee` dB, `makeup` dB.
    pub fn compand(
        attack_s: f32,
        release_s: f32,
        threshold_db: f32,
        ratio: f32,
        knee_db: f32,
        makeup_db: f32,
    ) -> Graph {
        Graph::op(
            OpKind::Compand,
            [attack_s, release_s, threshold_db, ratio, knee_db, makeup_db],
        )
    }

    /// Per-channel time reversal (no parameters).
    pub fn reverse() -> Graph {
        Graph::op(OpKind::Reverse, [])
    }

    /// A raw second-order section from explicit coefficients `b0 b1 b2 a1 a2` (`a0` normalized to 1).
    pub fn biquad(b0: f32, b1: f32, b2: f32, a1: f32, a2: f32) -> Graph {
        Graph::op(OpKind::Biquad, [b0, b1, b2, a1, a2])
    }

    /// Chorus: LFO-modulated fractional-delay voice — `rate` Hz, `depth` s, `delay` s, `mix` (0..1).
    pub fn chorus(rate_hz: f32, depth_s: f32, delay_s: f32, mix: f32) -> Graph {
        Graph::op(OpKind::Chorus, [rate_hz, depth_s, delay_s, mix])
    }

    /// Flanger: short modulated delay with feedback — `rate` Hz, `depth` s, `delay` s, `feedback`,
    /// `mix` (0..1).
    pub fn flanger(rate_hz: f32, depth_s: f32, delay_s: f32, feedback: f32, mix: f32) -> Graph {
        Graph::op(OpKind::Flanger, [rate_hz, depth_s, delay_s, feedback, mix])
    }

    /// True-peak limiter: holds the reconstructed waveform under `ceiling` dBTP, anticipating by
    /// `lookahead` seconds and recovering over `release` seconds.
    pub fn limiter(ceiling_db: f32, lookahead_s: f32, release_s: f32) -> Graph {
        Graph::op(OpKind::Limiter, [ceiling_db, lookahead_s, release_s])
    }

    /// Loudness normalize to `target` LUFS (BS.1770), holding the true peak under `ceiling` dBTP.
    pub fn loudnorm(target_lufs: f32, ceiling_db: f32) -> Graph {
        Graph::op(OpKind::Loudnorm, [target_lufs, ceiling_db])
    }

    /// Pitch-shift by `cents`, keeping the length. 1200 cents is an octave up.
    pub fn pitchshift(cents: f32) -> Graph {
        Graph::op(OpKind::PitchShift, [cents])
    }

    /// Phaser: LFO-swept all-pass cascade with feedback — `rate` Hz, `depth` (0..1), `feedback`,
    /// `mix` (0..1).
    pub fn phaser(rate_hz: f32, depth: f32, feedback: f32, mix: f32) -> Graph {
        Graph::op(OpKind::Phaser, [rate_hz, depth, feedback, mix])
    }
}

#[cfg(test)]
mod tests {
    // The prelude glob brings the constructors plus `Graph` / `Op` / `OpKind` into scope.
    use super::prelude::*;

    /// Assert a prelude-built graph is a single leaf op of the expected kind and parameters.
    fn assert_op(graph: Graph, kind: OpKind, params: &[f32]) {
        match graph {
            Graph::Op(op) => {
                assert_eq!(op.kind, kind, "op kind");
                assert_eq!(op.params, params, "op params for {}", kind.name());
            }
            other => panic!("expected a single op, got {other}"),
        }
    }

    #[test]
    fn prelude_constructors_build_the_right_ops() {
        assert_op(gain(0.5), OpKind::Gain, &[0.5]);
        assert_op(lowpass(800.0, 2), OpKind::Lowpass, &[800.0, 2.0]);
        assert_op(highpass(80.0, 4), OpKind::Highpass, &[80.0, 4.0]);
        assert_op(reverb(0.5, 0.3, 0.3), OpKind::Reverb, &[0.5, 0.3, 0.3]);
        assert_op(
            cheby2_lowpass(1000.0, 4, 40.0),
            OpKind::Cheby2Lowpass,
            &[1000.0, 4.0, 40.0],
        );
        assert_op(
            peaking(1_000.0, 6.0, 1.5),
            OpKind::Peaking,
            &[1_000.0, 6.0, 1.5],
        );
        assert_op(fade(0.1, 0.2, 1.0), OpKind::Fade, &[0.1, 0.2, 1.0]);
        assert_op(tremolo(5.0, 0.5), OpKind::Tremolo, &[5.0, 0.5]);
        assert_op(overdrive(20.0, 0.2), OpKind::Overdrive, &[20.0, 0.2]);
        assert_op(
            compand(0.01, 0.1, -20.0, 4.0, 6.0, 0.0),
            OpKind::Compand,
            &[0.01, 0.1, -20.0, 4.0, 6.0, 0.0],
        );
        assert_op(reverse(), OpKind::Reverse, &[]);
        assert_op(
            biquad(0.5, -0.2, 0.1, -0.3, 0.05),
            OpKind::Biquad,
            &[0.5, -0.2, 0.1, -0.3, 0.05],
        );
        assert_op(
            chorus(1.5, 0.002, 0.025, 0.5),
            OpKind::Chorus,
            &[1.5, 0.002, 0.025, 0.5],
        );
        assert_op(
            flanger(0.5, 0.002, 0.001, 0.5, 0.5),
            OpKind::Flanger,
            &[0.5, 0.002, 0.001, 0.5, 0.5],
        );
        assert_op(
            phaser(0.5, 0.5, 0.5, 0.5),
            OpKind::Phaser,
            &[0.5, 0.5, 0.5, 0.5],
        );
    }
}

//! The typed op catalog ([`OpKind`]) and concrete op instances ([`Op`]).

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::param::{ParamSpec, Unit};

/// The kind of a DSP leaf op. Each kind has a fixed parameter schema ([`OpKind::params`]).
///
/// This is the typed replacement for the earlier string-keyed placeholder: it makes the IR
/// self-describing (names, units, defaults, bounds) for validation, the CLI parser, and lowering.
/// New ops are added here as the `fluxion-ops` crate grows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum OpKind {
    /// Linear gain, `y = x * gain`.
    Gain,
    /// Butterworth low-pass filter (`cutoff` Hz, integer `order`). The `lowpass`/`lowpass_n` node.
    Lowpass,
    /// Butterworth high-pass filter (`cutoff` Hz, integer `order`). The `highpass`/`highpass_n` node.
    Highpass,
    /// Peak normalization to a target linear `peak`.
    Normalize,
    /// RBJ peaking EQ: `gain` dB around `frequency` with bandwidth `q`.
    Peaking,
    /// RBJ low shelf: `gain` dB below `frequency` (bandwidth `q`).
    LowShelf,
    /// RBJ high shelf: `gain` dB above `frequency` (bandwidth `q`).
    HighShelf,
    /// RBJ notch at `frequency` with bandwidth `q`.
    Notch,
    /// RBJ band-pass (0 dB peak) at `frequency` with bandwidth `q`.
    Bandpass,
    /// RBJ all-pass at `frequency` with bandwidth `q`.
    Allpass,
    /// Single delayed tap crossfaded with the dry signal (`time` s, `mix`).
    Delay,
    /// Feedback echo: `wet` repeating echoes spaced `time` s apart with `feedback`.
    Echo,
    /// Chebyshev Type I low-pass (`cutoff` Hz, `order`, passband `ripple` dB).
    Cheby1Lowpass,
    /// Chebyshev Type I high-pass (`cutoff` Hz, `order`, passband `ripple` dB).
    Cheby1Highpass,
    /// Chebyshev Type II low-pass (`cutoff` = stopband edge Hz, `order`, stopband `atten` dB).
    Cheby2Lowpass,
    /// Chebyshev Type II high-pass (`cutoff` = stopband edge Hz, `order`, stopband `atten` dB).
    Cheby2Highpass,
    /// Schroeder–Moorer reverb (`room` size, `damping`, wet/dry `mix`).
    Reverb,
    /// Direct-form FIR filter: `y[n] = Σ_k taps[k]·x[n-k]`. **Variadic** — its parameters are the tap
    /// vector itself (≥ 1), the realtime/graph form of a trained/frozen FIR (see [`OpKind::is_variadic`]).
    Fir,
    /// Amplitude fade: `fadein` seconds ramping in, `fadeout` seconds ramping out, with a `shape`
    /// curve (0 = linear, 1 = quarter-sine [the SoX default], 2 = half-sine). Length-preserving.
    Fade,
    /// Tremolo: amplitude LFO at `rate` Hz dipping by `depth` (0..1). Length-preserving.
    Tremolo,
    /// Overdrive: `gain` dB of drive through a `tanh` soft-clipper with a `colour` asymmetry bias.
    /// Nonlinear (not differentiable here).
    Overdrive,
    /// Feed-forward compressor / expander (compand): one-pole peak-envelope follower (`attack`,
    /// `release` s) driving a soft-knee gain computer (`threshold` dBFS, `ratio`, `knee` dB, `makeup`
    /// dB). Stateful per-channel — realtime-playable.
    Compand,
    /// Per-channel time reversal (no parameters). Length-preserving, but **not** realtime (it needs
    /// the whole signal).
    Reverse,
    /// A raw second-order section from explicit coefficients `b0 b1 b2 a1 a2` (`a0` normalized to 1).
    /// Reuses the biquad/SOS machinery, so it is differentiable / freezable / realtime like the
    /// designed filters.
    Biquad,
    /// Chorus: an LFO-modulated fractional-delay voice (`rate` Hz, `depth` s, `delay` s) blended by
    /// `mix`. Feed-forward (no feedback). Length-preserving.
    Chorus,
    /// Flanger: a short LFO-modulated delay (`rate` Hz, `depth` s, `delay` s) with `feedback`, blended
    /// by `mix`. Length-preserving.
    Flanger,
    /// Phaser: an LFO-swept cascade of first-order all-pass stages (`rate` Hz, `depth`) with
    /// `feedback`, blended by `mix`. Length-preserving.
    Phaser,
}

// Static parameter tables — one per kind.
static GAIN_PARAMS: [ParamSpec; 1] = [ParamSpec::new(
    "gain",
    Unit::Linear,
    1.0,
    f32::NEG_INFINITY,
    f32::INFINITY,
)];
static LOWPASS_PARAMS: [ParamSpec; 2] = [
    ParamSpec::new("cutoff", Unit::Hz, 1000.0, 0.0, f32::INFINITY),
    ParamSpec::new("order", Unit::Linear, 2.0, 1.0, 16.0),
];
static HIGHPASS_PARAMS: [ParamSpec; 2] = [
    ParamSpec::new("cutoff", Unit::Hz, 1000.0, 0.0, f32::INFINITY),
    ParamSpec::new("order", Unit::Linear, 2.0, 1.0, 16.0),
];
static NORMALIZE_PARAMS: [ParamSpec; 1] = [ParamSpec::new(
    "peak",
    Unit::Linear,
    1.0,
    0.0,
    f32::INFINITY,
)];
// frequency + gain + q (peaking and both shelves share this schema).
static PEQ_PARAMS: [ParamSpec; 3] = [
    ParamSpec::new("frequency", Unit::Hz, 1000.0, 0.0, f32::INFINITY),
    ParamSpec::new("gain", Unit::Db, 0.0, f32::NEG_INFINITY, f32::INFINITY),
    ParamSpec::new("q", Unit::Q, 0.707, 1e-3, 1000.0),
];
// frequency + q (notch, bandpass, allpass share this schema).
static FQ_PARAMS: [ParamSpec; 2] = [
    ParamSpec::new("frequency", Unit::Hz, 1000.0, 0.0, f32::INFINITY),
    ParamSpec::new("q", Unit::Q, 0.707, 1e-3, 1000.0),
];
static DELAY_PARAMS: [ParamSpec; 2] = [
    ParamSpec::new("time", Unit::Seconds, 0.25, 0.0, 60.0),
    ParamSpec::new("mix", Unit::Linear, 0.5, 0.0, 1.0),
];
static ECHO_PARAMS: [ParamSpec; 3] = [
    ParamSpec::new("time", Unit::Seconds, 0.25, 0.0, 60.0),
    ParamSpec::new("feedback", Unit::Linear, 0.3, 0.0, 0.99),
    ParamSpec::new("wet", Unit::Linear, 0.5, 0.0, 1.0),
];
// cutoff + order + ripple (Chebyshev I low/high-pass share this schema).
static CHEBY1_PARAMS: [ParamSpec; 3] = [
    ParamSpec::new("cutoff", Unit::Hz, 1000.0, 0.0, f32::INFINITY),
    ParamSpec::new("order", Unit::Linear, 4.0, 1.0, 16.0),
    ParamSpec::new("ripple", Unit::Db, 1.0, 1e-2, 12.0),
];
// cutoff (stopband edge) + order + stopband attenuation (Chebyshev II low/high-pass).
static CHEBY2_PARAMS: [ParamSpec; 3] = [
    ParamSpec::new("cutoff", Unit::Hz, 1000.0, 0.0, f32::INFINITY),
    ParamSpec::new("order", Unit::Linear, 4.0, 1.0, 16.0),
    ParamSpec::new("atten", Unit::Db, 40.0, 10.0, 120.0),
];
static REVERB_PARAMS: [ParamSpec; 3] = [
    ParamSpec::new("room", Unit::Linear, 0.5, 0.0, 1.0),
    ParamSpec::new("damping", Unit::Linear, 0.3, 0.0, 1.0),
    ParamSpec::new("mix", Unit::Linear, 0.3, 0.0, 1.0),
];
// The prototype for one FIR tap. `Fir` is variadic: its parameters are a `≥1`-length vector of
// these (any finite value), so the arity check is "at least one" rather than a fixed count.
static FIR_PARAMS: [ParamSpec; 1] = [ParamSpec::new(
    "tap",
    Unit::Linear,
    1.0,
    f32::NEG_INFINITY,
    f32::INFINITY,
)];
static FADE_PARAMS: [ParamSpec; 3] = [
    ParamSpec::new("fadein", Unit::Seconds, 0.0, 0.0, 3600.0),
    ParamSpec::new("fadeout", Unit::Seconds, 0.0, 0.0, 3600.0),
    // 0 = linear, 1 = quarter-sine (SoX default), 2 = half-sine.
    ParamSpec::new("shape", Unit::Linear, 1.0, 0.0, 2.0),
];
static TREMOLO_PARAMS: [ParamSpec; 2] = [
    ParamSpec::new("rate", Unit::Hz, 5.0, 0.0, 20_000.0),
    ParamSpec::new("depth", Unit::Linear, 0.5, 0.0, 1.0),
];
static OVERDRIVE_PARAMS: [ParamSpec; 2] = [
    ParamSpec::new("gain", Unit::Db, 20.0, 0.0, 100.0),
    ParamSpec::new("colour", Unit::Linear, 0.2, 0.0, 1.0),
];
static COMPAND_PARAMS: [ParamSpec; 6] = [
    ParamSpec::new("attack", Unit::Seconds, 0.01, 0.0, 10.0),
    ParamSpec::new("release", Unit::Seconds, 0.1, 0.0, 10.0),
    ParamSpec::new("threshold", Unit::Db, -20.0, -120.0, 0.0),
    ParamSpec::new("ratio", Unit::Linear, 4.0, 1.0, 100.0),
    ParamSpec::new("knee", Unit::Db, 6.0, 0.0, 48.0),
    ParamSpec::new("makeup", Unit::Db, 0.0, -48.0, 48.0),
];
// Reverse takes no parameters.
static NO_PARAMS: [ParamSpec; 0] = [];
// Raw biquad section: five coefficients, any finite value (`a0` is normalized to 1).
static BIQUAD_PARAMS: [ParamSpec; 5] = [
    ParamSpec::new("b0", Unit::Linear, 1.0, f32::NEG_INFINITY, f32::INFINITY),
    ParamSpec::new("b1", Unit::Linear, 0.0, f32::NEG_INFINITY, f32::INFINITY),
    ParamSpec::new("b2", Unit::Linear, 0.0, f32::NEG_INFINITY, f32::INFINITY),
    ParamSpec::new("a1", Unit::Linear, 0.0, f32::NEG_INFINITY, f32::INFINITY),
    ParamSpec::new("a2", Unit::Linear, 0.0, f32::NEG_INFINITY, f32::INFINITY),
];
static CHORUS_PARAMS: [ParamSpec; 4] = [
    ParamSpec::new("rate", Unit::Hz, 1.5, 0.0, 100.0),
    ParamSpec::new("depth", Unit::Seconds, 0.002, 0.0, 1.0),
    ParamSpec::new("delay", Unit::Seconds, 0.025, 0.0, 1.0),
    ParamSpec::new("mix", Unit::Linear, 0.5, 0.0, 1.0),
];
static FLANGER_PARAMS: [ParamSpec; 5] = [
    ParamSpec::new("rate", Unit::Hz, 0.5, 0.0, 100.0),
    ParamSpec::new("depth", Unit::Seconds, 0.002, 0.0, 1.0),
    ParamSpec::new("delay", Unit::Seconds, 0.001, 0.0, 1.0),
    ParamSpec::new("feedback", Unit::Linear, 0.5, -0.95, 0.95),
    ParamSpec::new("mix", Unit::Linear, 0.5, 0.0, 1.0),
];
static PHASER_PARAMS: [ParamSpec; 4] = [
    ParamSpec::new("rate", Unit::Hz, 0.5, 0.0, 100.0),
    ParamSpec::new("depth", Unit::Linear, 0.5, 0.0, 1.0),
    ParamSpec::new("feedback", Unit::Linear, 0.5, -0.95, 0.95),
    ParamSpec::new("mix", Unit::Linear, 0.5, 0.0, 1.0),
];

impl OpKind {
    /// Stable identifier used in the DSL / CLI / `.fxg`, e.g. `"lowpass"`.
    pub fn name(self) -> &'static str {
        match self {
            OpKind::Gain => "gain",
            OpKind::Lowpass => "lowpass",
            OpKind::Highpass => "highpass",
            OpKind::Normalize => "normalize",
            OpKind::Peaking => "peaking",
            OpKind::LowShelf => "lowshelf",
            OpKind::HighShelf => "highshelf",
            OpKind::Notch => "notch",
            OpKind::Bandpass => "bandpass",
            OpKind::Allpass => "allpass",
            OpKind::Delay => "delay",
            OpKind::Echo => "echo",
            OpKind::Cheby1Lowpass => "cheby1low",
            OpKind::Cheby1Highpass => "cheby1high",
            OpKind::Cheby2Lowpass => "cheby2low",
            OpKind::Cheby2Highpass => "cheby2high",
            OpKind::Reverb => "reverb",
            OpKind::Fir => "fir",
            OpKind::Fade => "fade",
            OpKind::Tremolo => "tremolo",
            OpKind::Overdrive => "overdrive",
            OpKind::Compand => "compand",
            OpKind::Reverse => "reverse",
            OpKind::Biquad => "biquad",
            OpKind::Chorus => "chorus",
            OpKind::Flanger => "flanger",
            OpKind::Phaser => "phaser",
        }
    }

    /// Whether this op is **variadic** — its parameters are a variable-length list of one repeated
    /// [`params`](OpKind::params) spec (`≥ 1` entries), rather than a fixed positional tuple. Only
    /// [`OpKind::Fir`] (the tap vector) is variadic today; [`Op::new`] validates it as "at least one,
    /// each within the single spec's bounds".
    pub fn is_variadic(self) -> bool {
        matches!(self, OpKind::Fir)
    }

    /// The parameter schema for this op, in positional order.
    pub fn params(self) -> &'static [ParamSpec] {
        match self {
            OpKind::Gain => &GAIN_PARAMS,
            OpKind::Lowpass => &LOWPASS_PARAMS,
            OpKind::Highpass => &HIGHPASS_PARAMS,
            OpKind::Normalize => &NORMALIZE_PARAMS,
            OpKind::Peaking | OpKind::LowShelf | OpKind::HighShelf => &PEQ_PARAMS,
            OpKind::Notch | OpKind::Bandpass | OpKind::Allpass => &FQ_PARAMS,
            OpKind::Delay => &DELAY_PARAMS,
            OpKind::Echo => &ECHO_PARAMS,
            OpKind::Cheby1Lowpass | OpKind::Cheby1Highpass => &CHEBY1_PARAMS,
            OpKind::Cheby2Lowpass | OpKind::Cheby2Highpass => &CHEBY2_PARAMS,
            OpKind::Reverb => &REVERB_PARAMS,
            OpKind::Fir => &FIR_PARAMS,
            OpKind::Fade => &FADE_PARAMS,
            OpKind::Tremolo => &TREMOLO_PARAMS,
            OpKind::Overdrive => &OVERDRIVE_PARAMS,
            OpKind::Compand => &COMPAND_PARAMS,
            OpKind::Reverse => &NO_PARAMS,
            OpKind::Biquad => &BIQUAD_PARAMS,
            OpKind::Chorus => &CHORUS_PARAMS,
            OpKind::Flanger => &FLANGER_PARAMS,
            OpKind::Phaser => &PHASER_PARAMS,
        }
    }

    /// Look up a kind by its DSL name (inverse of [`OpKind::name`]).
    pub fn from_name(name: &str) -> Option<OpKind> {
        OpKind::all().iter().copied().find(|k| k.name() == name)
    }

    /// Every op kind, for enumeration (CLI help, validation).
    pub fn all() -> &'static [OpKind] {
        &[
            OpKind::Gain,
            OpKind::Lowpass,
            OpKind::Highpass,
            OpKind::Normalize,
            OpKind::Peaking,
            OpKind::LowShelf,
            OpKind::HighShelf,
            OpKind::Notch,
            OpKind::Bandpass,
            OpKind::Allpass,
            OpKind::Delay,
            OpKind::Echo,
            OpKind::Cheby1Lowpass,
            OpKind::Cheby1Highpass,
            OpKind::Cheby2Lowpass,
            OpKind::Cheby2Highpass,
            OpKind::Reverb,
            OpKind::Fir,
            OpKind::Fade,
            OpKind::Tremolo,
            OpKind::Overdrive,
            OpKind::Compand,
            OpKind::Reverse,
            OpKind::Biquad,
            OpKind::Chorus,
            OpKind::Flanger,
            OpKind::Phaser,
        ]
    }

    /// The default parameter vector (one entry per [`ParamSpec`]).
    pub fn defaults(self) -> Vec<f32> {
        self.params().iter().map(|p| p.default).collect()
    }
}

/// Error from constructing or validating an [`Op`].
#[derive(Clone, Debug, PartialEq)]
pub enum OpError {
    /// Wrong number of parameters for the kind.
    Arity {
        /// The op whose arity was violated.
        kind: OpKind,
        /// Number of parameters the kind expects.
        expected: usize,
        /// Number of parameters supplied.
        got: usize,
    },
    /// A parameter was NaN or outside its static bounds.
    OutOfRange {
        /// The op whose parameter was invalid.
        kind: OpKind,
        /// Name of the offending parameter.
        param: &'static str,
        /// The supplied value.
        value: f32,
        /// Inclusive lower bound.
        min: f32,
        /// Inclusive upper bound.
        max: f32,
    },
}

impl fmt::Display for OpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpError::Arity {
                kind,
                expected,
                got,
            } => write!(
                f,
                "op '{}' expects {expected} parameter(s), got {got}",
                kind.name()
            ),
            OpError::OutOfRange {
                kind,
                param,
                value,
                min,
                max,
            } => write!(
                f,
                "op '{}' parameter '{param}' = {value} is out of range [{min}, {max}]",
                kind.name()
            ),
        }
    }
}

impl std::error::Error for OpError {}

/// A concrete leaf op: an [`OpKind`] plus its positional parameter values.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Op {
    /// What this op does.
    pub kind: OpKind,
    /// Parameter values, in the order of [`OpKind::params`].
    pub params: Vec<f32>,
}

impl Op {
    /// Validating constructor: checks arity and that each value is non-NaN and within bounds.
    ///
    /// ```
    /// use fluxion_core::{Op, OpKind};
    /// assert!(Op::new(OpKind::Lowpass, [800.0, 4.0]).is_ok());
    /// assert!(Op::new(OpKind::Lowpass, [800.0]).is_err());  // wrong arity
    /// assert!(Op::new(OpKind::Lowpass, [-1.0, 2.0]).is_err()); // cutoff out of range
    /// ```
    pub fn new(kind: OpKind, params: impl Into<Vec<f32>>) -> Result<Op, OpError> {
        let params = params.into();
        let specs = kind.params();

        if kind.is_variadic() {
            // One repeated spec (`specs[0]`); require at least one value, each within bounds.
            let spec = &specs[0];
            if params.is_empty() {
                return Err(OpError::Arity {
                    kind,
                    expected: 1,
                    got: 0,
                });
            }
            for &v in &params {
                if v.is_nan() || v < spec.min || v > spec.max {
                    return Err(OpError::OutOfRange {
                        kind,
                        param: spec.name,
                        value: v,
                        min: spec.min,
                        max: spec.max,
                    });
                }
            }
            return Ok(Op { kind, params });
        }

        if params.len() != specs.len() {
            return Err(OpError::Arity {
                kind,
                expected: specs.len(),
                got: params.len(),
            });
        }
        for (spec, &v) in specs.iter().zip(&params) {
            if v.is_nan() || v < spec.min || v > spec.max {
                return Err(OpError::OutOfRange {
                    kind,
                    param: spec.name,
                    value: v,
                    min: spec.min,
                    max: spec.max,
                });
            }
        }
        Ok(Op { kind, params })
    }
}

#[cfg(test)]
mod tests {
    use super::{Op, OpKind};

    #[test]
    fn name_roundtrips() {
        for &k in OpKind::all() {
            assert_eq!(OpKind::from_name(k.name()), Some(k));
        }
        assert_eq!(OpKind::from_name("nope"), None);
    }

    #[test]
    fn defaults_match_arity() {
        assert_eq!(OpKind::Lowpass.defaults(), vec![1000.0, 2.0]);
        assert_eq!(OpKind::Gain.defaults(), vec![1.0]);
        assert_eq!(OpKind::Peaking.defaults().len(), 3);
        assert_eq!(OpKind::Notch.defaults().len(), 2);
    }

    #[test]
    fn validation_rejects_bad_arity_and_range() {
        assert!(Op::new(OpKind::Gain, [1.0]).is_ok());
        assert!(Op::new(OpKind::Gain, []).is_err());
        assert!(Op::new(OpKind::Lowpass, [-5.0, 2.0]).is_err());
        assert!(Op::new(OpKind::Lowpass, [1000.0, f32::NAN]).is_err());
        assert!(Op::new(OpKind::Peaking, [1000.0, 6.0, 0.0]).is_err()); // q below min
    }

    #[test]
    fn new_effect_ops_validate() {
        // Reverse is the zero-parameter op.
        assert_eq!(OpKind::Reverse.defaults(), Vec::<f32>::new());
        assert!(Op::new(OpKind::Reverse, []).is_ok());
        assert!(Op::new(OpKind::Reverse, [1.0]).is_err()); // no params allowed

        // Fade: three params, shape bounded 0..2.
        assert_eq!(OpKind::Fade.defaults(), vec![0.0, 0.0, 1.0]);
        assert!(Op::new(OpKind::Fade, [0.1, 0.2, 1.0]).is_ok());
        assert!(Op::new(OpKind::Fade, [0.1, 0.2, 3.0]).is_err()); // shape out of range

        // Compand: six params; ratio must be >= 1.
        assert_eq!(OpKind::Compand.defaults().len(), 6);
        assert!(Op::new(OpKind::Compand, [0.01, 0.1, -20.0, 4.0, 6.0, 0.0]).is_ok());
        assert!(Op::new(OpKind::Compand, [0.01, 0.1, -20.0, 0.5, 6.0, 0.0]).is_err());

        // Biquad: five raw coefficients, any finite value.
        assert_eq!(OpKind::Biquad.defaults(), vec![1.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(Op::new(OpKind::Biquad, [0.5, -0.2, 0.1, -0.3, 0.05]).is_ok());
        assert!(Op::new(OpKind::Biquad, [0.5, 0.0, 0.0, 0.0, f32::NAN]).is_err());

        // Flanger feedback is bounded for BIBO stability.
        assert!(Op::new(OpKind::Flanger, [0.5, 0.002, 0.001, 0.5, 0.5]).is_ok());
        assert!(Op::new(OpKind::Flanger, [0.5, 0.002, 0.001, 1.5, 0.5]).is_err());
    }

    #[test]
    fn fir_is_variadic() {
        assert!(OpKind::Fir.is_variadic());
        assert!(Op::new(OpKind::Fir, [0.1, -0.2, 0.3, 0.05]).is_ok()); // any length ≥ 1
        assert!(Op::new(OpKind::Fir, [1.0]).is_ok());
        assert!(Op::new(OpKind::Fir, []).is_err()); // needs at least one tap
        assert!(Op::new(OpKind::Fir, [0.1, f32::NAN]).is_err()); // taps must be finite
        assert_eq!(OpKind::Fir.defaults(), vec![1.0]); // one identity tap
    }
}

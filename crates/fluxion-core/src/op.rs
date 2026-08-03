//! The typed op catalog ([`OpKind`]) and concrete op instances ([`Op`]).
//!
//! The catalog is declared **once**, in the `ops!` table below: one row per op carrying its doc
//! comment, its Rust variant, its stable text name, its catalogue group, and its parameter schema.
//! Everything an interface needs — [`OpKind::name`], [`OpKind::group`], [`OpKind::params`],
//! [`OpKind::all`] — is generated from that row, so adding an op is a single edit and no two lists
//! can drift apart.
//!
//! The variant identifier is what `.fxg` files store on disk (serde's default enum representation),
//! while [`OpKind::name`] is what the chain text, the CLI, Python and C use. Renaming the text name
//! is therefore safe for saved graphs; renaming a variant is not, and
//! `tests/fxg_compat.rs` is the guard.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::param::{ParamSpec, Unit};

/// Which section of the catalog an op belongs to.
///
/// This is a documentation grouping, not a semantic one: it decides whether an op appears under
/// `fluxion.filter` or `fluxion.effect` in Python, and which section of `docs/ops.md` lists it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Group {
    /// Shapes the spectrum: the designed IIR filters, the raw biquad, and FIR convolution.
    Filter,
    /// Everything else — amplitude, time, modulation, dynamics, space.
    Effect,
}

impl Group {
    /// The lowercase tag used in generated output (`"filter"`, `"effect"`).
    pub fn as_str(self) -> &'static str {
        match self {
            Group::Filter => "filter",
            Group::Effect => "effect",
        }
    }
}

impl fmt::Display for Group {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Declare the op catalog: one row per op, `Variant => "text name", Group, [params];`.
///
/// Generates the [`OpKind`] enum (doc comments and all) plus `name`, `group`, `params` and `all`.
/// Kept private — it is an implementation detail of this module, not public API.
macro_rules! ops {
    ($(
        $(#[doc = $doc:literal])*
        $variant:ident => $dsl:literal, $group:ident, [ $($spec:expr),* $(,)? ];
    )+) => {
        /// The kind of a DSP leaf op. Each kind has a fixed parameter schema ([`OpKind::params`]).
        ///
        /// This makes the IR self-describing (names, units, defaults, bounds) for validation, the
        /// chain-text parser, the CLI, and lowering. New ops are added to the `ops!` table in this
        /// module as the `fluxion-ops` crate grows.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
        #[non_exhaustive]
        pub enum OpKind {
            $( $(#[doc = $doc])* $variant, )+
        }

        impl OpKind {
            /// Stable identifier used in the chain text, the CLI, Python and C — e.g. `"lowpass"`.
            ///
            /// One name per op, on every interface. Not what `.fxg` stores (that is the variant
            /// identifier), so this string can be corrected without breaking saved graphs.
            pub fn name(self) -> &'static str {
                match self { $( OpKind::$variant => $dsl, )+ }
            }

            /// The Rust variant identifier — e.g. `"LowShelf"`.
            ///
            /// This is what `.fxg` stores, and what interfaces that spell ops as classes use as the
            /// class name (`fluxion.filter.LowShelf` in Python). Deriving it from the same table row
            /// as [`name`](OpKind::name) is what keeps the two from drifting.
            pub fn variant(self) -> &'static str {
                match self { $( OpKind::$variant => stringify!($variant), )+ }
            }

            /// The op's documentation, one entry per line of the catalog's doc comment.
            ///
            /// Written once, in the table, and used twice: rustdoc renders it on the variant, and
            /// the interface generator turns it into the Python docstring and the row in
            /// `docs/ops.md`. Lines keep rustdoc's leading space; trim before rendering.
            pub fn doc(self) -> &'static [&'static str] {
                match self { $( OpKind::$variant => &[ $($doc),* ], )+ }
            }

            /// Which catalogue section this op belongs to. See [`Group`].
            pub fn group(self) -> Group {
                match self { $( OpKind::$variant => Group::$group, )+ }
            }

            /// The parameter schema for this op, in positional order.
            pub fn params(self) -> &'static [ParamSpec] {
                match self {
                    $( OpKind::$variant => { const P: &[ParamSpec] = &[ $($spec),* ]; P } )+
                }
            }

            /// Every op kind, in catalog order — for enumeration (CLI help, validation, codegen).
            pub fn all() -> &'static [OpKind] {
                &[ $( OpKind::$variant, )+ ]
            }
        }
    };
}

ops! {
    /// Linear gain, `y = x * gain`.
    Gain => "gain", Effect, [
        ParamSpec::new("gain", Unit::Linear, 1.0, f32::NEG_INFINITY, f32::INFINITY),
    ];

    /// Butterworth low-pass filter (`cutoff` Hz, integer `order`). The `lowpass` node.
    Lowpass => "lowpass", Filter, [
        ParamSpec::new("cutoff", Unit::Hz, 1000.0, 0.0, f32::INFINITY),
        ParamSpec::new("order", Unit::Linear, 2.0, 1.0, 16.0),
    ];

    /// Butterworth high-pass filter (`cutoff` Hz, integer `order`). The `highpass` node.
    Highpass => "highpass", Filter, [
        ParamSpec::new("cutoff", Unit::Hz, 1000.0, 0.0, f32::INFINITY),
        ParamSpec::new("order", Unit::Linear, 2.0, 1.0, 16.0),
    ];

    /// Peak normalization to a target linear `peak`.
    Normalize => "normalize", Effect, [
        ParamSpec::new("peak", Unit::Linear, 1.0, 0.0, f32::INFINITY),
    ];

    /// RBJ peaking EQ: `gain` dB around `frequency` with bandwidth `q`.
    Peaking => "peaking", Filter, [
        ParamSpec::new("frequency", Unit::Hz, 1000.0, 0.0, f32::INFINITY),
        ParamSpec::new("gain", Unit::Db, 0.0, f32::NEG_INFINITY, f32::INFINITY),
        ParamSpec::new("q", Unit::Q, 0.707, 1e-3, 1000.0),
    ];

    /// RBJ low shelf: `gain` dB below `frequency` (bandwidth `q`).
    LowShelf => "lowshelf", Filter, [
        ParamSpec::new("frequency", Unit::Hz, 1000.0, 0.0, f32::INFINITY),
        ParamSpec::new("gain", Unit::Db, 0.0, f32::NEG_INFINITY, f32::INFINITY),
        ParamSpec::new("q", Unit::Q, 0.707, 1e-3, 1000.0),
    ];

    /// RBJ high shelf: `gain` dB above `frequency` (bandwidth `q`).
    HighShelf => "highshelf", Filter, [
        ParamSpec::new("frequency", Unit::Hz, 1000.0, 0.0, f32::INFINITY),
        ParamSpec::new("gain", Unit::Db, 0.0, f32::NEG_INFINITY, f32::INFINITY),
        ParamSpec::new("q", Unit::Q, 0.707, 1e-3, 1000.0),
    ];

    /// RBJ notch at `frequency` with bandwidth `q`.
    Notch => "notch", Filter, [
        ParamSpec::new("frequency", Unit::Hz, 1000.0, 0.0, f32::INFINITY),
        ParamSpec::new("q", Unit::Q, 0.707, 1e-3, 1000.0),
    ];

    /// RBJ band-pass (0 dB peak) at `frequency` with bandwidth `q`.
    Bandpass => "bandpass", Filter, [
        ParamSpec::new("frequency", Unit::Hz, 1000.0, 0.0, f32::INFINITY),
        ParamSpec::new("q", Unit::Q, 0.707, 1e-3, 1000.0),
    ];

    /// RBJ all-pass at `frequency` with bandwidth `q`.
    Allpass => "allpass", Filter, [
        ParamSpec::new("frequency", Unit::Hz, 1000.0, 0.0, f32::INFINITY),
        ParamSpec::new("q", Unit::Q, 0.707, 1e-3, 1000.0),
    ];

    /// Single delayed tap crossfaded with the dry signal (`time` s, `mix`).
    Delay => "delay", Effect, [
        ParamSpec::new("time", Unit::Seconds, 0.25, 0.0, 60.0),
        ParamSpec::new("mix", Unit::Linear, 0.5, 0.0, 1.0),
    ];

    /// Feedback echo: `wet` repeating echoes spaced `time` s apart with `feedback`.
    Echo => "echo", Effect, [
        ParamSpec::new("time", Unit::Seconds, 0.25, 0.0, 60.0),
        ParamSpec::new("feedback", Unit::Linear, 0.3, 0.0, 0.99),
        ParamSpec::new("wet", Unit::Linear, 0.5, 0.0, 1.0),
    ];

    /// Chebyshev Type I low-pass (`cutoff` Hz, `order`, passband `ripple` dB).
    Cheby1Lowpass => "cheby1_lowpass", Filter, [
        ParamSpec::new("cutoff", Unit::Hz, 1000.0, 0.0, f32::INFINITY),
        ParamSpec::new("order", Unit::Linear, 4.0, 1.0, 16.0),
        ParamSpec::new("ripple", Unit::Db, 1.0, 1e-2, 12.0),
    ];

    /// Chebyshev Type I high-pass (`cutoff` Hz, `order`, passband `ripple` dB).
    Cheby1Highpass => "cheby1_highpass", Filter, [
        ParamSpec::new("cutoff", Unit::Hz, 1000.0, 0.0, f32::INFINITY),
        ParamSpec::new("order", Unit::Linear, 4.0, 1.0, 16.0),
        ParamSpec::new("ripple", Unit::Db, 1.0, 1e-2, 12.0),
    ];

    /// Chebyshev Type II low-pass (`cutoff` = stopband edge Hz, `order`, stopband `atten` dB).
    Cheby2Lowpass => "cheby2_lowpass", Filter, [
        ParamSpec::new("cutoff", Unit::Hz, 1000.0, 0.0, f32::INFINITY),
        ParamSpec::new("order", Unit::Linear, 4.0, 1.0, 16.0),
        ParamSpec::new("atten", Unit::Db, 40.0, 10.0, 120.0),
    ];

    /// Chebyshev Type II high-pass (`cutoff` = stopband edge Hz, `order`, stopband `atten` dB).
    Cheby2Highpass => "cheby2_highpass", Filter, [
        ParamSpec::new("cutoff", Unit::Hz, 1000.0, 0.0, f32::INFINITY),
        ParamSpec::new("order", Unit::Linear, 4.0, 1.0, 16.0),
        ParamSpec::new("atten", Unit::Db, 40.0, 10.0, 120.0),
    ];

    /// Schroeder–Moorer reverb (`room` size, `damping`, wet/dry `mix`).
    Reverb => "reverb", Effect, [
        ParamSpec::new("room", Unit::Linear, 0.5, 0.0, 1.0),
        ParamSpec::new("damping", Unit::Linear, 0.3, 0.0, 1.0),
        ParamSpec::new("mix", Unit::Linear, 0.3, 0.0, 1.0),
    ];

    /// Direct-form FIR filter: `y[n] = Σ_k taps[k]·x[n-k]`. **Variadic** — its parameters are the
    /// tap vector itself (one or more), the realtime/graph form of a trained or frozen FIR. The one
    /// parameter listed is the prototype for a single tap.
    Fir => "fir", Filter, [
        ParamSpec::new("tap", Unit::Linear, 1.0, f32::NEG_INFINITY, f32::INFINITY),
    ];

    /// Amplitude fade: `fadein` seconds ramping in, `fadeout` seconds ramping out, with a `shape`
    /// curve (0 = linear, 1 = quarter-sine [the SoX default], 2 = half-sine). Length-preserving.
    Fade => "fade", Effect, [
        ParamSpec::new("fadein", Unit::Seconds, 0.0, 0.0, 3600.0),
        ParamSpec::new("fadeout", Unit::Seconds, 0.0, 0.0, 3600.0),
        ParamSpec::new("shape", Unit::Linear, 1.0, 0.0, 2.0),
    ];

    /// Tremolo: amplitude LFO at `rate` Hz dipping by `depth` (0..1). Length-preserving.
    Tremolo => "tremolo", Effect, [
        ParamSpec::new("rate", Unit::Hz, 5.0, 0.0, 20_000.0),
        ParamSpec::new("depth", Unit::Linear, 0.5, 0.0, 1.0),
    ];

    /// Overdrive: `gain` dB of drive through a `tanh` soft-clipper with a `colour` asymmetry bias.
    /// Nonlinear (not differentiable here).
    Overdrive => "overdrive", Effect, [
        ParamSpec::new("gain", Unit::Db, 20.0, 0.0, 100.0),
        ParamSpec::new("colour", Unit::Linear, 0.2, 0.0, 1.0),
    ];

    /// Feed-forward compressor / expander (compand): one-pole peak-envelope follower (`attack`,
    /// `release` s) driving a soft-knee gain computer (`threshold` dBFS, `ratio`, `knee` dB,
    /// `makeup` dB). Stateful per-channel — realtime-playable.
    Compand => "compand", Effect, [
        ParamSpec::new("attack", Unit::Seconds, 0.01, 0.0, 10.0),
        ParamSpec::new("release", Unit::Seconds, 0.1, 0.0, 10.0),
        ParamSpec::new("threshold", Unit::Db, -20.0, -120.0, 0.0),
        ParamSpec::new("ratio", Unit::Linear, 4.0, 1.0, 100.0),
        ParamSpec::new("knee", Unit::Db, 6.0, 0.0, 48.0),
        ParamSpec::new("makeup", Unit::Db, 0.0, -48.0, 48.0),
    ];

    /// Per-channel time reversal (no parameters). Length-preserving, but **not** realtime (it needs
    /// the whole signal).
    Reverse => "reverse", Effect, [];

    /// A raw second-order section from explicit coefficients `b0 b1 b2 a1 a2` (`a0` normalized to
    /// 1). Reuses the biquad/SOS machinery, so it is differentiable / freezable / realtime like the
    /// designed filters.
    Biquad => "biquad", Filter, [
        ParamSpec::new("b0", Unit::Linear, 1.0, f32::NEG_INFINITY, f32::INFINITY),
        ParamSpec::new("b1", Unit::Linear, 0.0, f32::NEG_INFINITY, f32::INFINITY),
        ParamSpec::new("b2", Unit::Linear, 0.0, f32::NEG_INFINITY, f32::INFINITY),
        ParamSpec::new("a1", Unit::Linear, 0.0, f32::NEG_INFINITY, f32::INFINITY),
        ParamSpec::new("a2", Unit::Linear, 0.0, f32::NEG_INFINITY, f32::INFINITY),
    ];

    /// Chorus: an LFO-modulated fractional-delay voice (`rate` Hz, `depth` s, `delay` s) blended by
    /// `mix`. Feed-forward (no feedback). Length-preserving.
    Chorus => "chorus", Effect, [
        ParamSpec::new("rate", Unit::Hz, 1.5, 0.0, 100.0),
        ParamSpec::new("depth", Unit::Seconds, 0.002, 0.0, 1.0),
        ParamSpec::new("delay", Unit::Seconds, 0.025, 0.0, 1.0),
        ParamSpec::new("mix", Unit::Linear, 0.5, 0.0, 1.0),
    ];

    /// Flanger: a short LFO-modulated delay (`rate` Hz, `depth` s, `delay` s) with `feedback`,
    /// blended by `mix`. Length-preserving.
    Flanger => "flanger", Effect, [
        ParamSpec::new("rate", Unit::Hz, 0.5, 0.0, 100.0),
        ParamSpec::new("depth", Unit::Seconds, 0.002, 0.0, 1.0),
        ParamSpec::new("delay", Unit::Seconds, 0.001, 0.0, 1.0),
        ParamSpec::new("feedback", Unit::Linear, 0.5, -0.95, 0.95),
        ParamSpec::new("mix", Unit::Linear, 0.5, 0.0, 1.0),
    ];

    /// Phaser: an LFO-swept cascade of first-order all-pass stages (`rate` Hz, `depth`) with
    /// `feedback`, blended by `mix`. Length-preserving.
    Phaser => "phaser", Effect, [
        ParamSpec::new("rate", Unit::Hz, 0.5, 0.0, 100.0),
        ParamSpec::new("depth", Unit::Linear, 0.5, 0.0, 1.0),
        ParamSpec::new("feedback", Unit::Linear, 0.5, -0.95, 0.95),
        ParamSpec::new("mix", Unit::Linear, 0.5, 0.0, 1.0),
    ];
}

impl OpKind {
    /// Whether this op is **variadic** — its parameters are a variable-length list of one repeated
    /// [`params`](OpKind::params) spec (`≥ 1` entries), rather than a fixed positional tuple. Only
    /// [`OpKind::Fir`] (the tap vector) is variadic today; [`Op::new`] validates it as "at least
    /// one, each within the single spec's bounds".
    pub fn is_variadic(self) -> bool {
        matches!(self, OpKind::Fir)
    }

    /// Look up a kind by its text name (inverse of [`OpKind::name`]).
    pub fn from_name(name: &str) -> Option<OpKind> {
        OpKind::all().iter().copied().find(|k| k.name() == name)
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
    use super::{Group, Op, OpKind};

    #[test]
    fn name_roundtrips() {
        for &k in OpKind::all() {
            assert_eq!(OpKind::from_name(k.name()), Some(k));
        }
        assert_eq!(OpKind::from_name("nope"), None);
    }

    /// The registry is only a single source of truth if the names in it are unique.
    #[test]
    fn names_are_unique_and_well_formed() {
        let mut seen = std::collections::BTreeSet::new();
        for &k in OpKind::all() {
            let name = k.name();
            assert!(seen.insert(name), "duplicate op name '{name}'");
            assert!(!name.is_empty(), "empty op name for {k:?}");
            assert!(
                name.chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_'),
                "op name '{name}' is not lowercase ascii/digits/underscore"
            );
            assert!(
                name.starts_with(|c: char| c.is_ascii_lowercase()),
                "op name '{name}' must start with a letter (it is an identifier on every interface)"
            );
        }
    }

    /// Every op lands in exactly one catalogue section, and both sections are non-empty — the
    /// `fluxion.filter` / `fluxion.effect` split depends on it.
    #[test]
    fn every_op_has_a_group() {
        let filters = OpKind::all()
            .iter()
            .filter(|k| k.group() == Group::Filter)
            .count();
        let effects = OpKind::all()
            .iter()
            .filter(|k| k.group() == Group::Effect)
            .count();
        assert_eq!(filters + effects, OpKind::all().len());
        assert!(filters > 0 && effects > 0);
        assert_eq!(OpKind::Lowpass.group(), Group::Filter);
        assert_eq!(OpKind::Gain.group(), Group::Effect);
        assert_eq!(Group::Filter.as_str(), "filter");
    }

    /// Every parameter name is a usable identifier on every interface (Python keyword argument, C
    /// doc, `name(cutoff=800)` in chain text), and unique within its op.
    #[test]
    fn param_names_are_identifiers_and_unique_per_op() {
        for &k in OpKind::all() {
            let mut seen = std::collections::BTreeSet::new();
            for spec in k.params() {
                assert!(
                    seen.insert(spec.name),
                    "op '{}' has two parameters named '{}'",
                    k.name(),
                    spec.name
                );
                assert!(
                    spec.name
                        .chars()
                        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_'),
                    "op '{}' parameter '{}' is not a lowercase identifier",
                    k.name(),
                    spec.name
                );
                assert!(
                    spec.min <= spec.default && spec.default <= spec.max,
                    "op '{}' parameter '{}' default {} is outside [{}, {}]",
                    k.name(),
                    spec.name,
                    spec.default,
                    spec.min,
                    spec.max
                );
            }
        }
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

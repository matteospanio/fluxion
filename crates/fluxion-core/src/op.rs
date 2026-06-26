//! The typed op catalog ([`OpKind`]) and concrete op instances ([`Op`]).

use std::fmt;

use crate::param::{ParamSpec, Unit};

/// The kind of a DSP leaf op. Each kind has a fixed parameter schema ([`OpKind::params`]).
///
/// This is the typed replacement for the earlier string-keyed placeholder: it makes the IR
/// self-describing (names, units, defaults, bounds) for validation, the CLI parser, and lowering.
/// New ops are added here as the `fluxion-ops` crate grows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OpKind {
    /// Linear gain, `y = x * gain`.
    Gain,
    /// Low-pass filter (cutoff in Hz). The API layer exposes this as the `Lo`-prefixed variant.
    Lowpass,
    /// High-pass filter (cutoff in Hz). The API layer exposes this as the `Hi`-prefixed variant.
    Highpass,
}

// Static parameter tables — one per kind.
static GAIN_PARAMS: [ParamSpec; 1] = [ParamSpec::new(
    "gain",
    Unit::Linear,
    1.0,
    f32::NEG_INFINITY,
    f32::INFINITY,
)];
static LOWPASS_PARAMS: [ParamSpec; 1] = [ParamSpec::new(
    "cutoff",
    Unit::Hz,
    1000.0,
    0.0,
    f32::INFINITY,
)];
static HIGHPASS_PARAMS: [ParamSpec; 1] = [ParamSpec::new(
    "cutoff",
    Unit::Hz,
    1000.0,
    0.0,
    f32::INFINITY,
)];

impl OpKind {
    /// Stable identifier used in the DSL / CLI / `.fxg`, e.g. `"lowpass"`.
    pub fn name(self) -> &'static str {
        match self {
            OpKind::Gain => "gain",
            OpKind::Lowpass => "lowpass",
            OpKind::Highpass => "highpass",
        }
    }

    /// The parameter schema for this op, in positional order.
    pub fn params(self) -> &'static [ParamSpec] {
        match self {
            OpKind::Gain => &GAIN_PARAMS,
            OpKind::Lowpass => &LOWPASS_PARAMS,
            OpKind::Highpass => &HIGHPASS_PARAMS,
        }
    }

    /// Look up a kind by its DSL name (inverse of [`OpKind::name`]).
    pub fn from_name(name: &str) -> Option<OpKind> {
        match name {
            "gain" => Some(OpKind::Gain),
            "lowpass" => Some(OpKind::Lowpass),
            "highpass" => Some(OpKind::Highpass),
            _ => None,
        }
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
#[derive(Clone, Debug, PartialEq)]
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
    /// assert!(Op::new(OpKind::Lowpass, [800.0]).is_ok());
    /// assert!(Op::new(OpKind::Lowpass, [800.0, 1.0]).is_err()); // wrong arity
    /// assert!(Op::new(OpKind::Lowpass, [-1.0]).is_err());       // out of range
    /// ```
    pub fn new(kind: OpKind, params: impl Into<Vec<f32>>) -> Result<Op, OpError> {
        let params = params.into();
        let specs = kind.params();
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
        for k in [OpKind::Gain, OpKind::Lowpass, OpKind::Highpass] {
            assert_eq!(OpKind::from_name(k.name()), Some(k));
        }
        assert_eq!(OpKind::from_name("nope"), None);
    }

    #[test]
    fn defaults_match_arity() {
        assert_eq!(OpKind::Lowpass.defaults(), vec![1000.0]);
    }

    #[test]
    fn validation_rejects_bad_arity_and_range() {
        assert!(Op::new(OpKind::Gain, [1.0]).is_ok());
        assert!(Op::new(OpKind::Gain, []).is_err());
        assert!(Op::new(OpKind::Lowpass, [-5.0]).is_err());
        assert!(Op::new(OpKind::Lowpass, [f32::NAN]).is_err());
    }
}

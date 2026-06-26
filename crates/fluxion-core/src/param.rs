//! Parameter metadata for DSP ops.

/// Physical unit of a parameter — drives CLI hints, display, and (later) validation/UX.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Unit {
    /// Frequency in hertz.
    Hz,
    /// Gain in decibels.
    Db,
    /// Dimensionless linear factor.
    Linear,
    /// Time in seconds.
    Seconds,
    /// A quality factor (Q).
    Q,
}

impl Unit {
    /// Short display suffix, e.g. `"Hz"` (empty for [`Unit::Linear`]).
    pub fn suffix(self) -> &'static str {
        match self {
            Unit::Hz => "Hz",
            Unit::Db => "dB",
            Unit::Linear => "",
            Unit::Seconds => "s",
            Unit::Q => "Q",
        }
    }
}

/// Static description of one op parameter: name, unit, default, and inclusive bounds.
///
/// `min`/`max` are *design-time* bounds; sample-rate-relative limits (e.g. cutoff below Nyquist)
/// are checked later, once `fs` is known.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParamSpec {
    /// Parameter name, e.g. `"cutoff"`.
    pub name: &'static str,
    /// Physical unit.
    pub unit: Unit,
    /// Default value.
    pub default: f32,
    /// Inclusive lower bound.
    pub min: f32,
    /// Inclusive upper bound.
    pub max: f32,
}

impl ParamSpec {
    /// `const` constructor so op catalogs can live in `static` tables.
    pub const fn new(name: &'static str, unit: Unit, default: f32, min: f32, max: f32) -> Self {
        Self {
            name,
            unit,
            default,
            min,
            max,
        }
    }
}

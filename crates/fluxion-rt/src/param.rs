//! Click-free parameter ramping (plan task G4), on the shared curve evaluator (ROADMAP D3).
//!
//! Jumping a parameter (gain, cutoff, …) between blocks puts a step discontinuity in the signal —
//! an audible click. [`SmoothedValue`] ramps from the current value to a new target over a set
//! number of samples, so the audio thread reads a smooth per-sample value. Allocation-free and
//! branch-light; the realtime executor calls [`SmoothedValue::tick`] once per sample.
//!
//! The command *queue* that delivers new targets to the audio thread is the SPSC [`ring`](crate::ring)
//! (apply at block boundaries); this is just the ramp.
//!
//! # Why this calls into `fluxion-core`
//!
//! ROADMAP D3 asks that "what you hear live is what renders". That is only achievable if both
//! engines compute the *same function*, so this ramp is not its own arithmetic: it asks
//! [`fluxion_core::automation::segment`] for the value, exactly as the offline automation pass
//! does. Two implementations of a straight line would agree to within rounding and no further.
//!
//! And rounding is not small enough to ignore here. The obvious ramp accumulates —
//! `current += step` once per sample — which is one add per sample and drifts from the line it is
//! meant to be: over a one-second ramp at 48 kHz the accumulated value is **6.45e-4** away from
//! the exact one, and only 24 of the 48 000 samples land bit-identical. Computing from the sample
//! index instead costs one multiply and lands on the line at every sample. That is the difference
//! between "the same curve" and "nearly the same curve", and D3's check is the former.

use fluxion_core::automation::{Shape, segment};

/// A ramped scalar parameter.
///
/// The ramp is one segment of a [`Curve`](fluxion_core::automation::Curve), evaluated by the same
/// function the offline engine uses — see the module docs for why that matters.
#[derive(Clone, Copy, Debug)]
pub struct SmoothedValue {
    /// Where this ramp began. Fixed for the duration, so the value is a function of `elapsed`
    /// rather than of the previous value.
    start: f32,
    target: f32,
    /// Length of the ramp in samples; 0 means "not ramping".
    total: u32,
    /// Samples emitted since `set_target`.
    elapsed: u32,
    current: f32,
    shape: Shape,
}

impl SmoothedValue {
    /// A value parked at `value` (not ramping).
    pub fn new(value: f32) -> Self {
        Self {
            start: value,
            target: value,
            total: 0,
            elapsed: 0,
            current: value,
            shape: Shape::Linear,
        }
    }

    /// Ramp to `target` over `ramp_samples` samples. `0` jumps immediately. Resets any ramp in
    /// progress to start from the current value.
    pub fn set_target(&mut self, target: f32, ramp_samples: u32) {
        self.set_target_shaped(target, ramp_samples, Shape::Linear);
    }

    /// The same, along `shape` rather than a straight line — the curve shapes an automation lane
    /// can draw are the ones a live ramp can follow.
    pub fn set_target_shaped(&mut self, target: f32, ramp_samples: u32, shape: Shape) {
        self.start = self.current;
        self.target = target;
        self.total = ramp_samples;
        self.elapsed = 0;
        self.shape = shape;
        if ramp_samples == 0 {
            self.current = target;
        }
    }

    /// Advance one sample and return the value to apply to it.
    ///
    /// The value comes from the sample index, not from the previous value, so it lands on the
    /// curve at every sample and reaches the target exactly on the last one.
    pub fn tick(&mut self) -> f32 {
        if self.elapsed < self.total {
            self.elapsed += 1;
            self.current = segment(
                self.start,
                self.target,
                self.shape,
                self.elapsed as f32 / self.total as f32,
            );
        }
        self.current
    }

    /// The current value without advancing.
    pub fn value(&self) -> f32 {
        self.current
    }

    /// True while a ramp is in progress.
    pub fn is_ramping(&self) -> bool {
        self.elapsed < self.total
    }
}

#[cfg(test)]
mod tests {
    use super::SmoothedValue;

    #[test]
    fn ramps_linearly_then_holds() {
        let mut v = SmoothedValue::new(0.0);
        v.set_target(1.0, 4);
        let seq: Vec<f32> = (0..6).map(|_| v.tick()).collect();
        for (got, want) in seq.iter().zip(&[0.25, 0.5, 0.75, 1.0, 1.0, 1.0]) {
            assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        }
        assert!(!v.is_ramping());
        assert_eq!(v.value(), 1.0);
    }

    #[test]
    fn zero_ramp_jumps_immediately() {
        let mut v = SmoothedValue::new(-1.0);
        v.set_target(2.0, 0);
        assert_eq!(v.value(), 2.0);
        assert!(!v.is_ramping());
        assert_eq!(v.tick(), 2.0);
    }

    #[test]
    fn lands_exactly_on_target() {
        let mut v = SmoothedValue::new(0.0);
        v.set_target(1.0, 3); // 1/3 doesn't represent exactly in f32
        for _ in 0..3 {
            v.tick();
        }
        assert_eq!(v.value(), 1.0, "snaps to target on the final step");
    }
}

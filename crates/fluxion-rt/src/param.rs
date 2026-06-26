//! Click-free parameter ramping (plan task G4).
//!
//! Jumping a parameter (gain, cutoff, …) between blocks puts a step discontinuity in the signal —
//! an audible click. [`SmoothedValue`] ramps linearly from the current value to a new target over a
//! set number of samples, so the audio thread reads a smooth per-sample value. Allocation-free and
//! branch-light; the realtime executor calls [`SmoothedValue::tick`] once per sample.
//!
//! The command *queue* that delivers new targets to the audio thread is the SPSC [`ring`](crate::ring)
//! (apply at block boundaries); this is just the ramp.

/// A linearly-ramped scalar parameter.
#[derive(Clone, Copy, Debug)]
pub struct SmoothedValue {
    current: f32,
    target: f32,
    step: f32,
    remaining: u32,
}

impl SmoothedValue {
    /// A value parked at `value` (not ramping).
    pub fn new(value: f32) -> Self {
        Self {
            current: value,
            target: value,
            step: 0.0,
            remaining: 0,
        }
    }

    /// Ramp to `target` over `ramp_samples` samples. `0` jumps immediately. Resets any ramp in
    /// progress to start from the current value.
    pub fn set_target(&mut self, target: f32, ramp_samples: u32) {
        self.target = target;
        if ramp_samples == 0 {
            self.current = target;
            self.step = 0.0;
            self.remaining = 0;
        } else {
            self.step = (target - self.current) / ramp_samples as f32;
            self.remaining = ramp_samples;
        }
    }

    /// Advance one sample and return the value to apply to it. Snaps exactly to the target on the
    /// final step (no float drift).
    pub fn tick(&mut self) -> f32 {
        if self.remaining > 0 {
            self.remaining -= 1;
            if self.remaining == 0 {
                self.current = self.target;
            } else {
                self.current += self.step;
            }
        }
        self.current
    }

    /// The current value without advancing.
    pub fn value(&self) -> f32 {
        self.current
    }

    /// True while a ramp is in progress.
    pub fn is_ramping(&self) -> bool {
        self.remaining > 0
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

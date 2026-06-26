//! Delay-based effects: a single delayed tap ([`delay`]) and a feedback [`echo`].
//!
//! Both are length-preserving (the feedback tail beyond the input is truncated), which matches the
//! realtime engine's chunk-length-preserving contract.
//!
//! ponytail: integer-sample delay for now; fractional (linear-interpolated) delay is a follow-up
//! when pitch/chorus-style modulation needs it.

/// The wet signal of a feedback delay line: `w[n] = x[n-d] + feedback·w[n-d]`.
fn feedback_delay(input: &[f32], d: usize, feedback: f32) -> Vec<f32> {
    let d = d.max(1);
    let mut w = vec![0.0f32; input.len()];
    for n in d..input.len() {
        w[n] = input[n - d] + feedback * w[n - d];
    }
    w
}

/// Single delayed tap crossfaded with the dry signal: `(1-mix)·x[n] + mix·x[n-d]`.
pub fn delay(input: &[f32], delay_samples: usize, mix: f32) -> Vec<f32> {
    let w = feedback_delay(input, delay_samples, 0.0); // w[n] = x[n-d]
    input
        .iter()
        .zip(&w)
        .map(|(&x, &xd)| (1.0 - mix) * x + mix * xd)
        .collect()
}

/// Feedback echo: the dry signal plus `wet`-scaled repeating echoes spaced `delay_samples` apart.
pub fn echo(input: &[f32], delay_samples: usize, feedback: f32, wet: f32) -> Vec<f32> {
    let w = feedback_delay(input, delay_samples, feedback);
    input.iter().zip(&w).map(|(&x, &e)| x + wet * e).collect()
}

#[cfg(test)]
mod tests {
    use super::{delay, echo};

    fn impulse(n: usize) -> Vec<f32> {
        let mut x = vec![0.0f32; n];
        x[0] = 1.0;
        x
    }

    #[test]
    fn delay_taps_dry_and_delayed() {
        let y = delay(&impulse(16), 4, 0.5);
        assert!((y[0] - 0.5).abs() < 1e-6); // (1-mix)·dry
        assert!((y[4] - 0.5).abs() < 1e-6); // mix·delayed
        assert!(y[1..4].iter().all(|&v| v == 0.0));
        assert_eq!(y.len(), 16); // length-preserving
    }

    #[test]
    fn echo_repeats_with_feedback() {
        let y = echo(&impulse(16), 3, 0.5, 1.0);
        assert!((y[0] - 1.0).abs() < 1e-6); // dry
        assert!((y[3] - 1.0).abs() < 1e-6); // 1st echo
        assert!((y[6] - 0.5).abs() < 1e-6); // 2nd (×feedback)
        assert!((y[9] - 0.25).abs() < 1e-6); // 3rd (×feedback²)
    }
}

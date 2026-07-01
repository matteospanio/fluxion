//! Schroeder–Moorer reverberation (plan task D8).
//!
//! Four parallel feedback comb filters (with a one-pole low-pass in each feedback loop for
//! frequency-dependent decay — the "damping") are summed and run through two series all-pass
//! filters, then crossfaded with the dry signal. Length-preserving (the tail beyond the input is
//! truncated), matching the engine's contract. Coefficients are the classic Freeverb delays.
//!
//! ponytail: fixed delays (tuned at 44.1 kHz) give a consistent character; scaling them by `fs` and
//! a differentiable VJP (E5) are refinements for later.

/// One-pole-damped feedback comb: `y[n] = x[n] + g·lp(y[n-d])`, the low-pass state taming the tail.
fn comb(input: &[f32], d: usize, g: f32, damp: f32) -> Vec<f32> {
    let mut y = vec![0.0f32; input.len()];
    let mut lp = 0.0f32;
    for n in 0..input.len() {
        let yd = if n >= d { y[n - d] } else { 0.0 };
        lp = yd * (1.0 - damp) + lp * damp; // low-pass the feedback
        y[n] = input[n] + g * lp;
    }
    y
}

/// Schroeder all-pass: `y[n] = -g·x[n] + x[n-d] + g·y[n-d]` (flat magnitude, diffuses phase).
fn allpass(input: &[f32], d: usize, g: f32) -> Vec<f32> {
    let mut y = vec![0.0f32; input.len()];
    for n in 0..input.len() {
        let xd = if n >= d { input[n - d] } else { 0.0 };
        let yd = if n >= d { y[n - d] } else { 0.0 };
        y[n] = -g * input[n] + xd + g * yd;
    }
    y
}

/// Reverberate `input`: `room_size` in `[0,1]` sets the comb feedback (tail length), `damping` in
/// `[0,1]` rolls off the high end of the tail, `mix` in `[0,1]` is the wet/dry blend.
pub fn reverb(input: &[f32], room_size: f32, damping: f32, mix: f32) -> Vec<f32> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }
    const COMB_DELAYS: [usize; 4] = [1557, 1617, 1491, 1422];
    const ALLPASS_DELAYS: [usize; 2] = [225, 556];
    let g = room_size.clamp(0.0, 0.98); // < 1 keeps the comb feedback BIBO-stable
    let damp = damping.clamp(0.0, 1.0);

    // Parallel combs, averaged.
    let mut wet = vec![0.0f32; n];
    for &d in &COMB_DELAYS {
        for (w, c) in wet.iter_mut().zip(comb(input, d, g, damp)) {
            *w += c;
        }
    }
    wet.iter_mut().for_each(|w| *w *= 0.25);

    // Series all-pass diffusion.
    for &d in &ALLPASS_DELAYS {
        wet = allpass(&wet, d, 0.5);
    }

    input
        .iter()
        .zip(&wet)
        .map(|(&x, &w)| (1.0 - mix) * x + mix * w)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::reverb;

    #[test]
    fn length_preserving_and_finite() {
        let x: Vec<f32> = (0..4000).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
        let y = reverb(&x, 0.8, 0.3, 0.5);
        assert_eq!(y.len(), x.len());
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn dry_when_mix_zero_and_decays() {
        let x: Vec<f32> = (0..4000).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
        // mix=0 → dry passthrough.
        let dry = reverb(&x, 0.8, 0.2, 0.0);
        assert!((dry[0] - 1.0).abs() < 1e-6);
        assert!(dry[1..].iter().all(|&v| v.abs() < 1e-6));
        // wet impulse response has energy spread into the tail (after the first comb delay).
        let wet = reverb(&x, 0.85, 0.2, 1.0);
        let tail: f32 = wet[1557..2200].iter().map(|v| v.abs()).sum();
        assert!(tail > 1e-3, "reverb tail is silent");
    }

    #[test]
    fn stable_at_max_room_size() {
        // room_size is clamped < 1, so even a long input stays bounded.
        let x: Vec<f32> = (0..20_000).map(|i| (0.05 * i as f32).sin()).collect();
        let y = reverb(&x, 1.0, 0.0, 1.0);
        assert!(y.iter().all(|v| v.abs() < 50.0), "reverb blew up");
    }
}

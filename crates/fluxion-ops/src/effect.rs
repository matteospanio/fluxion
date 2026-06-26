//! Simple amplitude effects.

/// Multiply a channel in place by a linear gain factor.
pub fn gain(channel: &mut [f32], factor: f32) {
    for x in channel {
        *x *= factor;
    }
}

/// Peak-normalize a multichannel signal in place so its largest absolute sample equals `target`.
///
/// The gain is computed from the global peak across all channels, preserving inter-channel balance.
/// A silent signal (peak 0) is left unchanged.
pub fn normalize_peak(channels: &mut [Vec<f32>], target: f32) {
    let peak = channels
        .iter()
        .flat_map(|c| c.iter())
        .fold(0.0f32, |m, &x| m.max(x.abs()));
    if peak > 0.0 {
        let g = target / peak;
        for c in channels.iter_mut() {
            gain(c, g);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{gain, normalize_peak};

    #[test]
    fn gain_scales() {
        let mut c = vec![1.0, -2.0, 0.5];
        gain(&mut c, 2.0);
        assert_eq!(c, vec![2.0, -4.0, 1.0]);
    }

    #[test]
    fn normalize_hits_target_peak() {
        let mut chans = vec![vec![0.0, 0.25, -0.5], vec![0.1, 0.2, 0.0]];
        normalize_peak(&mut chans, 1.0);
        let peak = chans.iter().flatten().fold(0.0f32, |m, &x| m.max(x.abs()));
        assert!((peak - 1.0).abs() < 1e-6);
    }
}

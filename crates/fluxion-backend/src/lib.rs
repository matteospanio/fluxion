//! `fluxion-backend` — graph execution.
//!
//! Today: a concrete scalar CPU executor ([`process`]) that walks a [`Graph`] and applies the
//! `fluxion-ops` kernels across a multichannel [`Signal`].
//!
//! ponytail: the generic `Backend` trait (one kernel set, many devices — plan task C1) is
//! deliberately *not* introduced yet. With only a CPU implementation it would be an abstraction
//! with one impl; it gets extracted at M2 when the Burn backend is the real second implementation
//! and the right primitive set is known. Until then, ops are called directly here.

use fluxion_core::{Graph, Op, OpKind, Signal};
use fluxion_ops::{
    Biquad, allpass, bandpass, biquad_forward, butterworth_highpass, butterworth_lowpass,
    chebyshev1_highpass, chebyshev1_lowpass, delay, echo, gain, high_shelf, low_shelf,
    normalize_peak, notch, peaking, sos_filter,
};

/// Run a graph over an input signal on the CPU, returning a new signal.
///
/// `|` chains in order; `+` runs both branches on the same input and sums their outputs.
pub fn process(graph: &Graph, input: &Signal) -> Signal {
    match graph {
        Graph::Id => input.clone(),
        Graph::Op(op) => apply_op(op, input),
        Graph::Series(a, b) => process(b, &process(a, input)),
        Graph::Parallel(a, b) => add_signals(&process(a, input), &process(b, input)),
    }
}

fn apply_op(op: &Op, input: &Signal) -> Signal {
    let fs = input.fs;
    let p = &op.params;
    let mut out = input.clone();
    match op.kind {
        OpKind::Gain => {
            let g = op.params[0];
            for ch in &mut out.channels {
                gain(ch, g);
            }
        }
        OpKind::Lowpass => {
            let order = op.params[1].round().max(1.0) as usize;
            let sos = butterworth_lowpass(order, op.params[0], fs);
            for ch in &mut out.channels {
                *ch = sos_filter(ch, &sos);
            }
        }
        OpKind::Highpass => {
            let order = op.params[1].round().max(1.0) as usize;
            let sos = butterworth_highpass(order, op.params[0], fs);
            for ch in &mut out.channels {
                *ch = sos_filter(ch, &sos);
            }
        }
        OpKind::Normalize => {
            normalize_peak(&mut out.channels, op.params[0]);
        }
        OpKind::Peaking => apply_biquad(&mut out, peaking(p[0], p[1], p[2], fs)),
        OpKind::LowShelf => apply_biquad(&mut out, low_shelf(p[0], p[1], p[2], fs)),
        OpKind::HighShelf => apply_biquad(&mut out, high_shelf(p[0], p[1], p[2], fs)),
        OpKind::Notch => apply_biquad(&mut out, notch(p[0], p[1], fs)),
        OpKind::Bandpass => apply_biquad(&mut out, bandpass(p[0], p[1], fs)),
        OpKind::Allpass => apply_biquad(&mut out, allpass(p[0], p[1], fs)),
        OpKind::Delay => {
            let d = (p[0] * fs as f32).round() as usize;
            for ch in &mut out.channels {
                *ch = delay(ch, d, p[1]);
            }
        }
        OpKind::Echo => {
            let d = (p[0] * fs as f32).round() as usize;
            for ch in &mut out.channels {
                *ch = echo(ch, d, p[1], p[2]);
            }
        }
        OpKind::Cheby1Lowpass => {
            let sos = chebyshev1_lowpass(p[1].round().max(1.0) as usize, p[0], p[2], fs);
            for ch in &mut out.channels {
                *ch = sos_filter(ch, &sos);
            }
        }
        OpKind::Cheby1Highpass => {
            let sos = chebyshev1_highpass(p[1].round().max(1.0) as usize, p[0], p[2], fs);
            for ch in &mut out.channels {
                *ch = sos_filter(ch, &sos);
            }
        }
        // `OpKind` is `#[non_exhaustive]`; future ops must be added above before use.
        kind => panic!("fluxion-backend: op '{}' is not implemented", kind.name()),
    }
    out
}

/// Apply a single biquad to every channel in place.
fn apply_biquad(sig: &mut Signal, bq: Biquad) {
    for ch in &mut sig.channels {
        *ch = biquad_forward(ch, &bq);
    }
}

/// Sum two signals channel-by-channel, zero-padding to the longer of each pair.
fn add_signals(a: &Signal, b: &Signal) -> Signal {
    let n = a.channels.len().max(b.channels.len());
    let mut channels = Vec::with_capacity(n);
    for c in 0..n {
        match (a.channels.get(c), b.channels.get(c)) {
            (Some(x), Some(y)) => {
                let len = x.len().max(y.len());
                let mut sum = vec![0.0f32; len];
                for (i, s) in sum.iter_mut().enumerate() {
                    *s = x.get(i).copied().unwrap_or(0.0) + y.get(i).copied().unwrap_or(0.0);
                }
                channels.push(sum);
            }
            (Some(x), None) => channels.push(x.clone()),
            (None, Some(y)) => channels.push(y.clone()),
            (None, None) => channels.push(Vec::new()),
        }
    }
    Signal::new(a.fs, channels)
}

#[cfg(test)]
mod tests {
    use super::process;
    use fluxion_core::{Graph, OpKind, Signal};

    fn sig(samples: Vec<f32>) -> Signal {
        Signal::new(48_000, vec![samples])
    }

    #[test]
    fn gain_op_scales() {
        let out = process(&Graph::op(OpKind::Gain, [0.5]), &sig(vec![2.0, -4.0]));
        assert_eq!(out.channels[0], vec![1.0, -2.0]);
    }

    #[test]
    fn series_applies_in_order() {
        let g = Graph::op(OpKind::Gain, [2.0]) | Graph::op(OpKind::Gain, [3.0]);
        let out = process(&g, &sig(vec![1.0]));
        assert_eq!(out.channels[0], vec![6.0]);
    }

    #[test]
    fn parallel_sums_branches() {
        let g = Graph::op(OpKind::Gain, [2.0]) + Graph::op(OpKind::Gain, [3.0]);
        let out = process(&g, &sig(vec![1.0, 10.0]));
        assert_eq!(out.channels[0], vec![5.0, 50.0]);
    }

    #[test]
    fn lowpass_preserves_dc() {
        let g = Graph::op(OpKind::Lowpass, [1_000.0, 4.0]);
        let out = process(&g, &sig(vec![1.0; 2_000]));
        assert!((out.channels[0][1_999] - 1.0).abs() < 1e-3);
    }
}

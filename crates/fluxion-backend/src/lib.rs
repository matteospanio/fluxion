//! `fluxion-backend` — graph execution and stability certification.
//!
//! Today: a concrete scalar CPU executor ([`process`]) that walks a [`Graph`] and applies the
//! `fluxion-ops` kernels across a multichannel [`Signal`], plus [`certify_graph`], which checks the
//! stability of a graph's frozen coefficients before they are shipped (`.fxg` export / realtime).
//!
//! ponytail: the generic `Backend` trait (one kernel set, many devices — plan task C1) is
//! deliberately *not* introduced yet. With only a CPU implementation it would be an abstraction
//! with one impl; it gets extracted at M2/M3 when the Burn backend is the real second
//! implementation. Until then, ops are called directly here.

#[cfg(feature = "cuda")]
pub mod cuda;

use fluxion_core::{Graph, Op, OpKind, Signal};
use fluxion_ops::{
    Sos, allpass, bandpass, butterworth_highpass, butterworth_lowpass, certify_sos,
    chebyshev1_highpass, chebyshev1_lowpass, delay, echo, gain, high_shelf, low_shelf,
    normalize_peak, notch, peaking, small_gain_certify, sos_filter,
};

pub use fluxion_ops::{Certificate, Verdict};

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

/// The SOS cascade an op lowers to, if it is a (cascade of) biquad(s); `None` for non-filter ops.
fn op_sos(op: &Op, fs: u32) -> Option<Sos> {
    let p = &op.params;
    let order = |i: usize| p[i].round().max(1.0) as usize;
    Some(match op.kind {
        OpKind::Lowpass => butterworth_lowpass(order(1), p[0], fs),
        OpKind::Highpass => butterworth_highpass(order(1), p[0], fs),
        OpKind::Cheby1Lowpass => chebyshev1_lowpass(order(1), p[0], p[2], fs),
        OpKind::Cheby1Highpass => chebyshev1_highpass(order(1), p[0], p[2], fs),
        OpKind::Peaking => vec![peaking(p[0], p[1], p[2], fs)],
        OpKind::LowShelf => vec![low_shelf(p[0], p[1], p[2], fs)],
        OpKind::HighShelf => vec![high_shelf(p[0], p[1], p[2], fs)],
        OpKind::Notch => vec![notch(p[0], p[1], fs)],
        OpKind::Bandpass => vec![bandpass(p[0], p[1], fs)],
        OpKind::Allpass => vec![allpass(p[0], p[1], fs)],
        _ => return None,
    })
}

fn apply_op(op: &Op, input: &Signal) -> Signal {
    let fs = input.fs;
    let p = &op.params;
    let mut out = input.clone();

    if let Some(sos) = op_sos(op, fs) {
        for ch in &mut out.channels {
            *ch = sos_filter(ch, &sos);
        }
        return out;
    }

    match op.kind {
        OpKind::Gain => {
            let g = p[0];
            for ch in &mut out.channels {
                gain(ch, g);
            }
        }
        OpKind::Normalize => normalize_peak(&mut out.channels, p[0]),
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
        // `OpKind` is `#[non_exhaustive]`; future ops must be added (here or in `op_sos`) before use.
        kind => panic!("fluxion-backend: op '{}' is not implemented", kind.name()),
    }
    out
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

/// Certify the stability of a graph's frozen coefficients at sample rate `fs`.
///
/// Returns the worst verdict across the graph: filter ops are certified from their designed SOS
/// poles, the feedback `echo` op via its small-gain loop bound, and feedforward ops (gain /
/// normalize / delay) are unconditionally stable. Series/parallel aggregate to their worst child.
pub fn certify_graph(graph: &Graph, fs: u32) -> Certificate {
    match graph {
        Graph::Id => Certificate::certified(),
        Graph::Op(op) => certify_op(op, fs),
        Graph::Series(a, b) | Graph::Parallel(a, b) => {
            certify_graph(a, fs).worst(certify_graph(b, fs))
        }
    }
}

fn certify_op(op: &Op, fs: u32) -> Certificate {
    if let Some(sos) = op_sos(op, fs) {
        return certify_sos(&sos);
    }
    match op.kind {
        // Echo is the one feedback op: its loop gain is the (frequency-flat) feedback coefficient.
        OpKind::Echo => small_gain_certify(|_| op.params[1].abs(), 1),
        // Gain / Normalize / Delay are feedforward and unconditionally stable.
        _ => Certificate::certified(),
    }
}

#[cfg(test)]
mod tests {
    use super::{certify_graph, process};
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
        assert_eq!(process(&g, &sig(vec![1.0])).channels[0], vec![6.0]);
    }

    #[test]
    fn parallel_sums_branches() {
        let g = Graph::op(OpKind::Gain, [2.0]) + Graph::op(OpKind::Gain, [3.0]);
        assert_eq!(
            process(&g, &sig(vec![1.0, 10.0])).channels[0],
            vec![5.0, 50.0]
        );
    }

    #[test]
    fn lowpass_preserves_dc() {
        let g = Graph::op(OpKind::Lowpass, [1_000.0, 4.0]);
        let out = process(&g, &sig(vec![1.0; 2_000]));
        assert!((out.channels[0][1_999] - 1.0).abs() < 1e-3);
    }

    #[test]
    fn certifies_a_stable_chain() {
        let g = Graph::op(OpKind::Cheby1Lowpass, [2_000.0, 6.0, 1.0])
            | Graph::op(OpKind::Echo, [0.1, 0.5, 0.4])
            | Graph::op(OpKind::Peaking, [3_000.0, 6.0, 1.0]);
        assert!(certify_graph(&g, 48_000).verdict.is_shippable());
    }
}

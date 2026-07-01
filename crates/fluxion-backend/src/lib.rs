//! `fluxion-backend` — graph execution and stability certification.
//!
//! [`eval`] is the one graph-walk / op-dispatch surface (plan task C1): it walks a [`Graph`] and
//! calls a [`Backend`]'s kernels. The scalar [`Cpu`] backend here powers [`process`]; the
//! differentiable Burn backend in `fluxion-autodiff` reuses the *same* `eval` to make a whole graph
//! differentiable (plan task E12). Also [`certify_graph`], which checks the stability of a graph's
//! frozen coefficients before they are shipped (`.fxg` export / realtime).

#[cfg(feature = "cuda")]
pub mod cuda;

use fluxion_core::{FrozenSos, Graph, Op, OpKind, Signal};
use fluxion_ops::{
    Biquad, Sos, allpass, bandpass, butterworth_highpass, butterworth_lowpass, certify_sos,
    chebyshev1_highpass, chebyshev1_lowpass, chebyshev2_highpass, chebyshev2_lowpass, delay, echo,
    gain, high_shelf, low_shelf, normalize_peak, notch, peaking, reverb, small_gain_certify,
    sos_filter,
};
use fluxion_rt::RtGraph;

pub use fluxion_ops::{Certificate, Verdict};

/// Run a graph over an input signal on the CPU, returning a new signal.
///
/// `|` chains in order; `+` runs both branches on the same input and sums their outputs. This is
/// [`eval`] over the [`Cpu`] backend — the same graph walk the differentiable Burn backend uses.
pub fn process(graph: &Graph, input: &Signal) -> Signal {
    Signal::new(
        input.fs,
        eval(&Cpu, graph, input.channels.clone(), input.fs),
    )
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
        OpKind::Cheby2Lowpass => chebyshev2_lowpass(order(1), p[0], p[2], fs),
        OpKind::Cheby2Highpass => chebyshev2_highpass(order(1), p[0], p[2], fs),
        OpKind::Peaking => vec![peaking(p[0], p[1], p[2], fs)],
        OpKind::LowShelf => vec![low_shelf(p[0], p[1], p[2], fs)],
        OpKind::HighShelf => vec![high_shelf(p[0], p[1], p[2], fs)],
        OpKind::Notch => vec![notch(p[0], p[1], fs)],
        OpKind::Bandpass => vec![bandpass(p[0], p[1], fs)],
        OpKind::Allpass => vec![allpass(p[0], p[1], fs)],
        _ => return None,
    })
}

/// The per-channel-set kernels the graph executor dispatches to — **one kernel set, many devices**
/// (plan task C1). [`eval`] walks a [`Graph`] and calls these, so the scalar [`Cpu`] executor here
/// and the differentiable Burn executor in `fluxion-autodiff` share a single dispatch surface.
///
/// `Buf` is a backend's signal representation (the CPU's is all channels at once; Burn's is one
/// differentiable tensor). `normalize`/`reverb` are cross-channel / non-differentiable — a backend
/// that can't express them (e.g. autodiff) may leave them `unimplemented!`, guarded by
/// [`is_differentiable`].
pub trait Backend {
    /// A backend's signal buffer.
    type Buf: Clone;
    /// Apply a designed SOS cascade.
    fn filter(&self, x: Self::Buf, sos: &[Biquad]) -> Self::Buf;
    /// Multiply by a linear gain.
    fn gain(&self, x: Self::Buf, factor: f32) -> Self::Buf;
    /// `mix`-blend a `samples`-delayed copy.
    fn delay(&self, x: Self::Buf, samples: usize, mix: f32) -> Self::Buf;
    /// Feedback echo.
    fn echo(&self, x: Self::Buf, samples: usize, feedback: f32, wet: f32) -> Self::Buf;
    /// Peak-normalize (cross-channel).
    fn normalize(&self, x: Self::Buf, peak: f32) -> Self::Buf;
    /// Schroeder–Moorer reverb.
    fn reverb(&self, x: Self::Buf, room: f32, damping: f32, mix: f32) -> Self::Buf;
    /// Sum two buffers (the `+` parallel combine).
    fn add(&self, a: Self::Buf, b: Self::Buf) -> Self::Buf;
}

/// Walk `graph`, dispatching each op to backend `b`'s kernels; `fs` converts delay/echo seconds to
/// samples. Series composes, parallel runs both branches on the same input and [`Backend::add`]s.
pub fn eval<B: Backend>(b: &B, graph: &Graph, x: B::Buf, fs: u32) -> B::Buf {
    match graph {
        Graph::Id => x,
        Graph::Op(op) => eval_op(b, op, x, fs),
        Graph::Series(a, c) => {
            let y = eval(b, a, x, fs);
            eval(b, c, y, fs)
        }
        Graph::Parallel(a, c) => b.add(eval(b, a, x.clone(), fs), eval(b, c, x, fs)),
    }
}

fn eval_op<B: Backend>(b: &B, op: &Op, x: B::Buf, fs: u32) -> B::Buf {
    if let Some(sos) = op_sos(op, fs) {
        return b.filter(x, &sos);
    }
    let p = &op.params;
    let samples = |secs: f32| (secs * fs as f32).round() as usize;
    match op.kind {
        OpKind::Gain => b.gain(x, p[0]),
        OpKind::Normalize => b.normalize(x, p[0]),
        OpKind::Delay => b.delay(x, samples(p[0]), p[1]),
        OpKind::Echo => b.echo(x, samples(p[0]), p[1], p[2]),
        OpKind::Reverb => b.reverb(x, p[0], p[1], p[2]),
        // `OpKind` is `#[non_exhaustive]`; future ops must be added (here or in `op_sos`) before use.
        kind => panic!("fluxion-backend: op '{}' is not implemented", kind.name()),
    }
}

/// True if every op in `graph` lowers to a differentiable Burn op (filters / gain / delay / echo) —
/// i.e. no cross-channel or non-differentiable op (`Normalize`, `Reverb`, or a future op). Guards
/// the autodiff graph lowering (`fluxion-autodiff`), which can't express those.
pub fn is_differentiable(graph: &Graph, fs: u32) -> bool {
    match graph {
        Graph::Id => true,
        Graph::Op(op) => {
            op_sos(op, fs).is_some()
                || matches!(op.kind, OpKind::Gain | OpKind::Delay | OpKind::Echo)
        }
        Graph::Series(a, b) | Graph::Parallel(a, b) => {
            is_differentiable(a, fs) && is_differentiable(b, fs)
        }
    }
}

/// The scalar CPU backend: a `Buf` is all channels of a signal, each kernel mapping over them (the
/// cross-channel `normalize` needs them together).
pub struct Cpu;

impl Backend for Cpu {
    type Buf = Vec<Vec<f32>>;

    fn filter(&self, mut x: Vec<Vec<f32>>, sos: &[Biquad]) -> Vec<Vec<f32>> {
        for ch in &mut x {
            *ch = sos_filter(ch, sos);
        }
        x
    }
    fn gain(&self, mut x: Vec<Vec<f32>>, factor: f32) -> Vec<Vec<f32>> {
        for ch in &mut x {
            gain(ch, factor);
        }
        x
    }
    fn delay(&self, mut x: Vec<Vec<f32>>, samples: usize, mix: f32) -> Vec<Vec<f32>> {
        for ch in &mut x {
            *ch = delay(ch, samples, mix);
        }
        x
    }
    fn echo(&self, mut x: Vec<Vec<f32>>, samples: usize, feedback: f32, wet: f32) -> Vec<Vec<f32>> {
        for ch in &mut x {
            *ch = echo(ch, samples, feedback, wet);
        }
        x
    }
    fn normalize(&self, mut x: Vec<Vec<f32>>, peak: f32) -> Vec<Vec<f32>> {
        normalize_peak(&mut x, peak);
        x
    }
    fn reverb(&self, mut x: Vec<Vec<f32>>, room: f32, damping: f32, mix: f32) -> Vec<Vec<f32>> {
        for ch in &mut x {
            *ch = reverb(ch, room, damping, mix);
        }
        x
    }
    /// Channel-by-channel sum, zero-padding to the longer of each pair (and the more channels).
    fn add(&self, a: Vec<Vec<f32>>, b: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let n = a.len().max(b.len());
        (0..n)
            .map(|c| match (a.get(c), b.get(c)) {
                (Some(x), Some(y)) => {
                    let mut sum = vec![0.0f32; x.len().max(y.len())];
                    for (i, s) in sum.iter_mut().enumerate() {
                        *s = x.get(i).copied().unwrap_or(0.0) + y.get(i).copied().unwrap_or(0.0);
                    }
                    sum
                }
                (Some(x), None) => x.clone(),
                (None, Some(y)) => y.clone(),
                (None, None) => Vec::new(),
            })
            .collect()
    }
}

/// Certify the stability of a graph's frozen coefficients at sample rate `fs`.
///
/// Returns the worst verdict across the graph: filter ops are certified from their designed SOS
/// poles, the feedback `echo`/`reverb` ops via their small-gain loop bound, and feedforward ops
/// (gain / normalize / delay) are unconditionally stable. Series/parallel aggregate to the worst
/// child.
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
        // Echo is a feedback op: its loop gain is the (frequency-flat) feedback coefficient.
        OpKind::Echo => small_gain_certify(|_| op.params[1].abs(), 1),
        // Reverb is parallel damped-feedback combs; each comb's loop gain is the room-size feedback
        // (the one-pole damping only attenuates), clamped < 1 in the design → BIBO-stable.
        OpKind::Reverb => small_gain_certify(|_| op.params[0].clamp(0.0, 0.98), 1),
        // Gain / Normalize / Delay are feedforward and unconditionally stable.
        _ => Certificate::certified(),
    }
}

/// Flatten a pure-filter **series** graph into a single SOS cascade at sample rate `fs`.
///
/// Returns `None` if the graph contains a non-filter op (gain/normalize/delay/echo) or a parallel
/// branch — i.e. anything that is not one concatenated biquad cascade. (Cascade associativity:
/// applying the concatenated sections equals applying the ops in series.)
pub fn graph_to_sos(graph: &Graph, fs: u32) -> Option<Sos> {
    match graph {
        Graph::Id => Some(Vec::new()),
        Graph::Op(op) => op_sos(op, fs),
        Graph::Series(a, b) => {
            let mut s = graph_to_sos(a, fs)?;
            s.extend(graph_to_sos(b, fs)?);
            Some(s)
        }
        Graph::Parallel(..) => None,
    }
}

/// Freeze a pure-filter series graph to a serializable [`FrozenSos`] plan at sample rate `fs` — the
/// designed coefficients, ready for the realtime executor (`fluxion-rt::SosStream::from_sections`)
/// with no design at load time. Returns `None` for graphs that don't reduce to one cascade (same
/// constraint as [`graph_to_sos`]).
pub fn freeze(graph: &Graph, fs: u32) -> Option<FrozenSos> {
    let sos = graph_to_sos(graph, fs)?;
    let sections = sos.iter().map(|b| [b.b0, b.b1, b.b2, b.a1, b.a2]).collect();
    Some(FrozenSos::new(fs, sections))
}

/// Lower a graph to a realtime [`RtGraph`] executor at sample rate `fs`, designing each op's
/// coefficients and converting delay times to samples. The result mirrors [`process`]'s per-op
/// semantics, so streaming it block-by-block matches the batch output. Call
/// [`RtGraph::prepare`](fluxion_rt::RtGraph::prepare) before going realtime.
///
/// Returns `None` for graphs containing an op that isn't chunk-local — currently `Normalize` (it
/// needs the whole signal's peak) and any not-yet-supported op.
pub fn to_rt_graph(graph: &Graph, fs: u32) -> Option<RtGraph> {
    match graph {
        Graph::Id => Some(RtGraph::gain(1.0)),
        Graph::Op(op) => op_rt(op, fs),
        Graph::Series(a, b) => Some(RtGraph::series(to_rt_graph(a, fs)?, to_rt_graph(b, fs)?)),
        Graph::Parallel(a, b) => Some(RtGraph::parallel(to_rt_graph(a, fs)?, to_rt_graph(b, fs)?)),
    }
}

fn op_rt(op: &Op, fs: u32) -> Option<RtGraph> {
    if let Some(sos) = op_sos(op, fs) {
        return Some(RtGraph::filter(sos));
    }
    let p = &op.params;
    let to_samples = |secs: f32| (secs * fs as f32).round() as usize;
    match op.kind {
        OpKind::Gain => Some(RtGraph::gain(p[0])),
        OpKind::Delay => Some(RtGraph::delay(to_samples(p[0]), p[1])),
        OpKind::Echo => Some(RtGraph::echo(to_samples(p[0]), p[1], p[2])),
        _ => None, // Normalize (whole-signal) and any future non-realtime op
    }
}

/// Filter a flat batch of `rows.len() / frames` equal-length rows through an SOS cascade (CPU).
///
/// The GPU kernel ([`cuda::sos_filter_batch`], `cuda` feature) is far faster in raw **compute**
/// (~59× on an RTX 3070), but a one-shot batch filter is dominated by host↔device transfer (~430 ms
/// vs ~300 ms CPU for 67 Msamples), so the CPU is the right default here. The GPU pays off for
/// **resident / repeated** workloads where the data stays on the device across many operations
/// (e.g. a differentiable training loop) — call `cuda::sos_filter_batch` explicitly for those.
pub fn sos_filter_batch(rows: &[f32], frames: usize, sos: &[Biquad]) -> Vec<f32> {
    let mut out = vec![0.0f32; rows.len()];
    for r in 0..rows.len() / frames {
        let y = sos_filter(&rows[r * frames..(r + 1) * frames], sos);
        out[r * frames..(r + 1) * frames].copy_from_slice(&y);
    }
    out
}

/// Process a batch of signals. A pure-filter chain over equal-length **mono** signals is routed to
/// the batched (GPU-when-`cuda`) path; anything else falls back to processing each signal on its own.
pub fn process_batch(graph: &Graph, batch: &[Signal]) -> Vec<Signal> {
    if let Some(out) = try_batched_filter(graph, batch) {
        return out;
    }
    batch.iter().map(|s| process(graph, s)).collect()
}

fn try_batched_filter(graph: &Graph, batch: &[Signal]) -> Option<Vec<Signal>> {
    let first = batch.first()?;
    let (fs, frames) = (first.fs, first.frames());
    if frames == 0
        || !batch
            .iter()
            .all(|s| s.channel_count() == 1 && s.frames() == frames && s.fs == fs)
    {
        return None;
    }
    let sos = graph_to_sos(graph, fs)?;
    if sos.is_empty() {
        return None; // identity / no filtering — nothing to accelerate
    }

    let mut rows = Vec::with_capacity(batch.len() * frames);
    for s in batch {
        rows.extend_from_slice(&s.channels[0]);
    }
    let out = sos_filter_batch(&rows, frames, &sos);
    Some(
        (0..batch.len())
            .map(|i| Signal::new(fs, vec![out[i * frames..(i + 1) * frames].to_vec()]))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::{
        FrozenSos, certify_graph, freeze, graph_to_sos, process, process_batch, to_rt_graph,
    };
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

    #[test]
    fn reverb_certifies_via_small_gain_not_the_catch_all() {
        // Reverb's feedback combs are now actually checked (room-size loop gain < 1), with a real
        // stability margin — not silently blessed by the `_ => certified()` fall-through.
        let c = certify_graph(&Graph::op(OpKind::Reverb, [0.8, 0.5, 0.3]), 48_000);
        assert!(c.verdict.is_shippable());
        assert!(
            c.margin > 0.0 && c.margin.is_finite(),
            "expected a real margin, got {}",
            c.margin
        );
    }

    #[test]
    fn graph_to_sos_flattens_filters_only() {
        // order-4 Butterworth (2 sections) | order-2 high-pass (1 section) = 3 sections.
        let g =
            Graph::op(OpKind::Lowpass, [2_000.0, 4.0]) | Graph::op(OpKind::Highpass, [100.0, 2.0]);
        assert_eq!(graph_to_sos(&g, 48_000).unwrap().len(), 3);
        // A non-filter op or a parallel branch is not one flat cascade.
        assert!(graph_to_sos(&Graph::op(OpKind::Gain, [0.5]), 48_000).is_none());
        let par = Graph::op(OpKind::Lowpass, [1_000.0, 2.0])
            + Graph::op(OpKind::Highpass, [2_000.0, 2.0]);
        assert!(graph_to_sos(&par, 48_000).is_none());
    }

    #[test]
    fn freeze_matches_graph_to_sos() {
        let g =
            Graph::op(OpKind::Lowpass, [2_000.0, 4.0]) | Graph::op(OpKind::Highpass, [100.0, 2.0]);
        let frozen = freeze(&g, 48_000).unwrap();
        let sos = graph_to_sos(&g, 48_000).unwrap();
        assert_eq!(frozen.fs, 48_000);
        assert_eq!(frozen.sections.len(), sos.len());
        for (sec, bq) in frozen.sections.iter().zip(&sos) {
            assert_eq!(*sec, [bq.b0, bq.b1, bq.b2, bq.a1, bq.a2]);
        }
        // Survives a JSON round-trip, and a non-cascade graph won't freeze.
        assert_eq!(FrozenSos::from_json(&frozen.to_json()).unwrap(), frozen);
        assert!(freeze(&Graph::op(OpKind::Gain, [0.5]), 48_000).is_none());
    }

    #[test]
    fn to_rt_graph_streams_like_process() {
        // lowpass | (echo + gain) — filters, a feedback effect, and a parallel sum together.
        let g = Graph::op(OpKind::Lowpass, [2_000.0, 4.0])
            | (Graph::op(OpKind::Echo, [0.01, 0.4, 0.6]) + Graph::op(OpKind::Gain, [0.5]));
        let fs = 48_000;
        let x: Vec<f32> = (0..2_000).map(|i| (0.05 * i as f32).sin()).collect();

        let batch = process(&g, &Signal::new(fs, vec![x.clone()]))
            .channels
            .remove(0);

        let mut rt = to_rt_graph(&g, fs).unwrap();
        rt.prepare(128);
        let mut out = vec![0.0f32; 128];
        let mut streamed = Vec::with_capacity(x.len());
        for chunk in x.chunks(128) {
            let out = &mut out[..chunk.len()];
            rt.process(chunk, out);
            streamed.extend_from_slice(out);
        }

        assert_eq!(streamed.len(), batch.len());
        for (a, b) in streamed.iter().zip(&batch) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
        // Normalize needs the whole signal, so it can't be lowered to a realtime graph.
        assert!(to_rt_graph(&Graph::op(OpKind::Normalize, [1.0]), fs).is_none());
    }

    #[test]
    fn process_batch_matches_per_signal() {
        let g = Graph::op(OpKind::Lowpass, [2_000.0, 4.0])
            | Graph::op(OpKind::Peaking, [1_000.0, 6.0, 1.0]);
        let batch: Vec<Signal> = (0..8)
            .map(|k| {
                Signal::new(
                    48_000,
                    vec![(0..256).map(|i| ((i + k) as f32 * 0.1).sin()).collect()],
                )
            })
            .collect();
        let batched = process_batch(&g, &batch);
        assert_eq!(batched.len(), batch.len());
        for (b, s) in batched.iter().zip(&batch) {
            let single = process(&g, s);
            for (x, y) in b.channels[0].iter().zip(&single.channels[0]) {
                assert!((x - y).abs() < 1e-5, "{x} vs {y}");
            }
        }
    }

    #[test]
    fn process_batch_falls_back_for_nonfilter() {
        // A gain chain isn't a pure SOS cascade -> per-signal fallback, still correct.
        let batch = vec![Signal::new(48_000, vec![vec![2.0, -4.0]])];
        let out = process_batch(&Graph::op(OpKind::Gain, [0.5]), &batch);
        assert_eq!(out[0].channels[0], vec![1.0, -2.0]);
    }
}

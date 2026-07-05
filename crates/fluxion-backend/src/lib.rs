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
    Biquad, FadeShape, Sos, allpass, bandpass, butterworth_highpass, butterworth_lowpass,
    certify_sos, chebyshev1_highpass, chebyshev1_lowpass, chebyshev2_highpass, chebyshev2_lowpass,
    chorus, compand, delay, echo, fade, flanger, gain, high_shelf, low_shelf, normalize_peak,
    notch, overdrive, peaking, phaser, reverb, reverse, small_gain_certify, sos_filter, tremolo,
};
use fluxion_rt::RtGraph;
use rayon::prelude::*;

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
        // Raw section from explicit coefficients — reuses the whole biquad/SOS machinery.
        OpKind::Biquad => vec![Biquad {
            b0: p[0],
            b1: p[1],
            b2: p[2],
            a1: p[3],
            a2: p[4],
        }],
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
    /// Direct-form FIR (`taps`).
    fn fir(&self, x: Self::Buf, taps: &[f32]) -> Self::Buf;
    /// Peak-normalize (cross-channel).
    fn normalize(&self, x: Self::Buf, peak: f32) -> Self::Buf;
    /// Schroeder–Moorer reverb.
    fn reverb(&self, x: Self::Buf, room: f32, damping: f32, mix: f32) -> Self::Buf;

    // --- Nonlinear / whole-signal effects ---
    //
    // These are CPU-only, non-differentiable ops. They are **defaulted** to `unimplemented!` so a
    // backend that can't express them (e.g. the autodiff Burn backend) need not implement them —
    // they never reach these ops, being rejected up front by [`is_differentiable`].

    /// Amplitude fade over `fade_in` / `fade_out` samples with a `shape` curve (0/1/2).
    fn fade(&self, _x: Self::Buf, _fade_in: usize, _fade_out: usize, _shape: f32) -> Self::Buf {
        unimplemented!("fade is a CPU-only effect — guard with is_differentiable")
    }
    /// Tremolo: amplitude LFO at `rate_hz`, dipping by `depth`.
    fn tremolo(&self, _x: Self::Buf, _rate_hz: f32, _depth: f32, _fs: u32) -> Self::Buf {
        unimplemented!("tremolo is a CPU-only effect — guard with is_differentiable")
    }
    /// Overdrive: `gain_db` of drive through a `tanh` soft-clipper with `colour` bias.
    fn overdrive(&self, _x: Self::Buf, _gain_db: f32, _colour: f32) -> Self::Buf {
        unimplemented!("overdrive is a CPU-only nonlinear effect — guard with is_differentiable")
    }
    /// Feed-forward compressor / expander (compand).
    #[allow(clippy::too_many_arguments)]
    fn compand(
        &self,
        _x: Self::Buf,
        _attack_s: f32,
        _release_s: f32,
        _threshold_db: f32,
        _ratio: f32,
        _knee_db: f32,
        _makeup_db: f32,
        _fs: u32,
    ) -> Self::Buf {
        unimplemented!("compand is a CPU-only effect — guard with is_differentiable")
    }
    /// Per-channel time reversal.
    fn reverse(&self, _x: Self::Buf) -> Self::Buf {
        unimplemented!("reverse is a whole-signal effect — guard with is_differentiable")
    }
    /// Chorus: one LFO-modulated fractional-delay voice blended by `mix`.
    fn chorus(
        &self,
        _x: Self::Buf,
        _rate_hz: f32,
        _depth_s: f32,
        _delay_s: f32,
        _mix: f32,
        _fs: u32,
    ) -> Self::Buf {
        unimplemented!("chorus is a CPU-only effect — guard with is_differentiable")
    }
    /// Flanger: a short LFO-modulated delay with `feedback`, blended by `mix`.
    #[allow(clippy::too_many_arguments)]
    fn flanger(
        &self,
        _x: Self::Buf,
        _rate_hz: f32,
        _depth_s: f32,
        _delay_s: f32,
        _feedback: f32,
        _mix: f32,
        _fs: u32,
    ) -> Self::Buf {
        unimplemented!("flanger is a CPU-only effect — guard with is_differentiable")
    }
    /// Phaser: an LFO-swept all-pass cascade with `feedback`, blended by `mix`.
    fn phaser(
        &self,
        _x: Self::Buf,
        _rate_hz: f32,
        _depth: f32,
        _feedback: f32,
        _mix: f32,
        _fs: u32,
    ) -> Self::Buf {
        unimplemented!("phaser is a CPU-only effect — guard with is_differentiable")
    }

    /// Sum two buffers (the `+` parallel combine).
    fn add(&self, a: Self::Buf, b: Self::Buf) -> Self::Buf;
    /// A feedback loop (the `~` operator): `y[n] = forward(x[n] + feedback(y)[n-1])`. A backend that
    /// can't run sample-recursive feedback (e.g. the block/autodiff engines) may leave it
    /// `unimplemented!` — guarded by [`is_differentiable`] / `to_rt_graph` returning `None`.
    fn feedback(&self, x: Self::Buf, forward: &Graph, feedback: &Graph, fs: u32) -> Self::Buf;
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
        // A named node is transparent — it runs exactly as its inner node (B8 node identity).
        Graph::Named { node, .. } => eval(b, node, x, fs),
        Graph::Feedback { forward, feedback } => b.feedback(x, forward, feedback, fs),
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
        OpKind::Fir => b.fir(x, p), // params ARE the taps (variadic op)
        OpKind::Fade => b.fade(x, samples(p[0]), samples(p[1]), p[2]),
        OpKind::Tremolo => b.tremolo(x, p[0], p[1], fs),
        OpKind::Overdrive => b.overdrive(x, p[0], p[1]),
        OpKind::Compand => b.compand(x, p[0], p[1], p[2], p[3], p[4], p[5], fs),
        OpKind::Reverse => b.reverse(x),
        OpKind::Chorus => b.chorus(x, p[0], p[1], p[2], p[3], fs),
        OpKind::Flanger => b.flanger(x, p[0], p[1], p[2], p[3], p[4], fs),
        OpKind::Phaser => b.phaser(x, p[0], p[1], p[2], p[3], fs),
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
        Graph::Named { node, .. } => is_differentiable(node, fs),
        Graph::Feedback { .. } => false, // sample-recursive; not wired into the autodiff engine
    }
}

/// The scalar CPU backend: a `Buf` is all channels of a signal, each kernel mapping over them (the
/// cross-channel `normalize` needs them together).
pub struct Cpu;

impl Backend for Cpu {
    type Buf = Vec<Vec<f32>>;

    /// Channel-parallel (rayon) when the work amortizes the fork cost; sequential otherwise.
    fn filter(&self, mut x: Vec<Vec<f32>>, sos: &[Biquad]) -> Vec<Vec<f32>> {
        // ponytail: fixed cutoff (~64k filtered samples) — an autotuned per-machine threshold is
        // follow-up work; below it the rayon fork/join overhead beats the win.
        const PAR_MIN_WORK: usize = 1 << 16;
        let work: usize = x.iter().map(Vec::len).sum::<usize>() * sos.len().max(1);
        if x.len() > 1 && work >= PAR_MIN_WORK {
            // In place: a fresh 11.5 MB Vec per channel serializes parallel channels
            // on page faults; the owned buffers are filtered where they sit.
            x.par_iter_mut()
                .for_each(|ch| fluxion_ops::sos_filter_in_place(ch, sos));
        } else {
            for ch in &mut x {
                fluxion_ops::sos_filter_in_place(ch, sos);
            }
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
    fn fir(&self, mut x: Vec<Vec<f32>>, taps: &[f32]) -> Vec<Vec<f32>> {
        for ch in &mut x {
            // Size-aware: overlap-save above the FFT threshold (long kernels).
            *ch = fluxion_ops::fir_filter_auto(ch, taps);
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
    fn fade(
        &self,
        mut x: Vec<Vec<f32>>,
        fade_in: usize,
        fade_out: usize,
        shape: f32,
    ) -> Vec<Vec<f32>> {
        let shape = FadeShape::from_param(shape);
        for ch in &mut x {
            fade(ch, fade_in, fade_out, shape);
        }
        x
    }
    fn tremolo(&self, mut x: Vec<Vec<f32>>, rate_hz: f32, depth: f32, fs: u32) -> Vec<Vec<f32>> {
        for ch in &mut x {
            tremolo(ch, rate_hz, depth, fs);
        }
        x
    }
    fn overdrive(&self, mut x: Vec<Vec<f32>>, gain_db: f32, colour: f32) -> Vec<Vec<f32>> {
        for ch in &mut x {
            overdrive(ch, gain_db, colour);
        }
        x
    }
    fn compand(
        &self,
        mut x: Vec<Vec<f32>>,
        attack_s: f32,
        release_s: f32,
        threshold_db: f32,
        ratio: f32,
        knee_db: f32,
        makeup_db: f32,
        fs: u32,
    ) -> Vec<Vec<f32>> {
        for ch in &mut x {
            *ch = compand(
                ch,
                attack_s,
                release_s,
                threshold_db,
                ratio,
                knee_db,
                makeup_db,
                fs,
            );
        }
        x
    }
    fn reverse(&self, mut x: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        for ch in &mut x {
            *ch = reverse(ch);
        }
        x
    }
    fn chorus(
        &self,
        mut x: Vec<Vec<f32>>,
        rate_hz: f32,
        depth_s: f32,
        delay_s: f32,
        mix: f32,
        fs: u32,
    ) -> Vec<Vec<f32>> {
        for ch in &mut x {
            *ch = chorus(ch, rate_hz, depth_s, delay_s, mix, fs);
        }
        x
    }
    fn flanger(
        &self,
        mut x: Vec<Vec<f32>>,
        rate_hz: f32,
        depth_s: f32,
        delay_s: f32,
        feedback: f32,
        mix: f32,
        fs: u32,
    ) -> Vec<Vec<f32>> {
        for ch in &mut x {
            *ch = flanger(ch, rate_hz, depth_s, delay_s, feedback, mix, fs);
        }
        x
    }
    fn phaser(
        &self,
        mut x: Vec<Vec<f32>>,
        rate_hz: f32,
        depth: f32,
        feedback: f32,
        mix: f32,
        fs: u32,
    ) -> Vec<Vec<f32>> {
        for ch in &mut x {
            *ch = phaser(ch, rate_hz, depth, feedback, mix, fs);
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
    fn feedback(
        &self,
        x: Vec<Vec<f32>>,
        forward: &Graph,
        feedback: &Graph,
        fs: u32,
    ) -> Vec<Vec<f32>> {
        x.into_iter()
            .map(|ch| feedback_channel(&ch, forward, feedback, fs))
            .collect()
    }
}

/// Reference CPU feedback `y[n] = forward(x[n] + feedback(y)[n-1])`, run sample-by-sample with the
/// sub-paths lowered to **stateful** [`RtGraph`]s (so filter state persists across the loop). Both
/// paths must be realtime-lowerable (filters / gain / delay / echo / reverb / FIR).
///
/// ponytail: one sample per `RtGraph::process` call — correct but O(N) dispatch; a fused feedback
/// kernel is a follow-up. Realtime and autodiff execution of `~` aren't wired yet (`to_rt_graph` /
/// `is_differentiable` return `None` / `false` for `Feedback`).
fn feedback_channel(ch: &[f32], forward: &Graph, feedback: &Graph, fs: u32) -> Vec<f32> {
    let mut fwd = to_rt_graph(forward, fs)
        .expect("feedback forward path must be realtime-lowerable (filters/gain/delay/…)");
    let mut fbk = to_rt_graph(feedback, fs)
        .expect("feedback loop-back path must be realtime-lowerable (filters/gain/delay/…)");
    fwd.prepare(1);
    fbk.prepare(1);
    let mut y = vec![0.0f32; ch.len()];
    let (mut fb_prev, mut yb, mut fbb) = (0.0f32, [0.0f32], [0.0f32]);
    for (n, &xn) in ch.iter().enumerate() {
        fwd.process(&[xn + fb_prev], &mut yb); // y[n] = forward(x[n] + feedback(y)[n-1])
        y[n] = yb[0];
        fbk.process(&yb, &mut fbb); // feedback(y)[n] → used next iteration as [n-1]
        fb_prev = fbb[0];
    }
    y
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
        Graph::Named { node, .. } => certify_graph(node, fs),
        // A general `~` loop's stability is a small-gain property of the *whole* loop; auto-certifying
        // an arbitrary feedback sub-graph isn't wired yet, so it carries no certificate (not shippable).
        Graph::Feedback { .. } => Certificate {
            verdict: Verdict::NotCertified,
            margin: f32::NAN,
        },
    }
}

fn certify_op(op: &Op, fs: u32) -> Certificate {
    // Filters — designed or a raw `Biquad` — certify from their poles.
    if let Some(sos) = op_sos(op, fs) {
        return certify_sos(&sos);
    }
    let p = &op.params;
    match op.kind {
        // Feedback ops certify via a small-gain bound on their loop coefficient.
        //
        // Echo is a feedback op: its loop gain is the (frequency-flat) feedback coefficient.
        OpKind::Echo => small_gain_certify(|_| p[1].abs(), 1),
        // Reverb is parallel damped-feedback combs; each comb's loop gain is the room-size feedback
        // (the one-pole damping only attenuates), clamped < 1 in the design → BIBO-stable.
        OpKind::Reverb => small_gain_certify(|_| p[0].clamp(0.0, 0.98), 1),
        // Flanger recirculates its delayed tap; the loop gain is the feedback coefficient.
        OpKind::Flanger => small_gain_certify(|_| p[3].abs(), 1),
        // Phaser recirculates the all-pass cascade (unity magnitude), so the loop gain is |feedback|.
        OpKind::Phaser => small_gain_certify(|_| p[2].abs(), 1),
        // Feed-forward / bounded-gain ops are unconditionally BIBO-stable. Enumerated explicitly (no
        // silent `_ => certified()`): a new op must be classified here or reach the panic below.
        OpKind::Gain
        | OpKind::Normalize
        | OpKind::Delay
        | OpKind::Fir
        | OpKind::Fade
        | OpKind::Tremolo
        | OpKind::Overdrive
        | OpKind::Compand
        | OpKind::Reverse
        | OpKind::Chorus => Certificate::certified(),
        // `OpKind` is `#[non_exhaustive]`; a new op must be classified above (or in `op_sos`) rather
        // than silently blessed as stable.
        kind => panic!(
            "fluxion-backend: op '{}' has no stability certification",
            kind.name()
        ),
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
        Graph::Named { node, .. } => graph_to_sos(node, fs), // transparent
        Graph::Parallel(..) | Graph::Feedback { .. } => None,
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
        Graph::Named { node, .. } => to_rt_graph(node, fs), // transparent
        Graph::Feedback { .. } => None, // realtime `~` (an alloc-free feedback node) is a follow-up
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
        OpKind::Reverb => Some(RtGraph::reverb(p[0], p[1], p[2])),
        OpKind::Fir => Some(RtGraph::fir(p.clone())), // taps ARE the params (G8 realtime FIR)
        // Compand is chunk-local (per-sample envelope follower) → realtime-playable.
        OpKind::Compand => Some(RtGraph::compand(p[0], p[1], p[2], p[3], p[4], p[5], fs)),
        // Not chunk-local or not-yet-wired for realtime:
        //  - Normalize / Fade / Reverse need the whole signal (global peak / total length / order).
        //  - Tremolo / Overdrive / Chorus / Flanger / Phaser are memoryless or modulated; a realtime
        //    LFO / modulated-delay node is a follow-up (ponytail).
        _ => None,
    }
}

/// Filter a flat batch of `rows.len() / frames` equal-length rows through an SOS cascade (CPU).
///
/// The batch runs in 16-row groups: each group is transposed to a
/// channel-interleaved (frame-major) layout, filtered by the SIMD
/// [`sos_filter_interleaved`](fluxion_ops::sos_filter_interleaved) kernel — the IIR recurrence
/// can't vectorize over *time*, but it vectorizes across the *batch* — and transposed back; the
/// independent groups run in parallel on rayon (plan tasks C3/C6). Per-row results are identical
/// to [`sos_filter`].
///
/// The GPU kernel (`cuda::sos_filter_batch`, `cuda` feature) is far faster in raw **compute**
/// (~59× on an RTX 3070), but a one-shot batch filter is dominated by host↔device transfer, so the
/// CPU is the right default here. The GPU pays off for **resident / repeated** workloads where the
/// data stays on the device (e.g. a differentiable training loop).
pub fn sos_filter_batch(rows: &[f32], frames: usize, sos: &[Biquad]) -> Vec<f32> {
    let n_rows = rows.len().checked_div(frames).unwrap_or(0);
    if n_rows <= 1 || sos.is_empty() {
        return sos_filter(rows, sos);
    }
    // Group size trades SIMD-lane fill (wider) against rayon task count (narrower).
    // On AVX2 the 8-row register-transpose kernel is both the fastest and the most
    // parallel choice (one vector of rows per group, no bounce buffer); elsewhere,
    // shrink from 16 toward 8 rows only when fixed groups would leave threads idle.
    #[cfg(target_arch = "x86_64")]
    let avx2 = sos.len() <= 8 && std::arch::is_x86_feature_detected!("avx2");
    #[cfg(not(target_arch = "x86_64"))]
    let avx2 = false;
    let threads = rayon::current_num_threads().max(1);
    let group = if avx2 {
        MIN_GROUP
    } else if n_rows / BATCH_GROUP >= threads {
        BATCH_GROUP
    } else {
        (n_rows / threads).clamp(MIN_GROUP, BATCH_GROUP)
    };
    let mut out = vec![0.0f32; rows.len()];
    out.par_chunks_mut(group * frames)
        .zip(rows.par_chunks(group * frames))
        .for_each(|(out_group, in_group)| dispatch_group(in_group, out_group, frames, sos));
    out
}

/// Route one row-group to the AVX2 register-transpose kernel when it applies
/// (full 8-row group, cascade depth ≤ 8, AVX2 present), else to the portable
/// gather/scatter path.
fn dispatch_group(in_group: &[f32], out_group: &mut [f32], frames: usize, sos: &[Biquad]) {
    #[cfg(target_arch = "x86_64")]
    if in_group.len() == 8 * frames && sos.len() <= 8 && std::arch::is_x86_feature_detected!("avx2")
    {
        // SAFETY: AVX2 support was just verified at runtime; the group is exactly 8 rows.
        unsafe { avx2_group::filter_group_8rows(in_group, out_group, frames, sos) };
        return;
    }
    filter_group(in_group, out_group, frames, sos);
}

/// Rows per SIMD group in [`sos_filter_batch`]: wide enough to fill AVX lanes with headroom for
/// superscalar overlap, narrow enough that a time-block's working set stays cache-resident.
const BATCH_GROUP: usize = 16;

/// Smallest useful group: one AVX2 f32 vector of rows.
const MIN_GROUP: usize = 8;

/// Frames per time-block inside a group. The interleave/deinterleave transposes only ever touch
/// `BATCH_GROUP × BLOCK_FRAMES` floats (≈256 KB) of hot scratch, while the big planar buffers are
/// read/written in sequential per-row runs — this avoids the power-of-two-stride cache-set aliasing
/// that makes a whole-signal transpose memory-bound (rows sit exactly `frames·4` bytes apart).
const BLOCK_FRAMES: usize = 4096;

/// Filter one row-group: gather a time block into an interleaved hot buffer, run the SIMD cascade
/// with carried per-channel state, scatter back — block by block.
fn filter_group(in_group: &[f32], out_group: &mut [f32], frames: usize, sos: &[Biquad]) {
    let group_rows = in_group.len() / frames;
    let mut inter = vec![0.0f32; group_rows * BLOCK_FRAMES.min(frames)];
    let mut state = vec![0.0f32; sos.len() * 2 * group_rows];

    let mut t0 = 0usize;
    while t0 < frames {
        let block = BLOCK_FRAMES.min(frames - t0);
        let inter = &mut inter[..group_rows * block];
        // Gather, row-outer: sequential planar reads; the strided hot-buffer writes stay in L2.
        // (Frame-outer would touch all 16 planar rows per frame — they sit exactly `frames·4`
        // bytes apart, a power-of-two stride that aliases to one cache set and thrashes; measured
        // ~35% slower.)
        for (r, row) in in_group.chunks_exact(frames).enumerate() {
            for (t, &x) in row[t0..t0 + block].iter().enumerate() {
                inter[t * group_rows + r] = x;
            }
        }
        fluxion_ops::sos_filter_interleaved_chunk(inter, group_rows, sos, &mut state);
        // Scatter, row-outer: strided hot-buffer reads, sequential planar writes.
        for (r, row) in out_group.chunks_exact_mut(frames).enumerate() {
            for (t, y) in row[t0..t0 + block].iter_mut().enumerate() {
                *y = inter[t * group_rows + r];
            }
        }
        t0 += block;
    }
}

/// Process a batch of signals. A pure-filter chain over equal-length **mono** signals is routed to
/// the batched SIMD path; anything else processes each signal independently in parallel (rayon).
pub fn process_batch(graph: &Graph, batch: &[Signal]) -> Vec<Signal> {
    if let Some(out) = try_batched_filter(graph, batch) {
        return out;
    }
    batch.par_iter().map(|s| process(graph, s)).collect()
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

/// AVX2 fast path for [`sos_filter_batch`]: one full 8-row group is filtered with
/// strided planar loads, an in-register 8×8 transpose, the fused K-section vector
/// cascade, and a transpose back to planar stores — no interleaved bounce buffer,
/// so the group touches each planar byte exactly once in and once out (half the
/// memory traffic of the gather/scatter path, which is what bounds it at full
/// thread count). Arithmetic per (section, frame, row) is identical to the scalar
/// kernel: same operand order, `mul`/`add`/`sub` intrinsics only (no FMA
/// contraction), so results stay bit-identical.
#[cfg(target_arch = "x86_64")]
mod avx2_group {
    use super::BLOCK_FRAMES;
    use fluxion_ops::Biquad;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    /// # Safety
    /// Caller must have verified AVX2 support; `in_group`/`out_group` must be exactly
    /// `8 * frames` long and `sos.len()` must be in `1..=8`.
    #[target_feature(enable = "avx2")]
    pub unsafe fn filter_group_8rows(
        in_group: &[f32],
        out_group: &mut [f32],
        frames: usize,
        sos: &[Biquad],
    ) {
        debug_assert_eq!(in_group.len(), 8 * frames);
        debug_assert_eq!(out_group.len(), 8 * frames);
        // SAFETY: forwarding the caller's contract (AVX2 verified, exact 8-row group).
        unsafe {
            match sos.len() {
                1 => run::<1>(in_group, out_group, frames, sos),
                2 => run::<2>(in_group, out_group, frames, sos),
                3 => run::<3>(in_group, out_group, frames, sos),
                4 => run::<4>(in_group, out_group, frames, sos),
                5 => run::<5>(in_group, out_group, frames, sos),
                6 => run::<6>(in_group, out_group, frames, sos),
                7 => run::<7>(in_group, out_group, frames, sos),
                8 => run::<8>(in_group, out_group, frames, sos),
                _ => unreachable!("dispatch_group only routes sos.len() <= 8 here"),
            }
        }
    }

    /// Padding (in f32s) added to the scratch row pitch so the 8 tile rows land in
    /// *different* L1/L2 cache sets. Planar batch rows sit exactly `frames·4` bytes
    /// apart — for the paper's 524k-frame workload that is a 2 MB power-of-two
    /// stride, which maps every row (and the output rows) to the same set: 16
    /// streams contending for 8 L1 ways thrash on every 8×8 tile and the kernel
    /// runs at LLC speed, killing multi-core scaling. A 16-float pad makes the
    /// pitch 257 cache lines, so consecutive rows shift by one set.
    const PITCH_PAD: usize = 16;

    #[target_feature(enable = "avx2")]
    unsafe fn run<const K: usize>(input: &[f32], out: &mut [f32], frames: usize, sos: &[Biquad]) {
        let b0: [__m256; K] = core::array::from_fn(|k| _mm256_set1_ps(sos[k].b0));
        let b1: [__m256; K] = core::array::from_fn(|k| _mm256_set1_ps(sos[k].b1));
        let b2: [__m256; K] = core::array::from_fn(|k| _mm256_set1_ps(sos[k].b2));
        let a1: [__m256; K] = core::array::from_fn(|k| _mm256_set1_ps(sos[k].a1));
        let a2: [__m256; K] = core::array::from_fn(|k| _mm256_set1_ps(sos[k].a2));
        let coeffs = (b0, b1, b2, a1, a2);
        let mut s1 = [_mm256_setzero_ps(); K];
        let mut s2 = [_mm256_setzero_ps(); K];

        // Bounce each time block through a small padded-pitch scratch: the gathers
        // and scatters are sequential per-row memcpys (no simultaneous strided
        // streams), and the kernel's 8 concurrent row streams read the scratch at a
        // set-spreading pitch instead of the pathological planar stride. The whole
        // scratch (8 × ~16 KB) stays L2-resident. Measured on the i9-10900KF
        // (Comet Lake, 32 KB 8-way L1): 64×524k batch 1.66 → 2.60 Gsamples/s
        // multi-thread (+57%) for a ~5% single-thread cost (the two extra copies);
        // direct strided loads thrash there because a 2 MB row stride lands all 16
        // streams in one L1/L2 set.
        let block_cap = BLOCK_FRAMES.min(frames);
        let pitch = block_cap + PITCH_PAD;
        let mut scratch = vec![0.0f32; 8 * pitch];
        let mut t0 = 0usize;
        while t0 < frames {
            let block = block_cap.min(frames - t0);
            for r in 0..8 {
                scratch[r * pitch..r * pitch + block]
                    .copy_from_slice(&input[r * frames + t0..r * frames + t0 + block]);
            }
            // SAFETY: AVX2 verified by the caller; scratch rows are `pitch` apart and
            // `block <= pitch` frames long.
            unsafe { tile::<K>(&mut scratch, pitch, block, &coeffs, &mut s1, &mut s2, sos) };
            for r in 0..8 {
                out[r * frames + t0..r * frames + t0 + block]
                    .copy_from_slice(&scratch[r * pitch..r * pitch + block]);
            }
            t0 += block;
        }
    }

    /// Coefficient vectors for a K-section cascade, splatted per lane.
    type CoeffVecs<const K: usize> = (
        [__m256; K],
        [__m256; K],
        [__m256; K],
        [__m256; K],
        [__m256; K],
    );

    /// Filter one time block **in place** on the padded scratch, carrying the
    /// per-lane section state across calls. Identical operand order to the scalar
    /// recurrence, so per-row results match [`fluxion_ops::sos_filter`] exactly.
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn tile<const K: usize>(
        scratch: &mut [f32],
        pitch: usize,
        block: usize,
        (b0, b1, b2, a1, a2): &CoeffVecs<K>,
        s1: &mut [__m256; K],
        s2: &mut [__m256; K],
        sos: &[Biquad],
    ) {
        let full = block - block % 8;
        let src = scratch.as_mut_ptr();
        let mut t = 0usize;
        while t < full {
            // SAFETY: rows r in 0..8, frames t..t+8 are in bounds (t + 8 <= full <= block <= pitch).
            let mut v: [__m256; 8] =
                core::array::from_fn(|r| unsafe { _mm256_loadu_ps(src.add(r * pitch + t)) });
            // SAFETY: AVX2 verified by the caller; arithmetic intrinsics are register-only.
            unsafe {
                transpose8x8(&mut v);
                for vi in v.iter_mut() {
                    let mut x = *vi;
                    for k in 0..K {
                        // y = b0*x + s1; s1 = (b1*x - a1*y) + s2; s2 = b2*x - a2*y
                        let y = _mm256_add_ps(_mm256_mul_ps(b0[k], x), s1[k]);
                        s1[k] = _mm256_add_ps(
                            _mm256_sub_ps(_mm256_mul_ps(b1[k], x), _mm256_mul_ps(a1[k], y)),
                            s2[k],
                        );
                        s2[k] = _mm256_sub_ps(_mm256_mul_ps(b2[k], x), _mm256_mul_ps(a2[k], y));
                        x = y;
                    }
                    *vi = x;
                }
                transpose8x8(&mut v);
            }
            for (r, vi) in v.iter().enumerate() {
                // SAFETY: same bounds as the loads above.
                unsafe { _mm256_storeu_ps(src.add(r * pitch + t), *vi) };
            }
            t += 8;
        }

        if t < block {
            // Tail frames: spill the vector states to per-row scalars, finish with
            // the exact scalar recurrence (identical operand order), and reload the
            // vectors so a following block continues from the right state.
            let mut s1a = [[0.0f32; 8]; K];
            let mut s2a = [[0.0f32; 8]; K];
            for k in 0..K {
                // SAFETY: arrays are 8 f32s; storeu has no alignment requirement.
                unsafe {
                    _mm256_storeu_ps(s1a[k].as_mut_ptr(), s1[k]);
                    _mm256_storeu_ps(s2a[k].as_mut_ptr(), s2[k]);
                }
            }
            for r in 0..8 {
                for tt in t..block {
                    let mut x = scratch[r * pitch + tt];
                    for k in 0..K {
                        let y = sos[k].b0 * x + s1a[k][r];
                        s1a[k][r] = sos[k].b1 * x - sos[k].a1 * y + s2a[k][r];
                        s2a[k][r] = sos[k].b2 * x - sos[k].a2 * y;
                        x = y;
                    }
                    scratch[r * pitch + tt] = x;
                }
            }
            for k in 0..K {
                // SAFETY: arrays are 8 f32s; loadu has no alignment requirement.
                unsafe {
                    s1[k] = _mm256_loadu_ps(s1a[k].as_ptr());
                    s2[k] = _mm256_loadu_ps(s2a[k].as_ptr());
                }
            }
        }
    }

    /// Canonical AVX2 8×8 f32 in-register transpose (unpack / shuffle / permute2f128).
    #[inline(always)]
    unsafe fn transpose8x8(v: &mut [__m256; 8]) {
        // SAFETY: register-only shuffles; AVX2 verified by the caller chain.
        unsafe {
            let t0 = _mm256_unpacklo_ps(v[0], v[1]);
            let t1 = _mm256_unpackhi_ps(v[0], v[1]);
            let t2 = _mm256_unpacklo_ps(v[2], v[3]);
            let t3 = _mm256_unpackhi_ps(v[2], v[3]);
            let t4 = _mm256_unpacklo_ps(v[4], v[5]);
            let t5 = _mm256_unpackhi_ps(v[4], v[5]);
            let t6 = _mm256_unpacklo_ps(v[6], v[7]);
            let t7 = _mm256_unpackhi_ps(v[6], v[7]);
            let u0 = _mm256_shuffle_ps(t0, t2, 0x44);
            let u1 = _mm256_shuffle_ps(t0, t2, 0xEE);
            let u2 = _mm256_shuffle_ps(t1, t3, 0x44);
            let u3 = _mm256_shuffle_ps(t1, t3, 0xEE);
            let u4 = _mm256_shuffle_ps(t4, t6, 0x44);
            let u5 = _mm256_shuffle_ps(t4, t6, 0xEE);
            let u6 = _mm256_shuffle_ps(t5, t7, 0x44);
            let u7 = _mm256_shuffle_ps(t5, t7, 0xEE);
            v[0] = _mm256_permute2f128_ps(u0, u4, 0x20);
            v[1] = _mm256_permute2f128_ps(u1, u5, 0x20);
            v[2] = _mm256_permute2f128_ps(u2, u6, 0x20);
            v[3] = _mm256_permute2f128_ps(u3, u7, 0x20);
            v[4] = _mm256_permute2f128_ps(u0, u4, 0x31);
            v[5] = _mm256_permute2f128_ps(u1, u5, 0x31);
            v[6] = _mm256_permute2f128_ps(u2, u6, 0x31);
            v[7] = _mm256_permute2f128_ps(u3, u7, 0x31);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        FrozenSos, certify_graph, freeze, graph_to_sos, is_differentiable, process, process_batch,
        sos_filter_batch, to_rt_graph,
    };
    use fluxion_core::{Graph, Op, OpKind, Signal};

    fn sig(samples: Vec<f32>) -> Signal {
        Signal::new(48_000, vec![samples])
    }

    /// Build a graph from an op kind and its default parameter vector.
    fn op_default(kind: OpKind) -> Graph {
        Graph::Op(Op::new(kind, kind.defaults()).unwrap())
    }

    #[test]
    fn biquad_op_is_a_filter_freezes_and_lowers() {
        // A raw one-pole low-pass biquad: y[n] = 0.2·x[n] + 0.8·y[n-1], DC gain 0.2/(1-0.8) = 1.
        let g = Graph::op(OpKind::Biquad, [0.2, 0.0, 0.0, -0.8, 0.0]);
        let fs = 48_000;
        assert!(is_differentiable(&g, fs)); // a biquad IS differentiable (reuses the SOS VJP)
        assert_eq!(graph_to_sos(&g, fs).unwrap().len(), 1);
        assert!(freeze(&g, fs).is_some());
        assert!(certify_graph(&g, fs).verdict.is_shippable());

        // DC passes at unity, and streaming matches the offline process.
        let x: Vec<f32> = (0..1_000).map(|i| (0.05 * i as f32).sin()).collect();
        let batch = process(&g, &Signal::new(fs, vec![x.clone()]))
            .channels
            .remove(0);
        assert!((process(&g, &sig(vec![1.0; 500])).channels[0][499] - 1.0).abs() < 1e-3);

        let mut rt = to_rt_graph(&g, fs).unwrap();
        rt.prepare(64);
        let mut o = vec![0.0f32; 64];
        let mut streamed = Vec::new();
        for chunk in x.chunks(64) {
            let o = &mut o[..chunk.len()];
            rt.process(chunk, o);
            streamed.extend_from_slice(o);
        }
        for (a, b) in streamed.iter().zip(&batch) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn nonlinear_ops_process_on_cpu_but_arent_realtime_or_differentiable() {
        let fs = 48_000;
        // Overdrive is a CPU waveshaper: hard drive saturates, silence stays silent.
        let od = process(
            &Graph::op(OpKind::Overdrive, [40.0, 0.0]),
            &sig(vec![0.0, 5.0, -5.0]),
        );
        assert!(od.channels[0][0].abs() < 1e-6 && od.channels[0][1] > 0.9);
        // Reverse flips the channel in time.
        assert_eq!(
            process(&op_default(OpKind::Reverse), &sig(vec![1.0, 2.0, 3.0])).channels[0],
            vec![3.0, 2.0, 1.0]
        );
        // These are neither differentiable nor realtime-lowerable (whole-signal or modulated).
        for kind in [
            OpKind::Fade,
            OpKind::Tremolo,
            OpKind::Overdrive,
            OpKind::Reverse,
            OpKind::Chorus,
            OpKind::Flanger,
            OpKind::Phaser,
        ] {
            let g = op_default(kind);
            assert!(
                !is_differentiable(&g, fs),
                "{} should not differentiate",
                kind.name()
            );
            assert!(
                to_rt_graph(&g, fs).is_none(),
                "{} should not be realtime",
                kind.name()
            );
        }
    }

    #[test]
    fn compand_lowers_to_realtime_and_certifies() {
        let g = Graph::op(OpKind::Compand, [0.005, 0.05, -20.0, 4.0, 6.0, 0.0]);
        let fs = 48_000;
        assert!(certify_graph(&g, fs).verdict.is_shippable()); // feed-forward, unconditionally stable
        let x: Vec<f32> = (0..3_000).map(|i| 0.8 * (0.05 * i as f32).sin()).collect();
        let batch = process(&g, &Signal::new(fs, vec![x.clone()]))
            .channels
            .remove(0);
        let mut rt = to_rt_graph(&g, fs).unwrap();
        rt.prepare(128);
        let mut o = vec![0.0f32; 128];
        let mut streamed = Vec::new();
        for chunk in x.chunks(128) {
            let o = &mut o[..chunk.len()];
            rt.process(chunk, o);
            streamed.extend_from_slice(o);
        }
        for (a, b) in streamed.iter().zip(&batch) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn certify_op_classifies_every_op_kind() {
        // Every op must be explicitly classified in `certify_op` (filter / feedback / feed-forward) —
        // an unclassified op reaches the panic arm and fails this test rather than being silently
        // blessed by a catch-all.
        for &kind in OpKind::all() {
            let c = certify_graph(&op_default(kind), 48_000);
            assert!(c.margin.is_nan() || c.margin.is_finite(), "{}", kind.name());
        }
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
    fn to_rt_graph_lowers_reverb() {
        // Reverb is now realtime-playable (G7): lowering + streaming matches the offline process.
        let g =
            Graph::op(OpKind::Lowpass, [3_000.0, 2.0]) | Graph::op(OpKind::Reverb, [0.8, 0.3, 0.5]);
        let fs = 48_000;
        let x: Vec<f32> = (0..6_000).map(|i| (0.05 * i as f32).sin()).collect();
        let batch = process(&g, &Signal::new(fs, vec![x.clone()]))
            .channels
            .remove(0);

        let mut rt = to_rt_graph(&g, fs).unwrap();
        rt.prepare(256);
        let mut out = vec![0.0f32; 256];
        let mut streamed = Vec::with_capacity(x.len());
        for chunk in x.chunks(256) {
            let out = &mut out[..chunk.len()];
            rt.process(chunk, out);
            streamed.extend_from_slice(out);
        }
        assert_eq!(streamed.len(), batch.len());
        for (i, (a, b)) in streamed.iter().zip(&batch).enumerate() {
            assert!((a - b).abs() < 1e-3, "at {i}: {a} vs {b}");
        }
    }

    #[test]
    fn fir_op_processes_and_lowers_to_realtime() {
        // FIR is now a first-class graph op (D13): CPU process == offline fir_filter, and it lowers
        // to the RtGraph::Fir node (G8) so a trained FIR is realtime-playable.
        let taps = vec![0.2f32, -0.5, 0.3, 0.1, -0.05];
        let g = Graph::op(OpKind::Fir, taps.clone());
        let fs = 48_000;
        let x: Vec<f32> = (0..500).map(|i| (0.1 * i as f32).sin()).collect();
        let want = fluxion_ops::fir_filter(&x, &taps);

        let out = process(&g, &Signal::new(fs, vec![x.clone()]))
            .channels
            .remove(0);
        for (a, b) in out.iter().zip(&want) {
            assert!((a - b).abs() < 1e-5, "process {a} vs {b}");
        }

        let mut rt = to_rt_graph(&g, fs).unwrap();
        rt.prepare(64);
        let mut o = vec![0.0f32; 64];
        let mut streamed = Vec::new();
        for chunk in x.chunks(64) {
            let o = &mut o[..chunk.len()];
            rt.process(chunk, o);
            streamed.extend_from_slice(o);
        }
        for (a, b) in streamed.iter().zip(&want) {
            assert!((a - b).abs() < 1e-4, "rt {a} vs {b}");
        }
    }

    #[test]
    fn process_runs_feedback_loops() {
        let fs = 48_000;
        // gain(1) ~ gain(0.5): a first-order IIR, y[n] = x[n] + 0.5·y[n-1] → impulse response 0.5ⁿ.
        let g = Graph::op(OpKind::Gain, [1.0]).feedback(Graph::op(OpKind::Gain, [0.5]));
        let mut impulse = vec![0.0f32; 16];
        impulse[0] = 1.0;
        let y = process(&g, &Signal::new(fs, vec![impulse]))
            .channels
            .remove(0);
        for (n, &v) in y.iter().enumerate() {
            assert!((v - 0.5f32.powi(n as i32)).abs() < 1e-5, "y[{n}] = {v}");
        }
        // A filter *inside* the loop (which eval_ref can't do): the small-gain loop stays bounded.
        let g2 =
            Graph::op(OpKind::Lowpass, [4_000.0, 2.0]).feedback(Graph::op(OpKind::Gain, [0.3]));
        let x: Vec<f32> = (0..2_000).map(|i| (0.1 * i as f32).sin()).collect();
        let y2 = process(&g2, &Signal::new(fs, vec![x])).channels.remove(0);
        assert!(
            y2.iter().all(|v| v.is_finite() && v.abs() < 50.0),
            "feedback loop blew up"
        );
    }

    #[test]
    fn sos_filter_batch_matches_per_row_scalar() {
        // The SIMD group path (transpose → interleaved → transpose) must equal per-row sos_filter,
        // including partial groups (rows % BATCH_GROUP ≠ 0) and partial transpose tiles
        // (frames % 32 ≠ 0).
        let sos = fluxion_ops::butterworth_lowpass(6, 3_000.0, 48_000);
        // 9_999 and 5_000 cross the AVX2 bounce-tile boundary (BLOCK_FRAMES) so the
        // carried section state and the final-block scalar tail are both exercised.
        let sizes = [
            (2usize, 100usize),
            (16, 333),
            (37, 1000),
            (5, 7),
            (8, 9_999),
            (24, 5_000),
        ];
        for (rows, frames) in sizes {
            let flat: Vec<f32> = (0..rows * frames)
                .map(|i| ((i % 97) as f32 * 0.13).sin())
                .collect();
            let got = sos_filter_batch(&flat, frames, &sos);
            for r in 0..rows {
                let want = fluxion_ops::sos_filter(&flat[r * frames..(r + 1) * frames], &sos);
                for (a, b) in got[r * frames..(r + 1) * frames].iter().zip(&want) {
                    assert!(
                        a.to_bits() == b.to_bits(),
                        "bit mismatch rows={rows} frames={frames}: {a} vs {b}"
                    );
                }
            }
        }
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

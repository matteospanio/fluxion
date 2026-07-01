//! General-graph realtime executor (plan task G3, beyond a linear cascade).
//!
//! [`SosStream`] runs one cascade; [`RtGraph`] runs the full graph algebra — `Series` (chain) and
//! `Parallel` (sum) composition of filter, gain, delay, echo, comb/all-pass (reverb) and FIR nodes —
//! block-by-block and allocation-free. The intermediate buffers that `Series`/`Parallel` need are sized once by
//! [`RtGraph::prepare`]; delay/echo own their fixed-size rings; after that [`RtGraph::process`]
//! never allocates, so it is audio-thread safe.
//!
//! Mirrors `fluxion_core::Graph`'s `|` (series) and `+` (parallel). Lowering an arbitrary
//! `fluxion_core::Graph` (designing each op's coefficients) to an `RtGraph` is
//! `fluxion_backend::to_rt_graph` (like `freeze`); here are the runtime building blocks.

use fluxion_ops::Biquad;

use crate::stream::SosStream;

/// A realtime processing graph: a tree of filter/gain leaves composed in series and parallel.
#[derive(Debug, Clone)]
pub enum RtGraph {
    /// An SOS cascade.
    Filter(SosStream),
    /// A constant gain (multiply).
    Gain(f32),
    /// Run `first`, feed its output into `second`. `scratch` holds the intermediate.
    Series {
        first: Box<RtGraph>,
        second: Box<RtGraph>,
        scratch: Vec<f32>,
    },
    /// Run `left` and `right` on the same input and sum. `scratch` holds the right branch's output.
    Parallel {
        left: Box<RtGraph>,
        right: Box<RtGraph>,
        scratch: Vec<f32>,
    },
    /// Single delayed tap crossfaded with the dry signal: `(1-mix)·x[n] + mix·x[n-d]`. `ring` is the
    /// `d`-sample history of inputs; `idx` is the write cursor.
    Delay {
        ring: Vec<f32>,
        idx: usize,
        mix: f32,
    },
    /// Feedback echo: `out = x[n] + wet·w[n]`, `w[n] = x[n-d] + feedback·w[n-d]`. `xring`/`wring` are
    /// the `d`-sample histories of the input and of `w`.
    Echo {
        xring: Vec<f32>,
        wring: Vec<f32>,
        idx: usize,
        feedback: f32,
        wet: f32,
    },
    /// Fractional delayed tap, linear-interpolated: `(1-mix)·x[n] + mix·x[n-D]` for a possibly
    /// non-integer `D = i + frac` (`xd = (1-frac)·x[n-i] + frac·x[n-i-1]`). `ring` is the input
    /// history; `w` is the write cursor.
    DelayFrac {
        ring: Vec<f32>,
        w: usize,
        i: usize,
        frac: f32,
        mix: f32,
    },
    /// Damped feedback comb: `y[n] = x[n] + g·lp(y[n-d])`, `lp = yd·(1-damp) + lp·damp` (the reverb
    /// building block). `yring` is the `d`-sample history of `y`; `lp` is the one-pole state.
    Comb {
        yring: Vec<f32>,
        idx: usize,
        lp: f32,
        g: f32,
        damp: f32,
    },
    /// Schroeder all-pass: `y[n] = -g·x[n] + x[n-d] + g·y[n-d]` (diffuses phase, flat magnitude).
    /// `xring`/`yring` are the `d`-sample histories of the input and output.
    Allpass {
        xring: Vec<f32>,
        yring: Vec<f32>,
        idx: usize,
        g: f32,
    },
    /// Direct-form FIR: `y[n] = Σ_k taps[k]·x[n-k]` (the realtime form of a trained/frozen FIR).
    /// `ring` holds the last `taps.len()` inputs; `idx` is the write cursor.
    Fir {
        taps: Vec<f32>,
        ring: Vec<f32>,
        idx: usize,
    },
}

impl RtGraph {
    /// A filter leaf from a frozen cascade.
    pub fn filter(sos: Vec<Biquad>) -> Self {
        RtGraph::Filter(SosStream::new(sos))
    }

    /// A constant-gain leaf.
    pub fn gain(g: f32) -> Self {
        RtGraph::Gain(g)
    }

    /// Series composition: `first` then `second` (the `|` of the graph algebra).
    pub fn series(first: RtGraph, second: RtGraph) -> Self {
        RtGraph::Series {
            first: Box::new(first),
            second: Box::new(second),
            scratch: Vec::new(),
        }
    }

    /// Parallel composition: `left + right`, summed (the `+` of the graph algebra).
    pub fn parallel(left: RtGraph, right: RtGraph) -> Self {
        RtGraph::Parallel {
            left: Box::new(left),
            right: Box::new(right),
            scratch: Vec::new(),
        }
    }

    /// A delayed-tap node: `(1-mix)·x[n] + mix·x[n-d]`, `d = max(1, delay_samples)`. The `d`-sample
    /// history is allocated here, so [`process`](Self::process) stays alloc-free.
    pub fn delay(delay_samples: usize, mix: f32) -> Self {
        let d = delay_samples.max(1);
        RtGraph::Delay {
            ring: vec![0.0; d],
            idx: 0,
            mix,
        }
    }

    /// A feedback-echo node: `x[n] + wet·w[n]`, `w[n] = x[n-d] + feedback·w[n-d]`,
    /// `d = max(1, delay_samples)`.
    pub fn echo(delay_samples: usize, feedback: f32, wet: f32) -> Self {
        let d = delay_samples.max(1);
        RtGraph::Echo {
            xring: vec![0.0; d],
            wring: vec![0.0; d],
            idx: 0,
            feedback,
            wet,
        }
    }

    /// A fractional (linear-interpolated) delay node: delays by `delay` samples, which may be
    /// non-integer — useful for chorus/flanger-style modulated delays and sub-sample tuning. The
    /// `⌈delay⌉+2`-sample history is allocated here, so [`process`](Self::process) stays alloc-free.
    pub fn delay_frac(delay: f32, mix: f32) -> Self {
        let delay = delay.max(0.0);
        let i = delay.floor() as usize;
        let frac = delay - i as f32;
        RtGraph::DelayFrac {
            ring: vec![0.0; i + 2],
            w: 0,
            i,
            frac,
            mix,
        }
    }

    /// A damped feedback-comb leaf: `y[n] = x[n] + g·lp(y[n-d])`, `d = max(1, delay_samples)`.
    pub fn comb(delay_samples: usize, g: f32, damp: f32) -> Self {
        RtGraph::Comb {
            yring: vec![0.0; delay_samples.max(1)],
            idx: 0,
            lp: 0.0,
            g,
            damp,
        }
    }

    /// A Schroeder all-pass leaf: `y[n] = -g·x[n] + x[n-d] + g·y[n-d]`, `d = max(1, delay_samples)`.
    pub fn allpass(delay_samples: usize, g: f32) -> Self {
        let d = delay_samples.max(1);
        RtGraph::Allpass {
            xring: vec![0.0; d],
            yring: vec![0.0; d],
            idx: 0,
            g,
        }
    }

    /// A direct-form FIR leaf from a tap vector — the realtime form of a frozen/trained FIR:
    /// `y[n] = Σ_k taps[k]·x[n-k]`. The `taps.len()`-sample history is allocated here, so
    /// [`process`](Self::process) stays alloc-free.
    pub fn fir(taps: Vec<f32>) -> Self {
        let n = taps.len().max(1);
        RtGraph::Fir {
            taps,
            ring: vec![0.0; n],
            idx: 0,
        }
    }

    /// A realtime Schroeder–Moorer reverb, built from the same Freeverb topology as the offline
    /// [`fluxion_ops::reverb`]: four parallel damped combs, averaged, then two series all-passes, wet/
    /// dry-blended by `mix`. `room_size` sets the comb feedback (clamped `< 1` for BIBO stability),
    /// `damping` rolls off the tail's high end. Streaming it matches the offline reverb sample-for-
    /// sample; assembled purely from [`comb`](Self::comb)/[`allpass`](Self::allpass) leaves and the
    /// series/parallel/gain algebra, so it is alloc-free after [`prepare`](Self::prepare).
    pub fn reverb(room_size: f32, damping: f32, mix: f32) -> Self {
        const COMB_DELAYS: [usize; 4] = [1557, 1617, 1491, 1422];
        const ALLPASS_DELAYS: [usize; 2] = [225, 556];
        let g = room_size.clamp(0.0, 0.98);
        let damp = damping.clamp(0.0, 1.0);

        // Sum of the four combs, then average (×0.25).
        let mut combs = RtGraph::comb(COMB_DELAYS[0], g, damp);
        for &d in &COMB_DELAYS[1..] {
            combs = RtGraph::parallel(combs, RtGraph::comb(d, g, damp));
        }
        let mut wet = RtGraph::series(combs, RtGraph::gain(0.25));
        // Series all-pass diffusion.
        for &d in &ALLPASS_DELAYS {
            wet = RtGraph::series(wet, RtGraph::allpass(d, 0.5));
        }
        // Wet/dry: (1-mix)·x + mix·wet — both branches on the same input, summed.
        RtGraph::parallel(
            RtGraph::gain(1.0 - mix),
            RtGraph::series(wet, RtGraph::gain(mix)),
        )
    }

    /// Pre-size every internal scratch buffer for blocks up to `max_block` samples. This is the only
    /// allocating step — call it before going realtime. Blocks passed to [`process`](Self::process)
    /// must not exceed `max_block`.
    pub fn prepare(&mut self, max_block: usize) {
        match self {
            // A filter leaf pre-sizes its crossfade scratch (for a live `set_coeffs`).
            RtGraph::Filter(s) => s.prepare(max_block),
            // The other leaves carry their own fixed-size state (delay rings) — no scratch.
            RtGraph::Gain(_)
            | RtGraph::Delay { .. }
            | RtGraph::Echo { .. }
            | RtGraph::DelayFrac { .. }
            | RtGraph::Comb { .. }
            | RtGraph::Allpass { .. }
            | RtGraph::Fir { .. } => {}
            RtGraph::Series {
                first,
                second,
                scratch,
            } => {
                scratch.resize(max_block, 0.0);
                first.prepare(max_block);
                second.prepare(max_block);
            }
            RtGraph::Parallel {
                left,
                right,
                scratch,
            } => {
                scratch.resize(max_block, 0.0);
                left.prepare(max_block);
                right.prepare(max_block);
            }
        }
    }

    /// Process one block in place. Allocation-free after [`prepare`](Self::prepare). `input` and
    /// `output` must be equal length and `≤ max_block`.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), output.len(), "block in/out length mismatch");
        match self {
            RtGraph::Filter(s) => s.process_block(input, output),
            RtGraph::Gain(g) => {
                let g = *g;
                for (o, i) in output.iter_mut().zip(input) {
                    *o = *i * g;
                }
            }
            RtGraph::Series {
                first,
                second,
                scratch,
            } => {
                let n = input.len();
                first.process(input, &mut scratch[..n]);
                second.process(&scratch[..n], output);
            }
            RtGraph::Parallel {
                left,
                right,
                scratch,
            } => {
                let n = input.len();
                left.process(input, output);
                right.process(input, &mut scratch[..n]);
                for (o, t) in output.iter_mut().zip(&scratch[..n]) {
                    *o += *t;
                }
            }
            RtGraph::Delay { ring, idx, mix } => {
                let d = ring.len();
                let mix = *mix;
                for (o, &x) in output.iter_mut().zip(input) {
                    let xd = ring[*idx]; // x[n-d]
                    *o = (1.0 - mix) * x + mix * xd;
                    ring[*idx] = x;
                    *idx += 1;
                    if *idx == d {
                        *idx = 0;
                    }
                }
            }
            RtGraph::Echo {
                xring,
                wring,
                idx,
                feedback,
                wet,
            } => {
                let d = xring.len();
                let (fb, wet) = (*feedback, *wet);
                for (o, &x) in output.iter_mut().zip(input) {
                    let w = xring[*idx] + fb * wring[*idx]; // x[n-d] + feedback·w[n-d]
                    *o = x + wet * w;
                    xring[*idx] = x;
                    wring[*idx] = w;
                    *idx += 1;
                    if *idx == d {
                        *idx = 0;
                    }
                }
            }
            RtGraph::DelayFrac {
                ring,
                w,
                i,
                frac,
                mix,
            } => {
                let n = ring.len();
                let (i, frac, mix) = (*i, *frac, *mix);
                for (o, &x) in output.iter_mut().zip(input) {
                    ring[*w] = x; // write current, then read taps relative to the new cursor
                    *w += 1;
                    if *w == n {
                        *w = 0;
                    }
                    let a = ring[(*w + n - 1 - i) % n]; // x[n-i]
                    let b = ring[(*w + n - 2 - i) % n]; // x[n-i-1]
                    let xd = (1.0 - frac) * a + frac * b;
                    *o = (1.0 - mix) * x + mix * xd;
                }
            }
            RtGraph::Comb {
                yring,
                idx,
                lp,
                g,
                damp,
            } => {
                let d = yring.len();
                let (g, damp) = (*g, *damp);
                for (o, &x) in output.iter_mut().zip(input) {
                    let yd = yring[*idx]; // y[n-d]
                    *lp = yd * (1.0 - damp) + *lp * damp;
                    let y = x + g * *lp;
                    *o = y;
                    yring[*idx] = y;
                    *idx += 1;
                    if *idx == d {
                        *idx = 0;
                    }
                }
            }
            RtGraph::Allpass {
                xring,
                yring,
                idx,
                g,
            } => {
                let d = xring.len();
                let g = *g;
                for (o, &x) in output.iter_mut().zip(input) {
                    let y = -g * x + xring[*idx] + g * yring[*idx]; // -g·x + x[n-d] + g·y[n-d]
                    *o = y;
                    xring[*idx] = x;
                    yring[*idx] = y;
                    *idx += 1;
                    if *idx == d {
                        *idx = 0;
                    }
                }
            }
            RtGraph::Fir { taps, ring, idx } => {
                let n = ring.len();
                for (o, &x) in output.iter_mut().zip(input) {
                    ring[*idx] = x; // x[n] at idx; x[n-k] at (idx + n - k) mod n
                    let mut acc = 0.0f32;
                    for (k, &h) in taps.iter().enumerate() {
                        acc += h * ring[(*idx + n - k) % n];
                    }
                    *o = acc;
                    *idx += 1;
                    if *idx == n {
                        *idx = 0;
                    }
                }
            }
        }
    }

    /// Reset all stateful nodes (filter state, delay rings) to silence (gains are stateless).
    pub fn reset(&mut self) {
        match self {
            RtGraph::Filter(s) => s.reset(),
            RtGraph::Gain(_) => {}
            RtGraph::Series { first, second, .. } => {
                first.reset();
                second.reset();
            }
            RtGraph::Parallel { left, right, .. } => {
                left.reset();
                right.reset();
            }
            RtGraph::Delay { ring, idx, .. } => {
                ring.iter_mut().for_each(|s| *s = 0.0);
                *idx = 0;
            }
            RtGraph::Echo {
                xring, wring, idx, ..
            } => {
                xring.iter_mut().for_each(|s| *s = 0.0);
                wring.iter_mut().for_each(|s| *s = 0.0);
                *idx = 0;
            }
            RtGraph::DelayFrac { ring, w, .. } => {
                ring.iter_mut().for_each(|s| *s = 0.0);
                *w = 0;
            }
            RtGraph::Comb { yring, idx, lp, .. } => {
                yring.iter_mut().for_each(|s| *s = 0.0);
                *idx = 0;
                *lp = 0.0;
            }
            RtGraph::Allpass {
                xring, yring, idx, ..
            } => {
                xring.iter_mut().for_each(|s| *s = 0.0);
                yring.iter_mut().for_each(|s| *s = 0.0);
                *idx = 0;
            }
            RtGraph::Fir { ring, idx, .. } => {
                ring.iter_mut().for_each(|s| *s = 0.0);
                *idx = 0;
            }
        }
    }

    /// Number of addressable filter (SOS) nodes, in depth-first order — the valid index range for
    /// [`set_coeffs`](Self::set_coeffs).
    pub fn filter_count(&self) -> usize {
        match self {
            RtGraph::Filter(_) => 1,
            RtGraph::Series { first, second, .. } => first.filter_count() + second.filter_count(),
            RtGraph::Parallel { left, right, .. } => left.filter_count() + right.filter_count(),
            _ => 0,
        }
    }

    /// Swap the `node`-th filter's coefficients (depth-first order) to `new_sos` with an equal-power
    /// crossfade over `fade_samples` — click-free live filter automation (plan task G9). The new
    /// coefficients are designed off the audio thread; the swap is allocation-free (after
    /// [`prepare`](Self::prepare)) when the section count is unchanged. Returns `false` if `node` is
    /// out of range. Apply between `process` blocks, e.g. driven from a command queue.
    pub fn set_coeffs(&mut self, node: usize, new_sos: &[Biquad], fade_samples: u32) -> bool {
        let mut remaining = node;
        self.set_coeffs_at(&mut remaining, new_sos, fade_samples)
    }

    fn set_coeffs_at(&mut self, remaining: &mut usize, new_sos: &[Biquad], fade: u32) -> bool {
        match self {
            RtGraph::Filter(s) => {
                if *remaining == 0 {
                    s.set_coeffs(new_sos, fade);
                    true
                } else {
                    *remaining -= 1;
                    false
                }
            }
            RtGraph::Series { first, second, .. } => {
                first.set_coeffs_at(remaining, new_sos, fade)
                    || second.set_coeffs_at(remaining, new_sos, fade)
            }
            RtGraph::Parallel { left, right, .. } => {
                left.set_coeffs_at(remaining, new_sos, fade)
                    || right.set_coeffs_at(remaining, new_sos, fade)
            }
            _ => false,
        }
    }

    /// Apply a [`SetCoeffs`] command drained from the ring on the audio thread — routes to the node
    /// and starts the crossfade. Returns `false` if the node index is out of range.
    pub fn apply(&mut self, cmd: &SetCoeffs) -> bool {
        let n = (cmd.count as usize).min(MAX_SETCOEFFS_SECTIONS);
        self.set_coeffs(cmd.node as usize, &cmd.sections[..n], cmd.fade_samples)
    }
}

/// The maximum SOS sections a [`SetCoeffs`] command carries (16th-order — 8 biquads).
pub const MAX_SETCOEFFS_SECTIONS: usize = 8;

/// A `Copy`, ring-friendly command to swap a filter node's coefficients live (plan task G9). Build it
/// off the audio thread with [`SetCoeffs::new`], send it over the lock-free [`ring`](crate::ring), and
/// apply it on the audio thread with [`RtGraph::apply`]. Fixed-capacity so it stays `Copy` and needs
/// no allocation on the audio thread; a cascade longer than [`MAX_SETCOEFFS_SECTIONS`] can't be sent
/// this way (design a shorter one, or swap it via [`RtGraph::set_coeffs`] between blocks).
#[derive(Clone, Copy, Debug)]
pub struct SetCoeffs {
    node: u32,
    fade_samples: u32,
    sections: [Biquad; MAX_SETCOEFFS_SECTIONS],
    count: u8,
}

impl SetCoeffs {
    /// Pack a coefficient swap for the `node`-th filter (crossfaded over `fade_samples`). Returns
    /// `None` if `sos` has more than [`MAX_SETCOEFFS_SECTIONS`] sections.
    pub fn new(node: u32, sos: &[Biquad], fade_samples: u32) -> Option<Self> {
        if sos.len() > MAX_SETCOEFFS_SECTIONS {
            return None;
        }
        let zero = Biquad {
            b0: 0.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        };
        let mut sections = [zero; MAX_SETCOEFFS_SECTIONS];
        sections[..sos.len()].copy_from_slice(sos);
        Some(Self {
            node,
            fade_samples,
            sections,
            count: sos.len() as u8,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::RtGraph;
    use fluxion_ops::{
        butterworth_highpass, butterworth_lowpass, delay, echo, fir_filter, reverb, sos_filter,
    };

    fn stream_chunks(g: &mut RtGraph, x: &[f32], block: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; block];
        let mut streamed = Vec::with_capacity(x.len());
        for chunk in x.chunks(block) {
            let out = &mut out[..chunk.len()];
            g.process(chunk, out);
            streamed.extend_from_slice(out);
        }
        streamed
    }

    fn signal(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (0.05 * i as f32).sin() + 0.3 * (0.27 * i as f32).sin())
            .collect()
    }

    #[test]
    fn series_equals_cascade() {
        let lp = butterworth_lowpass(4, 6_000.0, 48_000);
        let hp = butterworth_highpass(2, 200.0, 48_000);
        let x = signal(2000);

        let mut g = RtGraph::series(RtGraph::filter(lp.clone()), RtGraph::filter(hp.clone()));
        g.prepare(256);
        let mut out = vec![0.0f32; 256];
        let mut streamed = Vec::new();
        for chunk in x.chunks(200) {
            let out = &mut out[..chunk.len()];
            g.process(chunk, out);
            streamed.extend_from_slice(out);
        }

        // Reference: hp(lp(x)) — concatenated cascade.
        let mut cascade = lp;
        cascade.extend(hp);
        let want = sos_filter(&x, &cascade);
        for (a, b) in streamed.iter().zip(&want) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn parallel_then_gain_matches_reference() {
        // (lp + hp) | gain(0.5)
        let lp = butterworth_lowpass(4, 1_000.0, 48_000);
        let hp = butterworth_highpass(4, 5_000.0, 48_000);
        let x = signal(3000);

        let mut g = RtGraph::series(
            RtGraph::parallel(RtGraph::filter(lp.clone()), RtGraph::filter(hp.clone())),
            RtGraph::gain(0.5),
        );
        g.prepare(128);
        let mut out = vec![0.0f32; 128];
        let mut streamed = Vec::new();
        for chunk in x.chunks(128) {
            let out = &mut out[..chunk.len()];
            g.process(chunk, out);
            streamed.extend_from_slice(out);
        }

        let y_lp = sos_filter(&x, &lp);
        let y_hp = sos_filter(&x, &hp);
        for (i, got) in streamed.iter().enumerate() {
            let want = 0.5 * (y_lp[i] + y_hp[i]);
            assert!((got - want).abs() < 1e-4, "at {i}: {got} vs {want}");
        }
    }

    #[test]
    fn delay_node_matches_batch() {
        let x = signal(1000);
        let (d, mix) = (37usize, 0.6f32);
        let mut g = RtGraph::delay(d, mix); // block (64) > d, ring wraps within a block
        let streamed = stream_chunks(&mut g, &x, 64);
        let want = delay(&x, d, mix);
        for (a, b) in streamed.iter().zip(&want) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn echo_node_matches_batch() {
        let x = signal(1000);
        let (d, fb, wet) = (53usize, 0.5f32, 0.8f32);
        let mut g = RtGraph::echo(d, fb, wet); // block (50) < d
        let streamed = stream_chunks(&mut g, &x, 50);
        let want = echo(&x, d, fb, wet);
        for (a, b) in streamed.iter().zip(&want) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    // Reference fractional delay: (1-mix)·x[n] + mix·x[n-D], x[n-D] linearly interpolated.
    fn ref_delay_frac(x: &[f32], delay: f32, mix: f32) -> Vec<f32> {
        let i = delay.floor() as usize;
        let frac = delay - i as f32;
        (0..x.len())
            .map(|n| {
                let a = if n >= i { x[n - i] } else { 0.0 };
                let b = if n > i { x[n - i - 1] } else { 0.0 };
                (1.0 - mix) * x[n] + mix * ((1.0 - frac) * a + frac * b)
            })
            .collect()
    }

    #[test]
    fn fractional_delay_matches_reference() {
        let x = signal(1000);
        for &delay in &[0.0, 0.5, 3.7, 12.25, 40.9] {
            let mut g = RtGraph::delay_frac(delay, 0.7);
            let streamed = stream_chunks(&mut g, &x, 64);
            let want = ref_delay_frac(&x, delay, 0.7);
            for (k, (a, b)) in streamed.iter().zip(&want).enumerate() {
                assert!((a - b).abs() < 1e-5, "delay {delay} at {k}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn integer_fractional_delay_equals_delay_node() {
        // frac=0 ⇒ a pure integer delay; must agree with the integer Delay node.
        let x = signal(500);
        let mut frac = RtGraph::delay_frac(8.0, 0.5);
        let mut int = RtGraph::delay(8, 0.5);
        let a = stream_chunks(&mut frac, &x, 50);
        let b = stream_chunks(&mut int, &x, 50);
        for (p, q) in a.iter().zip(&b) {
            assert!((p - q).abs() < 1e-6, "{p} vs {q}");
        }
    }

    #[test]
    fn set_coeffs_targets_the_addressed_filter() {
        // series(lp, lp); swap node 0 to a high-pass with a crossfade → the graph behaves like
        // series(hp, lp), proving the index addresses the right node.
        let lp = butterworth_lowpass(2, 4_000.0, 48_000);
        let hp = butterworth_highpass(2, 4_000.0, 48_000);
        let x = signal(1500);

        let mut g = RtGraph::series(RtGraph::filter(lp.clone()), RtGraph::filter(lp.clone()));
        assert_eq!(g.filter_count(), 2);
        g.prepare(64);
        assert!(
            !g.set_coeffs(9, &hp, 4),
            "out-of-range node must return false"
        );
        assert!(g.set_coeffs(0, &hp, 1), "node 0 swap must apply");
        let got = stream_chunks(&mut g, &x, 64);

        let mut want_g = RtGraph::series(RtGraph::filter(hp), RtGraph::filter(lp));
        want_g.prepare(64);
        let want = stream_chunks(&mut want_g, &x, 64);
        for (a, b) in got.iter().zip(&want).skip(4) {
            // skip the brief crossfade transient
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }

    #[test]
    fn setcoeffs_command_rides_the_ring() {
        // The full G9 path: design coeffs off-thread → pack → lock-free ring → drain + apply on the
        // audio thread → crossfade. Here single-threaded, but the SetCoeffs is Copy so it's SPSC-safe.
        use super::{MAX_SETCOEFFS_SECTIONS, SetCoeffs};
        use crate::ring::channel;

        let lp = butterworth_lowpass(2, 4_000.0, 48_000);
        let hp = butterworth_highpass(2, 4_000.0, 48_000);
        let (mut tx, mut rx) = channel::<SetCoeffs>(4);
        tx.push(SetCoeffs::new(0, &hp, 1).unwrap()).unwrap();

        let x = signal(800);
        let mut g = RtGraph::filter(lp);
        g.prepare(64);
        while let Some(cmd) = rx.pop() {
            assert!(g.apply(&cmd)); // audio thread drains + applies
        }
        let got = stream_chunks(&mut g, &x, 64);

        let mut want_g = RtGraph::filter(hp.clone());
        want_g.prepare(64);
        let want = stream_chunks(&mut want_g, &x, 64);
        for (a, b) in got.iter().zip(&want).skip(4) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
        // A cascade longer than the fixed capacity can't ride the ring.
        let too_long = vec![hp[0]; MAX_SETCOEFFS_SECTIONS + 1];
        assert!(SetCoeffs::new(0, &too_long, 1).is_none());
    }

    #[test]
    fn fir_node_matches_batch() {
        // Streaming the direct-form FIR node must equal the offline fir_filter, across block sizes
        // (block smaller and larger than the tap count).
        let x = signal(1000);
        let taps = vec![0.2f32, -0.5, 0.3, 0.1, -0.05, 0.02];
        let want = fir_filter(&x, &taps);
        for block in [4, 64] {
            let mut g = RtGraph::fir(taps.clone());
            let streamed = stream_chunks(&mut g, &x, block);
            for (a, b) in streamed.iter().zip(&want) {
                assert!((a - b).abs() < 1e-5, "block {block}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn reverb_node_matches_batch() {
        // Streaming the RtGraph reverb must equal the offline Freeverb sample-for-sample.
        let x = signal(6000);
        let (room, damp, mix) = (0.8f32, 0.3f32, 0.5f32);
        let mut g = RtGraph::reverb(room, damp, mix);
        g.prepare(256);
        let streamed = stream_chunks(&mut g, &x, 256);
        let want = reverb(&x, room, damp, mix);
        assert_eq!(streamed.len(), want.len());
        for (i, (a, b)) in streamed.iter().zip(&want).enumerate() {
            assert!((a - b).abs() < 1e-4, "reverb at {i}: {a} vs {b}");
        }
    }

    #[test]
    fn nested_parallel_inside_series() {
        // lp1 | (lp2 + hp) — exercises a parallel branch fed by an upstream filter.
        let lp1 = butterworth_lowpass(2, 8_000.0, 48_000);
        let lp2 = butterworth_lowpass(2, 1_000.0, 48_000);
        let hp = butterworth_highpass(2, 3_000.0, 48_000);
        let x = signal(1500);

        let mut g = RtGraph::series(
            RtGraph::filter(lp1.clone()),
            RtGraph::parallel(RtGraph::filter(lp2.clone()), RtGraph::filter(hp.clone())),
        );
        g.prepare(64);
        let mut out = vec![0.0f32; 64];
        let mut streamed = Vec::new();
        for chunk in x.chunks(64) {
            let out = &mut out[..chunk.len()];
            g.process(chunk, out);
            streamed.extend_from_slice(out);
        }

        let mid = sos_filter(&x, &lp1); // lp1(x)
        let want_l = sos_filter(&mid, &lp2);
        let want_r = sos_filter(&mid, &hp);
        for (i, got) in streamed.iter().enumerate() {
            let want = want_l[i] + want_r[i];
            assert!((got - want).abs() < 1e-4, "at {i}: {got} vs {want}");
        }
    }
}

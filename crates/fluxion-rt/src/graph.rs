//! General-graph realtime executor (plan task G3, beyond a linear cascade).
//!
//! [`SosStream`] runs one cascade; [`RtGraph`] runs the full graph algebra — `Series` (chain) and
//! `Parallel` (sum) composition of filter and gain nodes — block-by-block and allocation-free. The
//! intermediate buffers that `Series`/`Parallel` need are sized once by [`RtGraph::prepare`]; after
//! that [`RtGraph::process`] never allocates, so it is audio-thread safe.
//!
//! Mirrors `fluxion_core::Graph`'s `|` (series) and `+` (parallel). Lowering an arbitrary
//! `fluxion_core::Graph` (designing each op's coefficients) to an `RtGraph` belongs in
//! `fluxion-backend` (next step, like `freeze`); here are the runtime building blocks.

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

    /// Pre-size every internal scratch buffer for blocks up to `max_block` samples. This is the only
    /// allocating step — call it before going realtime. Blocks passed to [`process`](Self::process)
    /// must not exceed `max_block`.
    pub fn prepare(&mut self, max_block: usize) {
        match self {
            RtGraph::Filter(_) | RtGraph::Gain(_) => {}
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
        }
    }

    /// Reset all filter state in the graph (gains are stateless).
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RtGraph;
    use fluxion_ops::{butterworth_highpass, butterworth_lowpass, sos_filter};

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

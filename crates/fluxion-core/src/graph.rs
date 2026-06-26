//! The DSP graph IR and its composition operators.

use std::ops::{Add, BitOr};

/// A reified node in the DSP graph — the IR the backends lower.
///
/// Build leaves with [`Graph::op`] and compose them with `|` (series) and `+`
/// (parallel, summed). Topologies nest arbitrarily:
///
/// ```
/// use fluxion_core::Graph;
/// let chain = Graph::op("lowpass", [800.0]) | Graph::op("gain", [-3.0]);
/// let split = Graph::op("loshelf", [200.0, 6.0]) + Graph::op("hishelf", [4000.0, -3.0]);
/// let nested = (split | chain).clone();
/// assert_eq!(nested.leaf_count(), 4);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub enum Graph {
    /// Pass-through; the identity element of series composition.
    Id,
    /// A named leaf operation with scalar `f32` parameters.
    ///
    /// `params` is a deliberate placeholder until typed, differentiable nodes land: it keeps
    /// the IR inspectable and serializable without pulling in a tensor backend.
    Op {
        /// Op identifier, e.g. `"lowpass"`, `"gain"`.
        name: &'static str,
        /// Design-time scalar parameters (Hz, dB, …) in op-defined order.
        params: Vec<f32>,
    },
    /// Series: feed the output of `.0` into `.1`.
    Series(Box<Graph>, Box<Graph>),
    /// Parallel: run `.0` and `.1` on the same input and sum their outputs.
    Parallel(Box<Graph>, Box<Graph>),
}

impl Graph {
    /// Construct a named leaf op, e.g. `Graph::op("gain", [-3.0])`.
    pub fn op(name: &'static str, params: impl Into<Vec<f32>>) -> Self {
        Graph::Op { name, params: params.into() }
    }

    /// Number of leaf ops in the graph (structural size; [`Graph::Id`] counts as zero).
    pub fn leaf_count(&self) -> usize {
        match self {
            Graph::Id => 0,
            Graph::Op { .. } => 1,
            Graph::Series(a, b) | Graph::Parallel(a, b) => a.leaf_count() + b.leaf_count(),
        }
    }

    /// `true` if the graph is the pure pass-through.
    pub fn is_empty(&self) -> bool {
        matches!(self, Graph::Id)
    }

    /// Naive scalar CPU **reference** evaluation — for testing the algebra only.
    ///
    /// ponytail: reference interpreter, not the hot path. Real execution is lowered to fused
    /// SIMD/GPU kernels in the backend crates; only `id` and `gain` are wired here so the
    /// operators can be unit-tested without a tensor runtime. Unknown ops panic on purpose —
    /// they are a test-only programming error, not a user-facing path.
    pub fn eval_ref(&self, input: &[f32]) -> Vec<f32> {
        match self {
            Graph::Id => input.to_vec(),
            Graph::Op { name, params } => match *name {
                "gain" => {
                    let g = params.first().copied().unwrap_or(1.0);
                    input.iter().map(|x| x * g).collect()
                }
                other => panic!("eval_ref: op '{other}' has no reference implementation"),
            },
            Graph::Series(a, b) => b.eval_ref(&a.eval_ref(input)),
            Graph::Parallel(a, b) => {
                let ya = a.eval_ref(input);
                let yb = b.eval_ref(input);
                ya.iter().zip(yb).map(|(x, y)| x + y).collect()
            }
        }
    }
}

/// `a | b` — series composition.
impl BitOr for Graph {
    type Output = Graph;
    fn bitor(self, rhs: Graph) -> Graph {
        Graph::Series(Box::new(self), Box::new(rhs))
    }
}

/// `a + b` — parallel composition (outputs summed).
impl Add for Graph {
    type Output = Graph;
    fn add(self, rhs: Graph) -> Graph {
        Graph::Parallel(Box::new(self), Box::new(rhs))
    }
}

#[cfg(test)]
mod tests {
    use super::Graph;

    #[test]
    fn series_chains_in_order() {
        // gain(2) | gain(3) == gain(6)
        let g = Graph::op("gain", [2.0]) | Graph::op("gain", [3.0]);
        assert_eq!(g.eval_ref(&[1.0, -1.0]), vec![6.0, -6.0]);
        assert_eq!(g.leaf_count(), 2);
    }

    #[test]
    fn parallel_sums_branches() {
        // gain(2) + gain(3) == gain(5)
        let g = Graph::op("gain", [2.0]) + Graph::op("gain", [3.0]);
        assert_eq!(g.eval_ref(&[1.0, 4.0]), vec![5.0, 20.0]);
    }

    #[test]
    fn id_is_passthrough() {
        assert_eq!(Graph::Id.eval_ref(&[1.0, 2.0]), vec![1.0, 2.0]);
        assert!(Graph::Id.is_empty());
        assert_eq!(Graph::Id.leaf_count(), 0);
    }

    #[test]
    fn nested_topology() {
        // (gain(2) + gain(3)) | gain(10) == gain(50)
        let g = (Graph::op("gain", [2.0]) + Graph::op("gain", [3.0])) | Graph::op("gain", [10.0]);
        assert_eq!(g.eval_ref(&[1.0]), vec![50.0]);
        assert_eq!(g.leaf_count(), 3);
    }
}

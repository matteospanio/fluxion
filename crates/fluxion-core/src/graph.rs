//! The DSP graph IR and its composition operators.

use std::ops::{Add, BitOr};

use crate::op::{Op, OpKind};

/// A reified node in the DSP graph — the IR the backends lower.
///
/// Build leaves with [`Graph::op`] and compose them with `|` (series) and `+`
/// (parallel, summed). Topologies nest arbitrarily:
///
/// ```
/// use fluxion_core::{Graph, OpKind};
/// let chain = Graph::op(OpKind::Lowpass, [800.0]) | Graph::op(OpKind::Gain, [0.5]);
/// let split = Graph::op(OpKind::Lowpass, [200.0]) + Graph::op(OpKind::Highpass, [4000.0]);
/// let nested = split | chain;
/// assert_eq!(nested.leaf_count(), 4);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub enum Graph {
    /// Pass-through; the identity element of series composition.
    Id,
    /// A leaf operation.
    Op(Op),
    /// Series: feed the output of `.0` into `.1`.
    Series(Box<Graph>, Box<Graph>),
    /// Parallel: run `.0` and `.1` on the same input and sum their outputs.
    Parallel(Box<Graph>, Box<Graph>),
}

impl Graph {
    /// Build a validated leaf-op graph.
    ///
    /// Panics if the parameters are invalid for the kind (wrong arity or out of static bounds) —
    /// that is a programming error at a Rust call site. For fallible construction from user input
    /// (e.g. the CLI parser), use [`Op::new`] and convert with `Graph::from`.
    pub fn op(kind: OpKind, params: impl Into<Vec<f32>>) -> Graph {
        Graph::Op(Op::new(kind, params).expect("valid op parameters"))
    }

    /// Number of leaf ops in the graph (structural size; [`Graph::Id`] counts as zero).
    pub fn leaf_count(&self) -> usize {
        match self {
            Graph::Id => 0,
            Graph::Op(_) => 1,
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
    /// SIMD/GPU kernels in the backend crates; only `Id` and `Gain` are wired here so the
    /// operators can be unit-tested without a tensor runtime. Unknown ops panic on purpose —
    /// they are a test-only programming error, not a user-facing path.
    pub fn eval_ref(&self, input: &[f32]) -> Vec<f32> {
        match self {
            Graph::Id => input.to_vec(),
            Graph::Op(op) => match op.kind {
                OpKind::Gain => {
                    let g = op.params[0]; // arity guaranteed by `Op::new`
                    input.iter().map(|x| x * g).collect()
                }
                other => panic!(
                    "eval_ref: op '{}' has no reference implementation",
                    other.name()
                ),
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

impl From<Op> for Graph {
    fn from(op: Op) -> Graph {
        Graph::Op(op)
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
    use crate::op::OpKind;

    fn gain(g: f32) -> Graph {
        Graph::op(OpKind::Gain, [g])
    }

    #[test]
    fn series_chains_in_order() {
        // gain(2) | gain(3) == gain(6)
        let g = gain(2.0) | gain(3.0);
        assert_eq!(g.eval_ref(&[1.0, -1.0]), vec![6.0, -6.0]);
        assert_eq!(g.leaf_count(), 2);
    }

    #[test]
    fn parallel_sums_branches() {
        // gain(2) + gain(3) == gain(5)
        let g = gain(2.0) + gain(3.0);
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
        let g = (gain(2.0) + gain(3.0)) | gain(10.0);
        assert_eq!(g.eval_ref(&[1.0]), vec![50.0]);
        assert_eq!(g.leaf_count(), 3);
    }
}

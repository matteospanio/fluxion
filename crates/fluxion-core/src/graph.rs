//! The DSP graph IR and its composition operators.

use std::fmt;
use std::ops::{Add, BitOr};

use serde::{Deserialize, Serialize};

use crate::op::{Op, OpKind};

/// A reified node in the DSP graph — the IR the backends lower.
///
/// Build leaves with [`Graph::op`] and compose them with `|` (series) and `+`
/// (parallel, summed). Topologies nest arbitrarily:
///
/// ```
/// use fluxion_core::{Graph, OpKind};
/// let chain = Graph::op(OpKind::Lowpass, [800.0, 2.0]) | Graph::op(OpKind::Gain, [0.5]);
/// let split = Graph::op(OpKind::Lowpass, [200.0, 2.0]) + Graph::op(OpKind::Highpass, [4000.0, 2.0]);
/// let nested = split | chain;
/// assert_eq!(nested.leaf_count(), 4);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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

/// DSL-style rendering: `lowpass(800, 4) | (loshelf(...) + peaking(...))`.
impl fmt::Display for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Graph::Id => f.write_str("id"),
            Graph::Op(op) => {
                f.write_str(op.kind.name())?;
                if !op.params.is_empty() {
                    let ps: Vec<String> = op.params.iter().map(|p| fmt_num(*p)).collect();
                    write!(f, "({})", ps.join(", "))?;
                }
                Ok(())
            }
            Graph::Series(a, b) => {
                write_child(f, a, true)?;
                f.write_str(" | ")?;
                write_child(f, b, true)
            }
            Graph::Parallel(a, b) => {
                write_child(f, a, false)?;
                f.write_str(" + ")?;
                write_child(f, b, false)
            }
        }
    }
}

/// Parenthesize a child only when the operator precedence would otherwise be ambiguous
/// (a parallel inside a series, or a series inside a parallel).
fn write_child(f: &mut fmt::Formatter<'_>, g: &Graph, in_series: bool) -> fmt::Result {
    let needs_parens = matches!(
        (g, in_series),
        (Graph::Parallel(..), true) | (Graph::Series(..), false)
    );
    if needs_parens {
        write!(f, "({g})")
    } else {
        write!(f, "{g}")
    }
}

/// Render a parameter, dropping a trailing `.0` for whole numbers (`800.0` → `800`).
fn fmt_num(x: f32) -> String {
    if x.is_finite() && x.fract() == 0.0 && x.abs() < 1e7 {
        format!("{}", x as i64)
    } else {
        format!("{x}")
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

    // --- B6: Display ---

    #[test]
    fn display_renders_dsl() {
        let g = (Graph::op(OpKind::LowShelf, [200.0, 6.0, 0.7])
            + Graph::op(OpKind::Peaking, [1000.0, 6.0, 1.5]))
            | Graph::op(OpKind::Gain, [0.5]);
        assert_eq!(
            g.to_string(),
            "(lowshelf(200, 6, 0.7) + peaking(1000, 6, 1.5)) | gain(0.5)"
        );
        assert_eq!(Graph::Id.to_string(), "id");
    }

    // --- B7: algebra laws (exact in f32 for the small integers used) ---

    #[test]
    fn series_is_associative() {
        let inp = [0.5, -1.0, 2.0];
        for (a, b, c) in [(2.0, 3.0, 5.0), (1.0, -2.0, 4.0)] {
            let left = ((gain(a) | gain(b)) | gain(c)).eval_ref(&inp);
            let right = (gain(a) | (gain(b) | gain(c))).eval_ref(&inp);
            assert_eq!(left, right);
        }
    }

    #[test]
    fn parallel_is_commutative_and_associative() {
        let inp = [1.0, 2.0, 3.0];
        assert_eq!(
            (gain(2.0) + gain(3.0)).eval_ref(&inp),
            (gain(3.0) + gain(2.0)).eval_ref(&inp),
        );
        assert_eq!(
            ((gain(1.0) + gain(2.0)) + gain(3.0)).eval_ref(&inp),
            (gain(1.0) + (gain(2.0) + gain(3.0))).eval_ref(&inp),
        );
    }

    #[test]
    fn id_is_series_identity() {
        let inp = [1.0, -2.0, 0.5];
        assert_eq!(
            (Graph::Id | gain(2.0)).eval_ref(&inp),
            gain(2.0).eval_ref(&inp)
        );
        assert_eq!(
            (gain(2.0) | Graph::Id).eval_ref(&inp),
            gain(2.0).eval_ref(&inp)
        );
    }
}

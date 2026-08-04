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
    /// A **named** node — transparent to execution (it runs exactly as `node`), but addressable by
    /// `name` for per-node parameter automation and named-module (checkpoint) import.
    /// See [`Graph::find_named`].
    Named {
        /// The node's stable identifier.
        name: String,
        /// The wrapped subgraph.
        node: Box<Graph>,
    },
    /// A **side input** (ROADMAP S1): read side signal `n` instead of what came down the chain.
    ///
    /// The one thing a series/parallel tree of one-input nodes cannot say: *this* audio, not the
    /// audio flowing here. A kick drum ducking a bass, a different microphone opening a gate. The
    /// side signals are handed to `process_with` when the chain runs; a side that was not supplied
    /// reads as silence, so a chain built for two inputs still runs on one rather than failing.
    Side(usize),
    /// A subgraph and the **key** feeding it (the `<` operator, ROADMAP S1).
    ///
    /// `key` is evaluated on whatever `node` was given, and its output is offered to every op
    /// inside `node` that declares a key input ([`OpKind::takes_key`]). Ops that do not are
    /// untouched, which is what keeps every one-input chain behaving exactly as before.
    Keyed {
        /// The subgraph being keyed.
        node: Box<Graph>,
        /// The signal that drives it — typically a [`Graph::Side`], possibly filtered.
        key: Box<Graph>,
    },
    /// A **feedback** loop (the `~` operator) — the third construct a series/parallel *tree* cannot
    /// encode, because it needs a cycle (plan task B8). Semantics: `y[n] = forward(x[n] +
    /// feedback(y)[n-1])`; the one-sample delay on the loop-back path breaks the algebraic loop so it
    /// is computable. `feedback = Id` makes a plain unit-delay feedback around `forward`.
    Feedback {
        /// The forward path (processes `input + delayed loop-back`).
        forward: Box<Graph>,
        /// The loop-back path applied to the (one-sample-delayed) output.
        feedback: Box<Graph>,
    },
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

    /// Attach a name to a subgraph (node identity) — transparent to execution, addressable by name.
    pub fn named(name: impl Into<String>, node: Graph) -> Graph {
        Graph::Named {
            name: name.into(),
            node: Box::new(node),
        }
    }

    /// Key `self` with `key` (the `<` operator): every op inside `self` that declares a key input
    /// is driven by `key`'s output instead of by the audio passing through it.
    pub fn keyed(self, key: Graph) -> Graph {
        Graph::Keyed {
            node: Box::new(self),
            key: Box::new(key),
        }
    }

    /// Read side signal `index` instead of the chain's own input — see [`Graph::Side`].
    pub fn side(index: usize) -> Graph {
        Graph::Side(index)
    }

    /// The highest side-input index the graph reaches for, plus one: how many side signals it
    /// wants. Zero for an ordinary one-input chain.
    pub fn side_inputs(&self) -> usize {
        match self {
            Graph::Side(n) => n + 1,
            Graph::Id | Graph::Op(_) => 0,
            Graph::Series(a, b) | Graph::Parallel(a, b) => a.side_inputs().max(b.side_inputs()),
            Graph::Named { node, .. } => node.side_inputs(),
            Graph::Keyed { node, key } => node.side_inputs().max(key.side_inputs()),
            Graph::Feedback { forward, feedback } => {
                forward.side_inputs().max(feedback.side_inputs())
            }
        }
    }

    /// Wrap `self` in a feedback loop with `feedback` on the loop-back path (the `~` operator):
    /// `y[n] = self(x[n] + feedback(y)[n-1])`. Pass [`Graph::Id`] for a plain unit-delay feedback.
    pub fn feedback(self, feedback: Graph) -> Graph {
        Graph::Feedback {
            forward: Box::new(self),
            feedback: Box::new(feedback),
        }
    }

    /// Number of leaf ops in the graph (structural size; [`Graph::Id`] counts as zero).
    pub fn leaf_count(&self) -> usize {
        match self {
            Graph::Id | Graph::Side(_) => 0,
            Graph::Op(_) => 1,
            Graph::Series(a, b) | Graph::Parallel(a, b) => a.leaf_count() + b.leaf_count(),
            Graph::Named { node, .. } => node.leaf_count(),
            // The key is part of the chain and its ops are real ops, so they count.
            Graph::Keyed { node, key } => node.leaf_count() + key.leaf_count(),
            Graph::Feedback { forward, feedback } => forward.leaf_count() + feedback.leaf_count(),
        }
    }

    /// `true` if the graph is the pure pass-through.
    pub fn is_empty(&self) -> bool {
        matches!(self, Graph::Id)
    }

    /// Find the subgraph wrapped by a [`Graph::Named`] node with this `name` (depth-first, first
    /// match), or `None` — the read side of node identity (address a node to inspect/re-design it).
    pub fn find_named(&self, name: &str) -> Option<&Graph> {
        match self {
            Graph::Named { name: n, node } if n == name => Some(node),
            Graph::Named { node, .. } => node.find_named(name),
            Graph::Series(a, b) | Graph::Parallel(a, b) => {
                a.find_named(name).or_else(|| b.find_named(name))
            }
            Graph::Feedback { forward, feedback } => forward
                .find_named(name)
                .or_else(|| feedback.find_named(name)),
            Graph::Keyed { node, key } => node.find_named(name).or_else(|| key.find_named(name)),
            Graph::Id | Graph::Op(_) | Graph::Side(_) => None,
        }
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
            Graph::Named { node, .. } => node.eval_ref(input),
            // The reference interpreter has one input, so a side signal is the silence a chain
            // gets when nothing was connected — the same answer the real engine gives.
            Graph::Side(_) => vec![0.0; input.len()],
            // No op wired into `eval_ref` reads a key, so keying is transparent here. That is not a
            // simplification: it is the property S1 has to have, checked in the backend where the
            // keyed ops actually live.
            Graph::Keyed { node, .. } => node.eval_ref(input),
            // y[n] = forward(x[n] + feedback(y)[n-1]); reference is sample-by-sample, so the
            // sub-paths must be **stateless** here (Id / Gain) — the same scope as the rest of
            // `eval_ref`. The stateful engines run feedback in the backend (`fluxion-backend`).
            Graph::Feedback { forward, feedback } => {
                let mut y = vec![0.0f32; input.len()];
                let mut fb_prev = 0.0f32;
                for (n, &x) in input.iter().enumerate() {
                    let yn = forward.eval_ref(&[x + fb_prev])[0];
                    y[n] = yn;
                    fb_prev = feedback.eval_ref(&[yn])[0];
                }
                y
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
                write_child(f, a, true, false)?;
                f.write_str(" | ")?;
                write_child(f, b, true, true)
            }
            Graph::Parallel(a, b) => {
                write_child(f, a, false, false)?;
                f.write_str(" + ")?;
                write_child(f, b, false, true)
            }
            // `name:` binds tighter than every operator, so a label on a primary needs no parens;
            // anything composite must say where it ends. `Feedback` already brackets itself.
            Graph::Named { name, node } => match **node {
                Graph::Id | Graph::Op(_) | Graph::Named { .. } | Graph::Feedback { .. } => {
                    write!(f, "{name}: {node}")
                }
                _ => write!(f, "{name}: ({node})"),
            },
            Graph::Feedback { forward, feedback } => write!(f, "({forward} ~ {feedback})"),
            Graph::Side(n) => write!(f, "side({n})"),
            // `<` is the loosest operator and non-associative, so a composite on either side has to
            // say where it ends — the same reason `Feedback` brackets itself.
            Graph::Keyed { node, key } => {
                write_keyed_child(f, node)?;
                f.write_str(" < ")?;
                write_keyed_child(f, key)
            }
        }
    }
}

/// Bracket anything that is not a primary, since `<` binds looser than `|`, `+` and `~`.
fn write_keyed_child(f: &mut fmt::Formatter<'_>, g: &Graph) -> fmt::Result {
    match g {
        Graph::Id
        | Graph::Op(_)
        | Graph::Side(_)
        | Graph::Named { .. }
        | Graph::Feedback { .. } => {
            write!(f, "{g}")
        }
        _ => write!(f, "({g})"),
    }
}

/// Parenthesize a child only when the rendering would otherwise be ambiguous: a different operator
/// (a parallel inside a series, or a series inside a parallel), or the **same** operator on the
/// right — since both parse left-associative, `a | (b | c)` needs its parens to survive a reparse
/// while `(a | b) | c` does not.
fn write_child(
    f: &mut fmt::Formatter<'_>,
    g: &Graph,
    parent_series: bool,
    right: bool,
) -> fmt::Result {
    let needs_parens = match g {
        Graph::Series(..) => !parent_series || right,
        Graph::Parallel(..) => parent_series || right,
        // `<` is looser than both, so a keyed subchain inside either needs brackets whichever side
        // it is on.
        Graph::Keyed { .. } => true,
        // Leaves, labels and feedback loops are already unambiguous on their own.
        _ => false,
    };
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

    // --- B8: node identity + feedback ---

    #[test]
    fn named_is_transparent_and_addressable() {
        let g = Graph::named("lp", gain(2.0)) | gain(3.0);
        assert_eq!(g.eval_ref(&[1.0]), vec![6.0]); // transparent to execution
        assert_eq!(g.leaf_count(), 2);
        assert_eq!(g.find_named("lp").unwrap().eval_ref(&[1.0]), vec![2.0]); // addressable
        assert!(g.find_named("nope").is_none());
    }

    #[test]
    fn feedback_is_a_first_order_iir() {
        // gain(1) ~ gain(0.5): y[n] = x[n] + 0.5·y[n-1] → impulse response 0.5ⁿ.
        let g = gain(1.0).feedback(gain(0.5));
        let y = g.eval_ref(&[1.0, 0.0, 0.0, 0.0, 0.0]);
        for (n, &v) in y.iter().enumerate() {
            assert!((v - 0.5f32.powi(n as i32)).abs() < 1e-6, "y[{n}] = {v}");
        }
    }

    #[test]
    fn display_and_serde_round_trip_named_feedback() {
        let g = Graph::named("core", gain(1.0)).feedback(gain(0.5));
        assert_eq!(g.to_string(), "(core: gain(1) ~ gain(0.5))");
        let json = serde_json::to_string(&g).unwrap();
        assert_eq!(serde_json::from_str::<Graph>(&json).unwrap(), g);
    }

    // --- B6: Display ---

    /// Pre-condition: nesting shapes the writer used to flatten or leave unlabelled.
    /// Post-condition: exactly one tree can produce each string — the precondition for a parser.
    #[test]
    fn display_is_unambiguous() {
        let (a, b, c) = (gain(1.0), gain(2.0), gain(3.0));

        // Both operators parse left-associative, so only a same-kind *right* operand needs parens.
        let ab_c = (a.clone() | b.clone()) | c.clone();
        let a_bc = a.clone() | (b.clone() | c.clone());
        assert_eq!(ab_c.to_string(), "gain(1) | gain(2) | gain(3)");
        assert_eq!(a_bc.to_string(), "gain(1) | (gain(2) | gain(3))");
        assert_ne!(ab_c.to_string(), a_bc.to_string());

        let ab_c = (a.clone() + b.clone()) + c.clone();
        let a_bc = a.clone() + (b.clone() + c.clone());
        assert_eq!(ab_c.to_string(), "gain(1) + gain(2) + gain(3)");
        assert_eq!(a_bc.to_string(), "gain(1) + (gain(2) + gain(3))");
        assert_ne!(ab_c.to_string(), a_bc.to_string());

        // Mixed nesting keeps its parens, on either side.
        assert_eq!(
            ((a.clone() + b.clone()) | c.clone()).to_string(),
            "(gain(1) + gain(2)) | gain(3)"
        );
        assert_eq!(
            ((a.clone() | b.clone()) + c.clone()).to_string(),
            "(gain(1) | gain(2)) + gain(3)"
        );

        // `name:` binds tighter than any operator, so a label on a leaf needs nothing...
        assert_eq!(
            (Graph::named("lp", a.clone()) | b.clone()).to_string(),
            "lp: gain(1) | gain(2)"
        );
        // ...and a label on a composite must say where it ends.
        assert_eq!(
            Graph::named("mix", a.clone() | b.clone()).to_string(),
            "mix: (gain(1) | gain(2))"
        );
        // Feedback already parenthesizes itself; a label must not double up.
        assert_eq!(
            Graph::named("fb", a.clone().feedback(b.clone())).to_string(),
            "fb: (gain(1) ~ gain(2))"
        );
    }

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

//! `fluxion-core` — the functional DSP graph algebra and intermediate representation.
//!
//! This crate defines the *reified* graph the whole library is built around: nodes are typed
//! [`Op`]s (filters/effects), edges are [`Signal`]s, and two operators compose them —
//! `|` for **series** (run `a`, then feed its output to `b`) and `+` for **parallel**
//! (run both on the same input and sum the outputs).
//!
//! The graph is plain, inspectable data — it carries no tensor runtime. Real execution is the job
//! of the backend crates (`fluxion-backend`, `fluxion-rt`); a tiny scalar CPU reference
//! interpreter ([`Graph::eval_ref`]) lives here only so the algebra is testable in isolation.
//!
//! ```
//! use fluxion_core::{Graph, OpKind};
//! let chain = Graph::op(OpKind::Lowpass, [800.0, 2.0]) | Graph::op(OpKind::Gain, [0.5]);
//! assert_eq!(chain.leaf_count(), 2);
//! ```

pub mod graph;
pub mod op;
pub mod param;
pub mod signal;

pub use graph::Graph;
pub use op::{Op, OpError, OpKind};
pub use param::{ParamSpec, Unit};
pub use signal::Signal;

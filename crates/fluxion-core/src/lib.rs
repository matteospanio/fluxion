//! `fluxion-core` — the functional DSP graph algebra and intermediate representation.
//!
//! This crate defines the *reified* graph the whole library is built around: nodes are
//! filters/effects, edges are signals, and two operators compose them —
//! `|` for **series** (run `a`, then feed its output to `b`) and `+` for **parallel**
//! (run both on the same input and sum the outputs).
//!
//! The graph is plain, inspectable data — it carries no tensor runtime. Real execution is
//! the job of the backend crates (`fluxion-backend`, `fluxion-rt`); a tiny scalar CPU
//! reference interpreter ([`Graph::eval_ref`]) lives here only so the algebra is testable
//! in isolation.
//!
//! ```
//! use fluxion_core::Graph;
//! let chain = Graph::op("lowpass", [800.0]) | Graph::op("gain", [-3.0]);
//! assert_eq!(chain.leaf_count(), 2);
//! ```

pub mod graph;

pub use graph::Graph;

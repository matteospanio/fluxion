//! Rendering a graph whose parameters move (ROADMAP D2).
//!
//! A lane names a node, a parameter and a [`Curve`](fluxion_core::automation::Curve); this
//! applies it. The rest of the graph is
//! untouched — an op nobody automated is processed exactly as [`process`](crate::process) would,
//! over the whole buffer at once, because a time-invariant op has no reason to be chopped up.
//!
//! # Two ways to vary a parameter, because the ops differ
//!
//! **Per sample.** A gain is a multiply: varying it costs nothing and is exact at every sample.
//! Ops in this class are rendered against the curve itself, with no approximation of any kind —
//! which is what ROADMAP D2's check asks for ("matches the exact envelope sample by sample").
//!
//! **Per block.** A filter's cutoff is not a multiply: it is an input to a *design*, and
//! redesigning a Butterworth cascade per sample would cost more than the filter. These are
//! redesigned every [`BLOCK`] frames and the coefficients handed to the same [`SosStream`] the
//! realtime engine uses, which carries the filter state across the change. The parameter
//! therefore follows a staircase of `BLOCK`-frame steps rather than the curve exactly. A lane
//! moves the coefficients by a hair per block, so carrying the state is both continuous and
//! cheap — see [`SosStream::set_coeffs_now`].
//!
//! Anything else is refused by name. An op whose parameter cannot be varied without redesigning
//! something fluxion has no streaming form for is an error the caller can act on, not a silently
//! frozen parameter.

use fluxion_core::automation::{Automation, Compiled, Lane};
use fluxion_core::{Graph, Op, OpKind, Signal};
use fluxion_ops::Biquad;
use fluxion_rt::stream::SosStream;

use crate::{Cpu, Ctx, eval_with, op_sos};

/// How many frames share one set of designed coefficients.
///
/// 64 frames is 1.33 ms at 48 kHz — 750 redesigns a second, which is nothing next to the filtering
/// itself, and short enough that even a fast sweep moves imperceptibly within one block. Small
/// enough to be inaudible, large enough not to dominate the cost.
pub const BLOCK: usize = 64;

/// Why a render could not be automated.
#[derive(Clone, Debug, PartialEq)]
pub enum AutomationError {
    /// No node in the graph carries this `name:` label.
    UnknownNode {
        /// The label the lane asked for.
        node: String,
    },
    /// The node exists, but the op it wraps has no parameter by that name.
    UnknownParam {
        /// The node's label.
        node: String,
        /// The op's registry name.
        op: &'static str,
        /// The parameter that was asked for.
        param: String,
        /// The parameters the op does have.
        available: Vec<&'static str>,
    },
    /// The node's label does not resolve to a single op.
    ///
    /// A lane drives one parameter of one op; a label wrapping a whole subchain is ambiguous about
    /// which op it meant.
    NotAnOp {
        /// The node's label.
        node: String,
    },
    /// The op exists and has the parameter, but fluxion cannot vary it over time.
    NotAutomatable {
        /// The op's registry name.
        op: &'static str,
        /// The parameter.
        param: String,
        /// What would have to exist for it to work.
        why: &'static str,
    },
}

impl std::fmt::Display for AutomationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AutomationError::UnknownNode { node } => {
                write!(f, "no node labelled '{node}' in the graph")
            }
            AutomationError::UnknownParam {
                node,
                op,
                param,
                available,
            } => write!(
                f,
                "node '{node}' is a '{op}', which has no parameter '{param}' — it has {}",
                available.join(", ")
            ),
            AutomationError::NotAnOp { node } => write!(
                f,
                "node '{node}' labels a subchain rather than a single op, so it is ambiguous \
                 which op's parameter to automate"
            ),
            AutomationError::NotAutomatable { op, param, why } => {
                write!(f, "'{op}' cannot have '{param}' automated: {why}")
            }
        }
    }
}

impl std::error::Error for AutomationError {}

/// How a given op's parameter can be varied.
enum Mode {
    /// Exact, per sample: the parameter is a multiply.
    PerSample,
    /// Per block: the parameter feeds a coefficient design.
    PerBlock,
}

/// Whether `op`'s parameter `index` can be automated, and how.
///
/// One function rather than a column on every parameter in the registry: this is a property of how
/// an op is *implemented*, and it changes when a kernel gains a streaming form — not when someone
/// adds a row to the catalog.
fn mode(op: &Op, index: usize) -> Result<Mode, AutomationError> {
    let name = op.kind.params()[index].name;
    match op.kind {
        // A gain is a multiply. Nothing to design, nothing to carry.
        OpKind::Gain => Ok(Mode::PerSample),
        // Everything that lowers to a biquad cascade can be redesigned and handed to `SosStream`,
        // which carries the state across the change.
        _ if op_sos(op, 48_000).is_some() => Ok(Mode::PerBlock),
        _ => Err(AutomationError::NotAutomatable {
            op: op.kind.name(),
            param: name.to_string(),
            why: "only a gain and the filters have a form whose parameters can change mid-render; \
                  the others would need a streaming kernel that carries their state across a \
                  coefficient change",
        }),
    }
}

/// Resolve every lane against the graph, failing loudly on anything that does not fit.
///
/// Done once, before a sample is rendered, so a mistyped node or parameter is an error rather than
/// a render that silently ignored it.
fn resolve<'a>(
    graph: &Graph,
    automation: &'a Automation,
) -> Result<Vec<(&'a Lane, usize)>, AutomationError> {
    let mut out = Vec::new();
    for lane in automation.lanes() {
        let node = graph
            .find_named(&lane.node)
            .ok_or_else(|| AutomationError::UnknownNode {
                node: lane.node.clone(),
            })?;
        let Graph::Op(op) = node else {
            return Err(AutomationError::NotAnOp {
                node: lane.node.clone(),
            });
        };
        let index = op
            .kind
            .params()
            .iter()
            .position(|p| p.name == lane.param)
            .ok_or_else(|| AutomationError::UnknownParam {
                node: lane.node.clone(),
                op: op.kind.name(),
                param: lane.param.clone(),
                available: op.kind.params().iter().map(|p| p.name).collect(),
            })?;
        mode(op, index)?;
        out.push((lane, index));
    }
    Ok(out)
}

/// Render `graph` over `input` with `automation` driving its parameters.
///
/// The audio is identical to [`process`](crate::process) wherever nothing is automated, so a host
/// can call this unconditionally. Time is absolute: frame 0 of `input` is time 0 of every curve,
/// and [`process_automated_from`] starts the curves elsewhere.
pub fn process_automated(
    graph: &Graph,
    input: &Signal,
    automation: &Automation,
) -> Result<Signal, AutomationError> {
    process_automated_from(graph, input, automation, 0)
}

/// The same, with `input`'s first frame sitting at absolute frame `from` on the timeline.
///
/// This is what makes an automated render seekable: the curves are read at absolute frames, so a
/// region rendered from the middle sees the same parameter values a render from the start saw
/// there (ROADMAP D4).
pub fn process_automated_from(
    graph: &Graph,
    input: &Signal,
    automation: &Automation,
    from: u64,
) -> Result<Signal, AutomationError> {
    let lanes = resolve(graph, automation)?;
    if lanes.is_empty() {
        return Ok(crate::process(graph, input));
    }
    let compiled: Vec<(String, usize, Compiled)> = lanes
        .iter()
        .map(|(lane, index)| (lane.node.clone(), *index, lane.curve.compile(input.fs)))
        .collect();

    let state = AutoCtx {
        lanes: &compiled,
        from,
        label: None,
    };
    Ok(Signal::new(
        input.fs,
        eval_auto(graph, input.channels.clone(), input.fs, &state),
    ))
}

/// The lanes in scope during the walk, and where on the timeline we are.
struct AutoCtx<'a> {
    lanes: &'a [(String, usize, Compiled)],
    from: u64,
    label: Option<&'a str>,
}

impl AutoCtx<'_> {
    /// The compiled lanes for the node currently in scope.
    fn here(&self) -> Vec<(usize, &Compiled)> {
        let Some(label) = self.label else {
            return Vec::new();
        };
        self.lanes
            .iter()
            .filter(|(node, _, _)| node == label)
            .map(|(_, index, curve)| (*index, curve))
            .collect()
    }
}

/// Walk the graph, rendering automated ops against their curves and everything else as usual.
fn eval_auto(graph: &Graph, x: Vec<Vec<f32>>, fs: u32, ctx: &AutoCtx<'_>) -> Vec<Vec<f32>> {
    match graph {
        Graph::Series(a, b) => {
            let y = eval_auto(a, x, fs, ctx);
            eval_auto(b, y, fs, ctx)
        }
        Graph::Parallel(a, b) => {
            let left = eval_auto(a, x.clone(), fs, ctx);
            let right = eval_auto(b, x, fs, ctx);
            crate::Backend::add(&Cpu, left, right)
        }
        Graph::Named { name, node } => eval_auto(
            node,
            x,
            fs,
            &AutoCtx {
                label: Some(name),
                ..*ctx
            },
        ),
        Graph::Op(op) => {
            let lanes = ctx.here();
            if lanes.is_empty() {
                // Nothing automated here: the ordinary whole-buffer path, so an un-automated op in
                // an automated graph is sample-for-sample what it always was.
                return eval_with(&Cpu, graph, x, fs, &Ctx::none());
            }
            eval_op_auto(op, &lanes, x, fs, ctx.from)
        }
        // Everything else is structural and carries no parameters of its own.
        other => eval_with(&Cpu, other, x, fs, &Ctx::none()),
    }
}

/// Render one automated op.
fn eval_op_auto(
    op: &Op,
    lanes: &[(usize, &Compiled)],
    x: Vec<Vec<f32>>,
    fs: u32,
    from: u64,
) -> Vec<Vec<f32>> {
    // `resolve` has already established that every lane here is one of these two.
    match mode(op, lanes[0].0) {
        Ok(Mode::PerSample) => per_sample(op, lanes, x, from),
        _ => per_block(op, lanes, x, fs, from),
    }
}

/// A gain against its curve: exact, one multiply per sample.
fn per_sample(
    op: &Op,
    lanes: &[(usize, &Compiled)],
    mut x: Vec<Vec<f32>>,
    from: u64,
) -> Vec<Vec<f32>> {
    let base = op.params[0];
    // Only `gain` reaches here, and it has one parameter, so there is one lane that matters.
    let curve = lanes.iter().find(|(i, _)| *i == 0).map(|(_, c)| *c);
    for channel in &mut x {
        for (n, sample) in channel.iter_mut().enumerate() {
            let g = match curve {
                Some(c) => c.at(from + n as u64),
                None => base,
            };
            *sample *= g;
        }
    }
    x
}

/// A filter whose design parameters move: redesign every `BLOCK` frames, carrying the state.
fn per_block(
    op: &Op,
    lanes: &[(usize, &Compiled)],
    x: Vec<Vec<f32>>,
    fs: u32,
    from: u64,
) -> Vec<Vec<f32>> {
    let frames = x.first().map_or(0, |c| c.len());
    let design_at = |n: u64| -> Vec<Biquad> {
        let mut params = op.params.clone();
        for (index, curve) in lanes {
            if let Some(slot) = params.get_mut(*index) {
                *slot = curve.at(n);
            }
        }
        // `Op::new` would reject a curve that leaves the parameter's static bounds; clamp instead,
        // because a lane running slightly past a bound should pin to it rather than fail a render
        // half way through.
        let clamped: Vec<f32> = params
            .iter()
            .zip(op.kind.params())
            .map(|(v, spec)| v.clamp(spec.min, spec.max))
            .collect();
        let designed = Op::new(op.kind, clamped).unwrap_or_else(|_| op.clone());
        op_sos(&designed, fs).unwrap_or_default()
    };

    x.into_iter()
        .map(|channel| {
            let mut stream = SosStream::new(design_at(from));
            stream.prepare(BLOCK);
            let mut out = vec![0.0f32; frames];
            let mut at = 0usize;
            while at < frames {
                let len = BLOCK.min(frames - at);
                if at > 0 {
                    // Keep the filter state across the change rather than crossfading two
                    // cascades. A lane moves a cutoff by a hair per block, so the state carried
                    // from the previous block is a good approximation for the new coefficients and
                    // the output is continuous without any blending — where crossfading would
                    // cold-start a second cascade 750 times a second and pay its transient each
                    // time.
                    stream.set_coeffs_now(&design_at(from + at as u64));
                }
                stream.process_block(&channel[at..at + len], &mut out[at..at + len]);
                at += len;
            }
            out
        })
        .collect()
}

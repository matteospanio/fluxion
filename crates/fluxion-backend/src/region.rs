//! Rendering part of a chain (ROADMAP D4).
//!
//! A waveform tile, a loop preview, a re-render of the bar someone just edited: all of them ask for
//! `[from, to)` of a chain rather than the whole thing, and all of them have to get **exactly** the
//! samples the whole render would have put there. Not nearly — a tile that disagrees with the
//! bounce is worse than no tile.
//!
//! # Why this costs what it costs
//!
//! Almost every interesting op has memory. A filter carries its biquad state, an echo carries a
//! ring of past input, a compressor carries its envelope. The sample at frame 100 000 is a function
//! of every sample before it, so there is no honest way to compute it without having computed
//! them — and [`render_region`] does exactly that: it runs the chain from frame 0 and returns the
//! window that was asked for.
//!
//! That makes it **O(to)** rather than O(to - from), which is the right trade for correctness and
//! the wrong one for drawing a thousand tiles. The cheap version is a checkpoint: snapshot each
//! op's state every few seconds and resume from the nearest one. That is a real piece of work with
//! a cost model of its own, and it is not this — see the roadmap. What is here is the correctness
//! floor everything else has to match, and it is exact by construction rather than by testing:
//! the samples returned *are* samples of the whole render.
//!
//! # What is exact, and what is not
//!
//! Ops that need the whole signal to decide anything cannot be regionally rendered in the sense
//! that matters, and [`render_region`] says so rather than returning a plausible answer:
//! `normalize` scales by the peak of what it was given, `loudnorm` measures the programme before
//! it changes it, and `reverse` needs to know where the end is. Rendering `[0, N)` of a chain
//! containing one of those is fine — it is the whole signal — but a shorter window would be scaled
//! by the peak of a fragment, which is not what the full render produces.

use fluxion_core::automation::Automation;
use fluxion_core::{Graph, OpKind, Signal};

use crate::process_automated_from;

/// Why a region could not be rendered.
#[derive(Clone, Debug, PartialEq)]
pub enum RegionError {
    /// The window is not a window: `from` is past `to`.
    Backwards {
        /// The requested start.
        from: usize,
        /// The requested end.
        to: usize,
    },
    /// The chain contains an op whose output depends on the whole signal, so a window of it is not
    /// a window of the full render.
    WholeSignalOp {
        /// The op's registry name.
        op: &'static str,
        /// What about it needs the whole signal.
        why: &'static str,
    },
    /// A lane did not resolve. See [`AutomationError`](crate::AutomationError).
    Automation(crate::AutomationError),
}

impl std::fmt::Display for RegionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegionError::Backwards { from, to } => {
                write!(f, "region [{from}, {to}) ends before it starts")
            }
            RegionError::WholeSignalOp { op, why } => write!(
                f,
                "'{op}' cannot be rendered a region at a time: {why} — render the whole signal, \
                 or take the op out of the chain and apply it to the finished render"
            ),
            RegionError::Automation(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for RegionError {}

/// Whether every op in `graph` can be rendered a window at a time.
fn check_regionable(graph: &Graph) -> Result<(), RegionError> {
    let mut found = None;
    walk(graph, &mut |op| {
        if found.is_some() {
            return;
        }
        found = match op {
            // These are not "stateful" — they are *whole-signal*. Their output at any frame
            // depends on samples that come after it, so a window is not a window of the whole.
            OpKind::Normalize => Some(("normalize", "it scales by the peak of the whole signal")),
            OpKind::Loudnorm => Some((
                "loudnorm",
                "it measures the programme's loudness before it changes anything",
            )),
            OpKind::Reverse => Some(("reverse", "its first sample is the signal's last")),
            OpKind::Limiter => Some((
                "limiter",
                "it looks ahead, so its gain at a frame depends on frames after it",
            )),
            _ => None,
        };
    });
    match found {
        Some((op, why)) => Err(RegionError::WholeSignalOp { op, why }),
        None => Ok(()),
    }
}

/// Visit every op in the graph.
fn walk(graph: &Graph, f: &mut impl FnMut(OpKind)) {
    match graph {
        Graph::Op(op) => f(op.kind),
        Graph::Series(a, b) | Graph::Parallel(a, b) => {
            walk(a, f);
            walk(b, f);
        }
        Graph::Named { node, .. } => walk(node, f),
        Graph::Keyed { node, key } => {
            walk(node, f);
            walk(key, f);
        }
        Graph::Feedback { forward, feedback } => {
            walk(forward, f);
            walk(feedback, f);
        }
        Graph::Id | Graph::Side(_) | Graph::Tap(_) => {}
    }
}

/// Render frames `[from, to)` of `graph` applied to `input`.
///
/// The result is bit-identical to the same window of a whole render, because it *is* that window:
/// the chain runs from frame 0 and everything before `from` is discarded. `to` past the end of the
/// input clamps to it.
///
/// See the module docs for what that costs and which ops are refused.
pub fn render_region(
    graph: &Graph,
    input: &Signal,
    from: usize,
    to: usize,
) -> Result<Signal, RegionError> {
    render_region_automated(graph, input, &Automation::new(), from, to)
}

/// The same, with automation driving the chain's parameters.
///
/// Curves are read at absolute frames, so the window sees the parameter values the whole render
/// would have had there — the property that makes an automated timeline seekable at all.
pub fn render_region_automated(
    graph: &Graph,
    input: &Signal,
    automation: &Automation,
    from: usize,
    to: usize,
) -> Result<Signal, RegionError> {
    if from > to {
        return Err(RegionError::Backwards { from, to });
    }
    check_regionable(graph)?;

    let frames = input.frames();
    let to = to.min(frames);
    let from = from.min(to);

    // Everything up to `to` has to be computed; nothing after it does. Trimming the tail is the one
    // saving available without a state snapshot, and for an early window it is most of the work.
    let head = Signal::new(
        input.fs,
        input
            .channels
            .iter()
            .map(|c| c[..to.min(c.len())].to_vec())
            .collect(),
    );
    let rendered =
        process_automated_from(graph, &head, automation, 0).map_err(RegionError::Automation)?;

    Ok(Signal::new(
        input.fs,
        rendered
            .channels
            .into_iter()
            .map(|c| c[from.min(c.len())..].to_vec())
            .collect(),
    ))
}

/// How many frames [`render_region`] has to compute to produce `[from, to)` — `to`, not
/// `to - from`.
///
/// Exposed so a caller can decide for itself whether to ask. Drawing tiles left to right over a
/// long timeline is quadratic in the number of tiles; drawing the one tile a user is looking at is
/// not, and this is how a caller tells the two apart.
pub fn frames_to_compute(_from: usize, to: usize) -> usize {
    to
}

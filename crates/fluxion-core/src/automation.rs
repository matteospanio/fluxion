//! Curves: one definition of a value that changes over time (ROADMAP S4, D2, D3).
//!
//! Breakpoint automation, an LFO and an ADSR are the same object seen three ways — a list of
//! points, and a rule for how time maps onto it. Writing them once is not tidiness: ROADMAP D3
//! asks that "what you hear live is what renders", and two implementations of the same curve
//! cannot promise that however carefully they are written.
//!
//! # Why frames, not seconds
//!
//! A curve is authored in seconds ([`Point::t`] is `f64`, because `f32` cannot name a frame two
//! hours into a timeline — its resolution there is 23 samples at 48 kHz). But it is *evaluated* on
//! an integer frame grid: [`Curve::compile`] quantizes the points once, and every value after that
//! comes from [`Compiled::at`] with a `u64` frame number.
//!
//! That is what makes the offline and realtime engines agree exactly rather than approximately.
//! Both ask the same function for the value at frame `n`, so there is nothing left to disagree
//! about — no accumulated phase, no block-size dependence, and nothing that a seek could
//! desynchronize.
//!
//! # The one evaluator
//!
//! Every curve value fluxion computes comes from [`segment`]. [`Compiled::at`] finds the pair of
//! knots a frame falls between and calls it; [`SmoothedValue`](../../fluxion_rt/param/struct.SmoothedValue.html)
//! on the audio thread calls it with its own two endpoints. There is no second implementation to
//! drift.

use serde::{Deserialize, Serialize};

/// How a segment travels from one point to the next.
///
/// The shape governs the segment **leaving** the point that carries it, so the last point's shape
/// is unused (there is nothing after it to travel to).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Shape {
    /// Hold the value, then jump at the next point. A stepped parameter, not a ramp.
    Step,
    /// A straight line.
    Linear,
    /// A raised cosine: flat at both ends, steepest in the middle. The same curve as
    /// `FadeShape::HalfSine`, and what a fade wants when a straight line sounds abrupt.
    Cosine,
    /// Exponential with curvature `k`: `(e^{ku} - 1)/(e^k - 1)`.
    ///
    /// `k < 0` is the capacitor discharge that a decay or a release actually is — fast at first,
    /// then a long tail. `k > 0` is the mirror image. `k = 0` is a straight line, and is
    /// canonicalized to [`Shape::Linear`] on construction so that two curves which evaluate
    /// identically also compare equal.
    Exp {
        /// Curvature. Negative decays, positive accelerates.
        k: f32,
    },
}

impl Shape {
    /// The shape that travels from `a` to `b` **geometrically** — a straight line in decibels.
    ///
    /// This is what "fade from 0 dB to -60 dB" means when someone draws it on a dB-scaled lane: a
    /// constant number of decibels per second, which in amplitude is a curve, not a line. A
    /// straight line in amplitude spends most of its time already almost silent — half way through
    /// it is at -6 dB, not -30.
    ///
    /// It is [`Shape::Exp`] with `k = ln(b/a)`, and that is not an approximation of geometric
    /// interpolation but exactly it: substituting gives `a·(b/a)^u`. Both values must be non-zero
    /// and share a sign — there is no geometric path to or through zero — and this returns
    /// [`Shape::Linear`] when they do not, because a fade that cannot be geometric should still be
    /// a fade.
    pub fn geometric(a: f32, b: f32) -> Shape {
        if a == 0.0 || b == 0.0 || a.signum() != b.signum() || !(a.is_finite() && b.is_finite()) {
            return Shape::Linear;
        }
        canonical(Shape::Exp { k: (b / a).ln() })
    }

    /// The fraction of the way from `a` to `b` at normalized position `u ∈ [0, 1]`.
    fn fraction(self, u: f32) -> f32 {
        match self {
            // Step holds until it arrives; `u = 1` is the next point itself.
            Shape::Step => {
                if u >= 1.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Shape::Linear => u,
            Shape::Cosine => 0.5 - 0.5 * (u * std::f32::consts::PI).cos(),
            Shape::Exp { k } => {
                // k is never 0 here (canonicalized to Linear), so the denominator is finite and
                // non-zero for every k a curve can hold.
                ((k * u).exp() - 1.0) / (k.exp() - 1.0)
            }
        }
    }
}

/// The value `u` of the way from `a` to `b` along `shape` — **the** curve evaluator.
///
/// Everything that computes a curve value in fluxion calls this: the offline automation pass, the
/// realtime parameter ramp, the LFO, the ADSR. That is the whole mechanism behind ROADMAP D3.
#[inline]
pub fn segment(a: f32, b: f32, shape: Shape, u: f32) -> f32 {
    let u = u.clamp(0.0, 1.0);
    // Endpoints exactly, whatever the shape's arithmetic would give: a curve that does not reach
    // the value it was told to reach is a bug that shows up as a click.
    if u == 0.0 {
        return a;
    }
    if u == 1.0 {
        return b;
    }
    a + (b - a) * shape.fraction(u)
}

/// One authored breakpoint.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Point {
    /// Time in seconds. `f64` because `f32` cannot name a frame late in a long timeline.
    pub t: f64,
    /// The value here.
    pub v: f32,
    /// How the curve travels from here to the next point.
    pub shape: Shape,
}

impl Point {
    /// A point with a straight line leaving it.
    pub fn new(t: f64, v: f32) -> Point {
        Point {
            t,
            v,
            shape: Shape::Linear,
        }
    }

    /// A point with `shape` leaving it.
    pub fn shaped(t: f64, v: f32, shape: Shape) -> Point {
        Point {
            t,
            v,
            shape: canonical(shape),
        }
    }
}

/// `Exp { k: 0 }` is a straight line; store it as one so equal curves compare equal.
fn canonical(shape: Shape) -> Shape {
    match shape {
        Shape::Exp { k } if k == 0.0 || !k.is_finite() => Shape::Linear,
        other => other,
    }
}

/// How wall-clock time maps onto the point list — the difference between automation, an LFO and an
/// envelope.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum Timing {
    /// Play the points once and hold the last value. Breakpoint automation.
    #[default]
    Once,
    /// Repeat the points at `rate_hz`, starting `phase` of the way in. An LFO.
    ///
    /// The point times are treated as a shape over `[0, 1]` rather than as seconds, so one
    /// description plays at any rate.
    Loop {
        /// Cycles per second.
        rate_hz: f32,
        /// Where in the cycle to start, as a fraction of it.
        phase: f32,
    },
    /// Hold at point `hold` until released, then play the rest. An ADSR.
    Sustain {
        /// Index of the point to hold at — the sustain level.
        hold: usize,
    },
}

/// A value that changes over time: breakpoint automation, an LFO, or an ADSR.
///
/// ```
/// use fluxion_core::automation::{Curve, Point};
///
/// // A one-second fade from unity to silence.
/// let fade = Curve::new([Point::new(0.0, 1.0), Point::new(1.0, 0.0)], Default::default());
/// let compiled = fade.compile(48_000);
/// assert_eq!(compiled.at(0), 1.0);
/// assert_eq!(compiled.at(24_000), 0.5);
/// assert_eq!(compiled.at(48_000), 0.0);
/// // Past the end it holds, rather than wrapping or going silent.
/// assert_eq!(compiled.at(96_000), 0.0);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Curve {
    points: Vec<Point>,
    timing: Timing,
}

impl Curve {
    /// Build a curve from points and a timing.
    ///
    /// Points are sorted by time and any `Exp { k: 0 }` is canonicalized. A curve with no points is
    /// a constant 0; with one point, that constant.
    pub fn new(points: impl IntoIterator<Item = Point>, timing: Timing) -> Curve {
        let mut points: Vec<Point> = points
            .into_iter()
            .map(|p| Point {
                shape: canonical(p.shape),
                ..p
            })
            .collect();
        points.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));
        Curve { points, timing }
    }

    /// A constant — the curve an un-automated parameter has.
    pub fn constant(value: f32) -> Curve {
        Curve::new([Point::new(0.0, value)], Timing::Once)
    }

    /// A straight line from `from` to `to` over `seconds`, then held.
    pub fn ramp(from: f32, to: f32, seconds: f64) -> Curve {
        Curve::new(
            [Point::new(0.0, from), Point::new(seconds, to)],
            Timing::Once,
        )
    }

    /// A fade from `from` to `to` over `seconds` at a constant rate **in decibels**, then held.
    ///
    /// The fade a user means when they drag a gain lane from 0 dB to -60 dB. See
    /// [`Shape::geometric`] for why that is a curve in amplitude rather than a line.
    pub fn db_ramp(from: f32, to: f32, seconds: f64) -> Curve {
        Curve::new(
            [
                Point::shaped(0.0, from, Shape::geometric(from, to)),
                Point::new(seconds, to),
            ],
            Timing::Once,
        )
    }

    /// An LFO sweeping between `low` and `high` at `rate_hz`, starting `phase` (0..1) into its
    /// cycle.
    ///
    /// The shape is a raised cosine, which is a sine in every way that matters here: it starts at
    /// `low`, reaches `high` at the halfway point, and returns — exactly `low + (high-low)·(1 -
    /// cos 2πft)/2`. Built from the same [`Point`]s as any other curve, so an LFO and a hand-drawn
    /// automation lane are not two mechanisms.
    pub fn lfo(rate_hz: f32, low: f32, high: f32, phase: f32) -> Curve {
        Curve::new(
            [
                Point::shaped(0.0, low, Shape::Cosine),
                Point::shaped(0.5, high, Shape::Cosine),
                Point::new(1.0, low),
            ],
            Timing::Loop { rate_hz, phase },
        )
    }

    /// An ADSR envelope: rise to 1 over `attack`, fall to `sustain` over `decay`, hold there until
    /// released, then fall to 0 over `release`.
    ///
    /// The decay and release are [`Shape::Exp`] with negative curvature, because that is what a
    /// physical envelope does — fast at first, then a long tail. Evaluate it with
    /// [`Compiled::at_gated`], passing the frame the note was released on.
    pub fn adsr(attack: f64, decay: f64, sustain: f32, release: f64) -> Curve {
        // -3 gives a recognisably exponential fall that still reaches its endpoint: at the halfway
        // point it has already covered 82% of the distance, against 50% for a straight line.
        const FALL: Shape = Shape::Exp { k: -3.0 };
        Curve::new(
            [
                Point::new(0.0, 0.0),
                Point::shaped(attack, 1.0, FALL),
                Point::shaped(attack + decay, sustain, FALL),
                Point::new(attack + decay + release, 0.0),
            ],
            Timing::Sustain { hold: 2 },
        )
    }

    /// The points, in time order.
    pub fn points(&self) -> &[Point] {
        &self.points
    }

    /// How time maps onto them.
    pub fn timing(&self) -> Timing {
        self.timing
    }

    /// Quantize to the frame grid at `fs` — do this once, then evaluate with [`Compiled::at`].
    ///
    /// This is where seconds become frames, and it is the only place they do. Both engines
    /// evaluate the compiled form, so neither can round a time differently from the other.
    pub fn compile(&self, fs: u32) -> Compiled {
        let fs = fs.max(1);
        let (knots, period) = match self.timing {
            // A looping curve's point times are a shape over [0, 1], scaled by the rate.
            Timing::Loop { rate_hz, phase } => {
                let cycle = if rate_hz > 0.0 {
                    f64::from(fs) / f64::from(rate_hz)
                } else {
                    // A rate of 0 is a held LFO, not a division by zero: one very long cycle.
                    f64::from(u32::MAX)
                };
                let span = self.span().max(1e-12);
                let offset = f64::from(phase.rem_euclid(1.0)) * cycle;
                let knots = self
                    .points
                    .iter()
                    .map(|p| Knot {
                        frame: (((p.t - self.first_t()) / span) * cycle - offset).round() as i64,
                        v: p.v,
                        shape: p.shape,
                    })
                    .collect();
                (knots, Some(cycle.round().max(1.0) as u64))
            }
            _ => {
                let knots = self
                    .points
                    .iter()
                    .map(|p| Knot {
                        frame: (p.t * f64::from(fs)).round() as i64,
                        v: p.v,
                        shape: p.shape,
                    })
                    .collect();
                (knots, None)
            }
        };
        Compiled {
            knots,
            period,
            hold: match self.timing {
                Timing::Sustain { hold } => Some(hold),
                _ => None,
            },
        }
    }

    /// The value at `t` seconds, without compiling — for inspection and tests.
    ///
    /// Prefer [`compile`](Curve::compile) plus [`Compiled::at`] anywhere the answer has to match
    /// another engine's: this samples continuous time, and the engines sample frames.
    pub fn at(&self, t: f64) -> f32 {
        // 1 GHz is far above any audio rate, so this is the continuous-time answer to the
        // precision a caller of this method could care about.
        self.compile(1_000_000_000)
            .at((t * 1e9).round().max(0.0) as u64)
    }

    fn first_t(&self) -> f64 {
        self.points.first().map_or(0.0, |p| p.t)
    }

    fn span(&self) -> f64 {
        match (self.points.first(), self.points.last()) {
            (Some(a), Some(b)) => b.t - a.t,
            _ => 0.0,
        }
    }
}

/// A point quantized to the frame grid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Knot {
    /// Frame this point lands on. Signed, because an LFO's phase offset can put the first knot
    /// before frame 0.
    pub frame: i64,
    /// The value here.
    pub v: f32,
    /// How the curve travels from here to the next knot.
    pub shape: Shape,
}

/// A curve on the frame grid — what both engines actually evaluate.
#[derive(Clone, Debug, PartialEq)]
pub struct Compiled {
    knots: Vec<Knot>,
    /// Cycle length in frames, for a looping curve.
    period: Option<u64>,
    /// Index of the knot to hold at, for a sustaining one.
    hold: Option<usize>,
}

impl Compiled {
    /// The value at absolute frame `n`.
    ///
    /// Before the first knot it is the first value, after the last it is the last — a curve does
    /// not invent a shape outside the range it was drawn over.
    pub fn at(&self, n: u64) -> f32 {
        self.at_gated(n, None)
    }

    /// The value at frame `n`, with a note released at frame `release`.
    ///
    /// Only a [`Timing::Sustain`] curve reads `release`; the others ignore it, so this is the one
    /// entry point and [`at`](Compiled::at) is the sugar.
    pub fn at_gated(&self, n: u64, release: Option<u64>) -> f32 {
        if self.knots.is_empty() {
            return 0.0;
        }
        let n = self.position(n, release);
        value_at(&self.knots, n)
    }

    /// Map an absolute frame onto the frame the point list should be read at.
    fn position(&self, n: u64, release: Option<u64>) -> i64 {
        // A looping curve wraps in whole frames, so the phase cannot drift no matter how long it
        // runs or where a render starts — which is what lets a region render mid-timeline agree
        // with a render from 0.
        if let Some(period) = self.period {
            let first = self.knots[0].frame;
            return first + (n % period) as i64;
        }
        match (self.hold, release) {
            (Some(hold), Some(release)) => {
                let hold_frame = self.knots[hold.min(self.knots.len() - 1)].frame;
                if n < release {
                    // Before the note-off, hold rather than running past the sustain point.
                    (n as i64).min(hold_frame)
                } else {
                    // After it, carry on from the sustain point.
                    hold_frame + (n - release) as i64
                }
            }
            // Never released: hold at the sustain point forever.
            (Some(hold), None) => {
                let hold_frame = self.knots[hold.min(self.knots.len() - 1)].frame;
                (n as i64).min(hold_frame)
            }
            _ => n as i64,
        }
    }

    /// The knots, for inspection.
    pub fn knots(&self) -> &[Knot] {
        &self.knots
    }

    /// The last frame the curve does anything on — after this it holds. `None` for a loop, which
    /// never finishes.
    pub fn end(&self) -> Option<i64> {
        match self.period {
            Some(_) => None,
            None => self.knots.last().map(|k| k.frame),
        }
    }
}

/// Read a knot list at frame `n`. Shared by [`Compiled`] and by the realtime ramp.
///
/// Linear in the number of knots, which is the right shape for the handful a lane has and for the
/// two a ramp has. A lane with thousands of points wants a cursor; that is a change to *this*
/// function, not a second copy of it.
pub fn value_at(knots: &[Knot], n: i64) -> f32 {
    let Some(first) = knots.first() else {
        return 0.0;
    };
    if n <= first.frame {
        return first.v;
    }
    let last = knots[knots.len() - 1];
    if n >= last.frame {
        return last.v;
    }
    for pair in knots.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        if n >= a.frame && n < b.frame {
            let span = (b.frame - a.frame) as f32;
            // Two knots on the same frame are a jump, not a division by zero.
            if span <= 0.0 {
                return b.v;
            }
            return segment(a.v, b.v, a.shape, (n - a.frame) as f32 / span);
        }
    }
    last.v
}

/// One automation lane: a curve driving one parameter of one named node (ROADMAP D2).
///
/// The node is addressed by the `name:` label it was given in the chain — which is what
/// [`Graph::Named`](crate::Graph::Named) has always been for — and the parameter by its name in the
/// registry, not by position. A name survives a parameter being reordered; an index does not, and
/// would retarget silently.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Lane {
    /// The `name:` label of the node to automate.
    pub node: String,
    /// The parameter's name, as the registry spells it (`"cutoff"`, `"gain"`, …).
    pub param: String,
    /// What the parameter does over time.
    pub curve: Curve,
}

impl Lane {
    /// A lane driving `param` of the node labelled `node`.
    pub fn new(node: impl Into<String>, param: impl Into<String>, curve: Curve) -> Lane {
        Lane {
            node: node.into(),
            param: param.into(),
            curve,
        }
    }
}

/// Every lane for a render — the side table handed to the automated process call.
///
/// Deliberately **not** part of [`Graph`](crate::Graph). A graph is a description of a signal
/// path and round-trips through the chain text; automation is a description of a *performance*
/// over a particular stretch of time, and it is the thing a host rebuilds every time a user drags
/// a breakpoint. Keeping them apart means an automated render still prints, parses and freezes as
/// the chain it is.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Automation {
    lanes: Vec<Lane>,
}

impl Automation {
    /// No automation — every parameter holds the value the graph gives it.
    pub fn new() -> Automation {
        Automation::default()
    }

    /// Add a lane.
    pub fn with(mut self, lane: Lane) -> Automation {
        self.lanes.push(lane);
        self
    }

    /// The lanes, in the order they were added.
    pub fn lanes(&self) -> &[Lane] {
        &self.lanes
    }

    /// Whether anything is automated at all — the cheap check that lets a caller skip the
    /// automated path entirely.
    pub fn is_empty(&self) -> bool {
        self.lanes.is_empty()
    }

    /// Every lane targeting the node labelled `name`.
    pub fn for_node(&self, name: &str) -> impl Iterator<Item = &Lane> {
        self.lanes.iter().filter(move |l| l.node == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FS: u32 = 48_000;

    /// The plain case, and the one D2's check is built on: a straight line hits the value the
    /// closed form says, at every frame — not near it.
    #[test]
    fn a_linear_ramp_is_the_closed_form() {
        let c = Curve::ramp(0.0, 1.0, 1.0).compile(FS);
        for n in 0..=48_000u64 {
            let want = n as f32 / 48_000.0;
            let got = c.at(n);
            assert!((got - want).abs() < 1e-6, "frame {n}: {got} vs {want}");
        }
    }

    /// Endpoints are exact whatever the shape does in between. A curve that lands at 0.999 instead
    /// of 1 is a click at the end of every fade.
    #[test]
    fn every_shape_hits_both_endpoints_exactly() {
        for shape in [
            Shape::Step,
            Shape::Linear,
            Shape::Cosine,
            Shape::Exp { k: -3.0 },
            Shape::Exp { k: 4.0 },
        ] {
            assert_eq!(segment(0.25, 0.75, shape, 0.0), 0.25, "{shape:?} at u=0");
            assert_eq!(segment(0.25, 0.75, shape, 1.0), 0.75, "{shape:?} at u=1");
        }
    }

    /// The shapes are what they claim: cosine is flat at the ends and halfway at the middle, a
    /// negative exponential has covered most of the distance by the middle, step has covered none.
    #[test]
    fn the_shapes_have_the_curvature_they_claim() {
        assert!((segment(0.0, 1.0, Shape::Linear, 0.5) - 0.5).abs() < 1e-6);
        assert!((segment(0.0, 1.0, Shape::Cosine, 0.5) - 0.5).abs() < 1e-6);
        // Flat at the ends: a small step near u=0 moves the value far less than linear would.
        assert!(segment(0.0, 1.0, Shape::Cosine, 0.01) < 0.01);

        let decay = segment(0.0, 1.0, Shape::Exp { k: -3.0 }, 0.5);
        assert!(
            (decay - 0.81757).abs() < 0.001,
            "an exponential decay is 82% done at the midpoint, got {decay}"
        );
        assert_eq!(segment(0.0, 1.0, Shape::Step, 0.99), 0.0);
    }

    /// `Exp { k: 0 }` is a straight line, so it must *be* one — otherwise two curves that evaluate
    /// identically compare unequal and a round-trip is not a fixed point.
    #[test]
    fn a_zero_curvature_exponential_is_canonicalized() {
        let p = Point::shaped(0.0, 1.0, Shape::Exp { k: 0.0 });
        assert_eq!(p.shape, Shape::Linear);
        assert_eq!(
            Curve::new(
                [Point::shaped(0.0, 0.0, Shape::Exp { k: 0.0 })],
                Timing::Once
            ),
            Curve::new([Point::new(0.0, 0.0)], Timing::Once)
        );
    }

    /// The LFO reduces to breakpoints *exactly*, not approximately — it is a raised cosine, and
    /// that is what `Shape::Cosine` over two half-cycles is.
    #[test]
    fn the_lfo_is_a_raised_cosine() {
        let rate = 2.0f32;
        let c = Curve::lfo(rate, -1.0, 1.0, 0.0).compile(FS);
        for n in 0..FS as u64 {
            let t = n as f32 / FS as f32;
            let want = -(std::f32::consts::TAU * rate * t).cos();
            let got = c.at(n);
            assert!(
                (got - want).abs() < 1e-3,
                "frame {n}: LFO {got}, raised cosine {want}"
            );
        }
    }

    /// And it keeps its phase for as long as it runs: wrapping in whole frames means cycle
    /// 100 000 is the same shape as cycle 0, which a phase accumulator could not promise.
    #[test]
    fn the_lfo_does_not_drift() {
        let c = Curve::lfo(1.0, 0.0, 1.0, 0.0).compile(FS);
        for cycle in [0u64, 1, 1_000, 100_000] {
            let base = cycle * FS as u64;
            assert_eq!(c.at(base), c.at(0), "cycle {cycle} start");
            assert_eq!(c.at(base + 12_000), c.at(12_000), "cycle {cycle} quarter");
            assert_eq!(c.at(base + 24_000), c.at(24_000), "cycle {cycle} half");
        }
    }

    /// The ADSR: rises, falls to sustain, holds there however long the note is held, then releases
    /// from wherever it was.
    #[test]
    fn the_adsr_holds_until_it_is_released() {
        let c = Curve::adsr(0.01, 0.05, 0.5, 0.2).compile(FS);
        let attack = (0.01 * FS as f64) as u64;
        let sustain_at = ((0.01 + 0.05) * FS as f64) as u64;

        assert_eq!(c.at_gated(0, None), 0.0);
        assert!(
            (c.at_gated(attack, None) - 1.0).abs() < 1e-3,
            "peak of the attack"
        );
        assert!(
            (c.at_gated(sustain_at, None) - 0.5).abs() < 1e-3,
            "sustain level"
        );

        // Held for a second, then for ten: the level is the same, which is what "sustain" means.
        assert!((c.at_gated(FS as u64, None) - 0.5).abs() < 1e-3);
        assert!((c.at_gated(FS as u64 * 10, None) - 0.5).abs() < 1e-3);

        // Released at 1 s, it falls to 0 over the release time and stays there.
        let release = FS as u64;
        assert!((c.at_gated(release, Some(release)) - 0.5).abs() < 1e-3);
        let after = release + (0.2 * FS as f64) as u64;
        assert!(
            c.at_gated(after, Some(release)).abs() < 1e-3,
            "end of the release"
        );
        assert!(
            c.at_gated(after * 2, Some(release)).abs() < 1e-3,
            "and it stays"
        );
    }

    /// Outside the drawn range a curve holds its end values rather than extrapolating a shape
    /// nobody asked for.
    #[test]
    fn a_curve_holds_outside_its_range() {
        let c =
            Curve::new([Point::new(1.0, 0.25), Point::new(2.0, 0.75)], Timing::Once).compile(FS);
        assert_eq!(c.at(0), 0.25);
        assert_eq!(c.at(FS as u64), 0.25);
        assert_eq!(c.at(FS as u64 * 2), 0.75);
        assert_eq!(c.at(FS as u64 * 100), 0.75);
    }

    /// Degenerate inputs give a constant rather than a panic.
    #[test]
    fn empty_and_single_point_curves_are_constants() {
        assert_eq!(Curve::new([], Timing::Once).compile(FS).at(1_000), 0.0);
        assert_eq!(Curve::constant(0.3).compile(FS).at(1_000), 0.3);
    }

    /// Points arrive in whatever order the caller had them; the curve is defined by time.
    #[test]
    fn points_are_sorted_by_time() {
        let c = Curve::new([Point::new(1.0, 1.0), Point::new(0.0, 0.0)], Timing::Once);
        assert_eq!(c.points()[0].t, 0.0);
        assert_eq!(c.compile(FS).at(0), 0.0);
    }

    /// A dB-linear fade is geometric in amplitude, and `Shape::geometric` gives exactly that —
    /// the half-way point of a 0 → -60 dB fade is -30 dB, where a straight line in amplitude
    /// would be at -6 dB. Getting this wrong is the most common automation mistake there is.
    #[test]
    fn a_db_fade_is_geometric_not_linear() {
        let silent = 10f32.powf(-60.0 / 20.0);
        let db = Curve::db_ramp(1.0, silent, 1.0).compile(FS);
        let linear = Curve::ramp(1.0, silent, 1.0).compile(FS);

        let half_db = 20.0 * db.at(24_000).log10();
        let half_linear = 20.0 * linear.at(24_000).log10();
        assert!(
            (half_db + 30.0).abs() < 0.01,
            "half way through a dB fade should be -30 dB, got {half_db:.2}"
        );
        assert!(
            (half_linear + 6.02).abs() < 0.01,
            "half way through a linear fade is -6 dB, got {half_linear:.2}"
        );

        // Constant decibels per second, all the way down: every tenth is 6 dB below the last.
        for tenth in 1..=10u64 {
            let want = -6.0 * tenth as f32;
            let got = 20.0 * db.at(4_800 * tenth).log10();
            assert!(
                (got - want).abs() < 0.02,
                "at {}/10 of the fade: {got:.2} dB, expected {want:.2}",
                tenth
            );
        }
        // Both land on the same endpoints, whatever they do in between.
        assert_eq!(db.at(0), 1.0);
        assert_eq!(db.at(48_000), silent);
    }

    /// A geometric shape needs two non-zero values of the same sign; anything else falls back to a
    /// straight line rather than producing a NaN.
    #[test]
    fn geometric_falls_back_where_it_cannot_apply() {
        assert_eq!(Shape::geometric(0.0, 1.0), Shape::Linear);
        assert_eq!(Shape::geometric(1.0, 0.0), Shape::Linear);
        assert_eq!(Shape::geometric(-1.0, 1.0), Shape::Linear);
        assert_eq!(
            Shape::geometric(0.5, 0.5),
            Shape::Linear,
            "b/a = 1 -> k = 0"
        );
        assert!(matches!(Shape::geometric(1.0, 0.001), Shape::Exp { .. }));
    }

    /// A curve rides in a `.fxg`, so it has to survive serde unchanged.
    #[test]
    fn a_curve_round_trips_through_serde() {
        for curve in [
            Curve::ramp(1.0, 0.0, 2.5),
            Curve::lfo(3.0, -1.0, 1.0, 0.25),
            Curve::adsr(0.01, 0.1, 0.6, 0.3),
            Curve::new(
                [
                    Point::shaped(0.0, 0.0, Shape::Cosine),
                    Point::shaped(1.0, 1.0, Shape::Exp { k: -2.0 }),
                    Point::shaped(2.0, 0.5, Shape::Step),
                ],
                Timing::Once,
            ),
        ] {
            let text = serde_json::to_string(&curve).expect("a curve serializes");
            let back: Curve = serde_json::from_str(&text).expect("and comes back");
            assert_eq!(back, curve);
        }
    }
}

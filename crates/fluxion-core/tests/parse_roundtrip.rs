//! The chain text syntax is the exact inverse of `Graph`'s rendering.
//!
//! Everything downstream leans on this: the CLI's `--chain`, `fluxion.chain()` in Python,
//! `fx_chain_from_text` in C and `Chain.fromText` in the browser all describe a graph with one
//! string, and `--dry-run` / `.fxg` inspection print it back. If parsing were not the exact left
//! inverse of printing, a chain would quietly mean something different depending on which door you
//! came in through.
//!
//! The corpus is written out by hand rather than generated: it pins the precise nesting shapes the
//! writer is able to emit — associativity in both directions, labels around every node kind,
//! feedback nested either way — which is what a property generator tends to under-sample.

use fluxion_core::{Graph, Op, OpKind, ParseError};

/// Every shape the renderer can produce.
fn corpus() -> Vec<Graph> {
    let a = || Graph::op(OpKind::Gain, [0.5]);
    let b = || Graph::op(OpKind::Lowpass, [800.0, 4.0]);
    let c = || Graph::op(OpKind::Peaking, [1000.0, 6.0, 1.5]);
    let mut out = Vec::new();

    // Every op in the catalog, at its defaults.
    for &kind in OpKind::all() {
        out.push(Graph::op(kind, kind.defaults()));
    }

    // The pass-through, alone and composed.
    out.push(Graph::Id);
    out.push(Graph::Id | a());
    out.push(a() | Graph::Id);
    out.push(Graph::Id + Graph::Id);

    // Associativity in both directions, for both operators, and mixed.
    out.push((a() | b()) | c());
    out.push(a() | (b() | c()));
    out.push((a() + b()) + c());
    out.push(a() + (b() + c()));
    out.push((a() + b()) | c());
    out.push(a() | (b() + c()));
    out.push((a() | b()) + c());
    out.push(a() + (b() | c()));

    // Labels around every node kind, and either side of an operator.
    out.push(Graph::named("x", Graph::Id));
    out.push(Graph::named("x", a()));
    out.push(Graph::named("x", a() | b()));
    out.push(Graph::named("x", a() + b()));
    out.push(Graph::named("outer", Graph::named("inner", a())));
    out.push(Graph::named("x", a().feedback(b())));
    out.push(Graph::named("lp", a()) | b());
    out.push(a() | Graph::named("lp", b()));
    out.push(Graph::named("l", a()) + Graph::named("r", b()));

    // Feedback, nested on each side.
    out.push(a().feedback(b()));
    out.push(a().feedback(Graph::Id));
    out.push((a() | b()).feedback(c()));
    out.push(a().feedback(b() | c()));
    out.push((a() + b()).feedback(c()));
    out.push(a().feedback(b()).feedback(c()));
    out.push(a().feedback(b().feedback(c())));

    // Side inputs and keys (ROADMAP S1). `<` is the loosest operator, so what is worth pinning is
    // that a composite on either side survives a reparse.
    out.push(Graph::side(0));
    out.push(Graph::side(3));
    out.push(Graph::side(0) | b());
    out.push(a() + Graph::side(1));
    out.push(a().keyed(Graph::side(0)));
    out.push(a().keyed(Graph::side(0) | b()));
    out.push((a() | b()).keyed(Graph::side(0)));
    out.push((a() + b()).keyed(c()));
    out.push(a().keyed(b().feedback(c())));
    out.push(Graph::named("gated", a().keyed(Graph::side(0))));
    out.push(a().keyed(Graph::side(0)) | b());
    out.push(b() | a().keyed(Graph::side(0)));
    out.push(a().keyed(Graph::side(0)) + b());
    out.push(a().keyed(Graph::side(0)).keyed(b()));

    // Something deep enough to exercise the bracketing rules together.
    out.push(((a() | b()) + (c() | a())) | Graph::named("tail", b() + c()));

    // The variadic op at several lengths.
    out.push(Graph::op(OpKind::Fir, [1.0]));
    out.push(Graph::op(OpKind::Fir, [0.5, 0.3, 0.2]));
    out.push(Graph::op(OpKind::Fir, [-0.25, 0.0, 1.0, -1.0, 0.125]));

    // Numeric edges the renderer formats differently: whole numbers, fractions, exponents,
    // negatives, signed zero, and the unbounded value `Op::new` actually accepts.
    out.push(Graph::op(OpKind::Peaking, [20.0, -12.5, 0.707]));
    out.push(Graph::op(OpKind::Lowpass, [1e7, 16.0]));
    out.push(Graph::op(OpKind::Peaking, [1000.0, 0.0, 1e-3]));
    out.push(Graph::op(
        OpKind::Compand,
        [0.001, 0.25, -60.0, 100.0, 0.0, -48.0],
    ));
    out.push(Graph::op(OpKind::Gain, [-0.0]));
    out.push(Graph::op(OpKind::Gain, [f32::INFINITY]));
    out.push(Graph::op(OpKind::Gain, [f32::NEG_INFINITY]));
    out.push(Graph::op(
        OpKind::Biquad,
        [1.0, -1.999_9, 0.999_9, -1.5, 0.75],
    ));

    out
}

/// Pre-condition: a graph built in Rust, rendered with `Display`.
/// Post-condition: parsing that text returns the identical graph, and re-rendering it is a
/// fixed point — so the printed form really is canonical.
#[test]
fn parse_is_the_left_inverse_of_display() {
    for graph in corpus() {
        let text = graph.to_string();
        let back = text
            .parse::<Graph>()
            .unwrap_or_else(|e: ParseError| panic!("{}", e.render(&text)));
        assert_eq!(back, graph, "round-trip changed the graph: `{text}`");
        assert_eq!(back.to_string(), text, "rendering is not a fixed point");
    }
}

/// The coverage check for the interfaces that reach ops **through text** rather than through one
/// symbol per op — C and JS. If an op is not reachable from its bare name, it is not on those
/// interfaces at all, whatever `docs/ops.md` claims.
#[test]
fn every_registry_op_parses_from_its_bare_name() {
    for &kind in OpKind::all() {
        let graph = kind
            .name()
            .parse::<Graph>()
            .unwrap_or_else(|e: ParseError| panic!("op '{}': {e}", kind.name()));
        assert_eq!(
            graph,
            Graph::Op(Op::new(kind, kind.defaults()).expect("defaults are valid by construction")),
            "op '{}' did not parse to itself at its defaults",
            kind.name()
        );
    }
}

/// Every op survives the full trip at its defaults, including the zero-parameter and variadic ones.
#[test]
fn every_registry_op_round_trips_at_defaults() {
    for &kind in OpKind::all() {
        let graph = Graph::op(kind, kind.defaults());
        let text = graph.to_string();
        assert_eq!(
            text.parse::<Graph>().ok(),
            Some(graph),
            "op '{}' failed to round-trip as `{text}`",
            kind.name()
        );
    }
}

/// The input conveniences — the `name=value` shorthand, omitted trailing parameters, named
/// arguments, unit suffixes — all collapse to one canonical rendering. There is exactly one way a
/// graph comes back out, however it went in.
#[test]
fn every_shorthand_normalizes_to_the_canonical_form() {
    let equivalent = [
        [
            "highpass(80, 2)",
            "highpass(80)",
            "highpass=80",
            "highpass(cutoff=80)",
        ],
        [
            "lowpass(1000, 2)",
            "lowpass",
            "lowpass=1k",
            "lowpass(order=2)",
        ],
        [
            "fir(0.5, 0.3)",
            "fir=0.5,0.3",
            "fir(0.5, 0.3)",
            "fir(0.5,0.3)",
        ],
    ];
    for spellings in equivalent {
        let canonical = spellings[0].parse::<Graph>().unwrap().to_string();
        for spelling in spellings {
            let graph = spelling
                .parse::<Graph>()
                .unwrap_or_else(|e: ParseError| panic!("{}", e.render(spelling)));
            assert_eq!(
                graph.to_string(),
                canonical,
                "`{spelling}` rendered differently"
            );
        }
        assert!(
            !canonical.contains('=') && !canonical.contains("dB"),
            "the canonical form should carry no shorthand or suffixes: `{canonical}`"
        );
    }

    // A suffix is applied on the way in and never printed again.
    let gain = "gain=-3dB".parse::<Graph>().unwrap();
    match &gain {
        Graph::Op(op) => assert!((op.params[0] - 0.707_945_8).abs() < 1e-6, "{:?}", op.params),
        other => panic!("expected a gain op, got {other}"),
    }
    assert_eq!(gain.to_string().parse::<Graph>().unwrap(), gain);
}

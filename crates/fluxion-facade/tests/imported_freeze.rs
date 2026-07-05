//! An imported checkpoint (`fluxion import` output: a series of raw `biquad` ops)
//! must keep flowing through the certified realtime freeze pipeline — freeze to a
//! `FrozenSos` plan and lower to an `RtGraph` for hot-swap.
#![cfg(feature = "realtime")]

use fluxion::{Graph, OpKind};

#[test]
fn imported_biquad_chain_freezes_and_lowers() {
    let g = Graph::op(OpKind::Biquad, [1.0, 0.5, 0.25, -0.3, 0.1])
        | Graph::op(OpKind::Biquad, [0.8, 0.0, 0.0, -0.2, 0.0]);
    let frozen = fluxion::freeze(&g, 48_000).expect("pure biquad series must freeze");
    assert_eq!(frozen.sections.len(), 2);
    assert!(fluxion::to_rt_graph(&g, 48_000).is_some());
}

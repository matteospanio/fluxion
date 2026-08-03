//! On-disk compatibility guard for the `.fxg` format.
//!
//! `Graph` and `OpKind` are serialized with serde's default enum representation, which keys on the
//! **Rust variant identifier** (`"Lowpass"`, `"LowShelf"`), not on the DSL name [`OpKind::name`]
//! returns. So renaming a DSL string is invisible to saved graphs, while renaming a *variant* would
//! silently orphan every `.fxg` a user has on disk — with no compile error anywhere.
//!
//! `tests/fixtures/all_ops_v1.fxg` is a frozen file holding one op of every kind that existed when
//! it was written, wrapped so that every `Graph` variant name appears too. The tests below are the
//! tripwire: refactors of the op catalog may reshape it freely, but that file must keep loading.

use fluxion_core::{Graph, OpKind, fxg};

/// The committed pre-refactor file. `include_str!` resolves relative to this source file.
const V1: &str = include_str!("fixtures/all_ops_v1.fxg");

/// Ops live when the fixture was written. This is a frozen fact about the file, not about today's
/// catalog — the catalog may grow, but none of these 27 may disappear or be renamed.
const V1_OP_COUNT: usize = 27;

/// A graph exercising every `OpKind` at its defaults plus every `Graph` variant, so the fixture
/// pins the serde name of all of them.
fn every_op_graph() -> Graph {
    let series = OpKind::all()
        .iter()
        .fold(Graph::Id, |acc, &k| acc | Graph::op(k, k.defaults()));
    Graph::named("all", series + Graph::Id).feedback(Graph::Id)
}

/// Pre-condition: `all_ops_v1.fxg` was written by an earlier version of this crate.
/// Post-condition: it still deserializes, and still contains every op it did when written.
///
/// A failure here means a variant identifier moved. Fix the rename — do **not** regenerate the
/// fixture — unless you are deliberately breaking the format and bumping `FORMAT_VERSION`.
#[test]
fn the_v1_file_with_every_op_still_loads() {
    let loaded = fxg::from_json(V1).expect("the committed v1 graph must still load");
    assert_eq!(
        loaded.leaf_count(),
        V1_OP_COUNT,
        "the v1 fixture lost ops: a variant was renamed or removed"
    );
}

/// Post-condition: every op in *today's* catalog survives a save/load round-trip. This is the
/// living half of the check — the fixture pins the past, this pins the present.
#[test]
fn every_current_op_round_trips_through_fxg() {
    for &kind in OpKind::all() {
        let g = Graph::op(kind, kind.defaults());
        let back = fxg::from_json(&fxg::to_json(&g))
            .unwrap_or_else(|e| panic!("op '{}' failed to round-trip: {e}", kind.name()));
        assert_eq!(back, g, "op '{}' changed across a round-trip", kind.name());
    }
}

/// Regenerate the fixture. Only legitimate when the op set grows *and* you have confirmed no
/// existing variant moved (the point of the file is to catch exactly that):
///
/// ```text
/// cargo test -p fluxion-core --test fxg_compat -- --ignored write_fixture
/// ```
///
/// Then bump `V1_OP_COUNT` to match and say why in the commit message.
#[test]
#[ignore = "writes the committed fixture; run explicitly"]
fn write_fixture() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/all_ops_v1.fxg");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, fxg::to_json(&every_op_graph())).unwrap();
    eprintln!("wrote {} ops to {}", OpKind::all().len(), path.display());
}

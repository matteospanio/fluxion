//! `fluxion-ffi` — the stable C ABI surface, consumed by C / C++ / Swift / JS and (later) via
//! DLPack for zero-copy tensor handoff.
//!
//! Backbone: a single `extern "C"` smoke function proves the `cdylib`/`staticlib` builds and
//! links against the core. The real surface — graph build/parse, `fx_process_dlpack`, and the
//! lifecycle functions — lands later; the C header will be generated with `cbindgen`.

use std::os::raw::c_int;

use fluxion_core::Graph;

/// ABI smoke test: builds a trivial `gain | gain` graph and returns its leaf count (`2`).
///
/// Exists so the C ABI links and round-trips before the real surface is built.
///
/// # Safety
/// None — takes no pointers and has no preconditions.
#[unsafe(no_mangle)]
pub extern "C" fn fx_abi_smoke() -> c_int {
    let g = Graph::op("gain", [1.0]) | Graph::op("gain", [1.0]);
    g.leaf_count() as c_int
}

#[cfg(test)]
mod tests {
    use super::fx_abi_smoke;

    #[test]
    fn abi_smoke_links() {
        assert_eq!(fx_abi_smoke(), 2);
    }
}

//! `fluxion-ffi` — the stable C ABI surface, consumed by C / C++ / Swift / JS.
//!
//! A minimal but *real* surface (plan tasks K1–K3): load a reified graph from a `.fxg` file, run it
//! over an interleaved audio buffer in place, and free it. Errors are reported out-of-band via a
//! thread-local last-error string ([`fx_last_error`]) so the value-returning calls can use a plain
//! sentinel (NULL / negative status). Every entry point that executes fluxion code is wrapped in
//! [`catch_unwind`]: a Rust panic must **never** unwind across the `extern "C"` boundary (that is
//! UB) — it is converted to an error status instead ([`fx_last_error`] itself is a plain
//! thread-local read and cannot panic). DLPack zero-copy tensor handoff and graph-building calls
//! land later.
//!
//! The C header (`include/fluxion.h`) is generated from these signatures with cbindgen (config in
//! `cbindgen.toml`) and checked in; `examples/smoke.c` links against the staticlib to prove the ABI.

use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::{ptr, slice};

use fluxion_core::{Graph, OpKind, Signal};

/// Success.
pub const FX_OK: c_int = 0;
/// A required pointer argument was NULL.
pub const FX_ERR_NULL_ARG: c_int = -1;
/// An argument was structurally invalid (e.g. `channels == 0`, or `frames * channels` overflows).
pub const FX_ERR_INVALID_ARG: c_int = -2;
/// A panic was caught at the FFI boundary and stopped from unwinding into C.
pub const FX_ERR_PANIC: c_int = -3;

thread_local! {
    /// The most recent error message on *this* thread. Owned `CString` so [`fx_last_error`] can hand
    /// out a stable pointer; overwritten by the next fallible call on the thread.
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

/// Store `msg` as this thread's last error (scrubbing any interior NUL so it always round-trips).
fn set_last_error(msg: impl Into<Vec<u8>>) {
    let scrubbed: Vec<u8> = msg
        .into()
        .into_iter()
        .map(|b| if b == 0 { b'?' } else { b })
        .collect();
    let c = CString::new(scrubbed).unwrap_or_else(|_| CString::new("error").unwrap());
    LAST_ERROR.with(|e| *e.borrow_mut() = Some(c));
}

/// Clear this thread's last error (called at the start of a fallible entry point).
fn clear_last_error() {
    LAST_ERROR.with(|e| *e.borrow_mut() = None);
}

/// An owned, reified DSP graph. Opaque to C: construct with [`fx_graph_load_fxg`], pass by pointer,
/// release with [`fx_graph_free`].
pub struct FxGraph {
    graph: Graph,
}

/// Return this thread's most recent error message as a NUL-terminated C string, or NULL if the last
/// fluxion call on this thread succeeded.
///
/// The pointer is owned by fluxion and valid only until the **next** fluxion call on this thread;
/// copy the string if you need to keep it. Never call `free` on it.
///
/// # Safety
/// None to uphold on the caller side, but the returned pointer must be treated as borrowed (see above).
#[unsafe(no_mangle)]
pub extern "C" fn fx_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ref().map_or(ptr::null(), |c| c.as_ptr()))
}

/// Load a reified graph from a `.fxg` file. Returns an owning handle, or NULL on error (see
/// [`fx_last_error`]). Free the handle with [`fx_graph_free`].
///
/// # Safety
/// `path` must be either NULL or a valid pointer to a NUL-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fx_graph_load_fxg(path: *const c_char) -> *mut FxGraph {
    clear_last_error();
    if path.is_null() {
        set_last_error("fx_graph_load_fxg: path is NULL");
        return ptr::null_mut();
    }
    let loaded = catch_unwind(AssertUnwindSafe(|| {
        let c = unsafe { CStr::from_ptr(path) };
        let path_str = c
            .to_str()
            .map_err(|_| "fx_graph_load_fxg: path is not valid UTF-8".to_string())?;
        fluxion_core::fxg::load(path_str).map_err(|e| format!("fx_graph_load_fxg: {e}"))
    }));
    match loaded {
        Ok(Ok(graph)) => Box::into_raw(Box::new(FxGraph { graph })),
        Ok(Err(msg)) => {
            set_last_error(msg);
            ptr::null_mut()
        }
        Err(_) => {
            set_last_error("fx_graph_load_fxg: panic while loading graph");
            ptr::null_mut()
        }
    }
}

/// Free a graph handle returned by [`fx_graph_load_fxg`]. NULL is a no-op; double-free is undefined.
///
/// # Safety
/// `graph` must be NULL or a pointer previously returned by [`fx_graph_load_fxg`] and not yet freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fx_graph_free(graph: *mut FxGraph) {
    clear_last_error(); // a successful free must not leave a stale prior error visible
    if !graph.is_null()
        && catch_unwind(AssertUnwindSafe(|| drop(unsafe { Box::from_raw(graph) }))).is_err()
    {
        set_last_error("fx_graph_free: panic during drop");
    }
}

/// Process an interleaved audio buffer in place through `graph`.
///
/// **Contract.** `data` points to `frames * channels` `f32`s in **interleaved (frame-major)** order —
/// `[c0f0, c1f0, …, c0f1, c1f1, …]`. `fs` is the sample rate in Hz. The graph runs on the CPU batch
/// engine; the result (same shape — the op set is length- and channel-preserving) is written back
/// over `data`. Returns [`FX_OK`] or a negative `FX_ERR_*` (see [`fx_last_error`] for a message).
///
/// This is the offline/batch path, not the realtime engine — it allocates; do not call it from an
/// audio callback.
///
/// # Safety
/// `graph` must be a live handle from [`fx_graph_load_fxg`]. If `frames * channels > 0`, `data` must
/// point to at least that many writable, initialized `f32`s.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fx_process(
    graph: *const FxGraph,
    data: *mut f32,
    frames: usize,
    channels: usize,
    fs: u32,
) -> c_int {
    clear_last_error();
    if graph.is_null() || data.is_null() {
        set_last_error("fx_process: NULL graph or data pointer");
        return FX_ERR_NULL_ARG;
    }
    if channels == 0 {
        set_last_error("fx_process: channels must be > 0");
        return FX_ERR_INVALID_ARG;
    }
    if frames == 0 {
        return FX_OK; // nothing to process
    }
    let Some(len) = frames.checked_mul(channels) else {
        set_last_error("fx_process: frames * channels overflows usize");
        return FX_ERR_INVALID_ARG;
    };
    let ran = catch_unwind(AssertUnwindSafe(|| {
        let g = unsafe { &(*graph).graph };
        let buf = unsafe { slice::from_raw_parts_mut(data, len) };
        // Deinterleave frame-major → planar channel-first (what `Signal` holds).
        let mut planar: Vec<Vec<f32>> = vec![vec![0.0f32; frames]; channels];
        for f in 0..frames {
            for (c, plane) in planar.iter_mut().enumerate() {
                plane[f] = buf[f * channels + c];
            }
        }
        let out = fluxion_backend::process(g, &Signal::new(fs, planar));
        // Re-interleave in place. Ops are length/channel preserving; clamp defensively so a short
        // or narrow result can never write out of bounds.
        for f in 0..frames {
            for c in 0..channels {
                let v = out.channels.get(c).and_then(|ch| ch.get(f)).copied();
                buf[f * channels + c] = v.unwrap_or(0.0);
            }
        }
    }));
    match ran {
        Ok(()) => FX_OK,
        Err(_) => {
            set_last_error("fx_process: panic during processing");
            FX_ERR_PANIC
        }
    }
}

/// ABI smoke test: builds a trivial `gain | gain` graph and returns its leaf count (`2`), or
/// [`FX_ERR_PANIC`] if the graph construction panicked.
///
/// Exists so the C ABI links and round-trips with no arguments or state.
///
/// # Safety
/// None — takes no pointers and has no preconditions.
#[unsafe(no_mangle)]
pub extern "C" fn fx_abi_smoke() -> c_int {
    clear_last_error();
    match catch_unwind(|| {
        let g = Graph::op(OpKind::Gain, [1.0]) | Graph::op(OpKind::Gain, [1.0]);
        g.leaf_count() as c_int
    }) {
        Ok(n) => n,
        Err(_) => {
            set_last_error("fx_abi_smoke: panic");
            FX_ERR_PANIC
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn abi_smoke_links() {
        assert_eq!(fx_abi_smoke(), 2);
    }

    #[test]
    fn load_process_free_roundtrip() {
        // Save a gain(0.5) graph, load it through the C ABI, and process interleaved stereo audio.
        let g = Graph::op(OpKind::Gain, [0.5]);
        let mut path = std::env::temp_dir();
        path.push(format!("fluxion_ffi_test_{}.fxg", std::process::id()));
        fluxion_core::fxg::save(&g, &path).unwrap();
        let c_path = CString::new(path.to_str().unwrap()).unwrap();

        let handle = unsafe { fx_graph_load_fxg(c_path.as_ptr()) };
        assert!(!handle.is_null(), "load returned NULL");

        // 2 channels × 3 frames, interleaved [c0f0, c1f0, c0f1, c1f1, c0f2, c1f2].
        let mut data = vec![1.0f32, 2.0, 1.0, 2.0, 1.0, 2.0];
        let st = unsafe { fx_process(handle, data.as_mut_ptr(), 3, 2, 48_000) };
        assert_eq!(st, FX_OK);
        for (i, &v) in data.iter().enumerate() {
            let want = if i % 2 == 0 { 0.5 } else { 1.0 }; // channel 0 was 1.0, channel 1 was 2.0
            assert!((v - want).abs() < 1e-6, "sample {i} = {v}");
        }

        unsafe { fx_graph_free(handle) };
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn null_args_are_reported_not_crashed() {
        let st = unsafe { fx_process(ptr::null(), ptr::null_mut(), 4, 2, 48_000) };
        assert_eq!(st, FX_ERR_NULL_ARG);

        let handle = unsafe { fx_graph_load_fxg(ptr::null()) };
        assert!(handle.is_null());
        assert!(
            !fx_last_error().is_null(),
            "a NULL path should leave a last-error message"
        );

        // Freeing NULL is a safe no-op — and, as a successful call, clears the stale error.
        unsafe { fx_graph_free(ptr::null_mut()) };
        assert!(
            fx_last_error().is_null(),
            "a successful call must clear the previous error"
        );
    }

    #[test]
    fn zero_frames_is_ok_noop() {
        let g = Graph::op(OpKind::Gain, [0.5]);
        let handle = Box::into_raw(Box::new(FxGraph { graph: g }));
        let mut data = [0.0f32; 0];
        let st = unsafe { fx_process(handle, data.as_mut_ptr(), 0, 2, 48_000) };
        assert_eq!(st, FX_OK);
        unsafe { fx_graph_free(handle) };
    }
}

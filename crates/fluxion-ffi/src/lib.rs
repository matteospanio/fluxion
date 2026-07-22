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

use fluxion_backend::Verdict;
use fluxion_core::{Graph, OpKind, Signal};
use fluxion_ops::{Biquad, certify_sos};
use fluxion_rt::RtGraph;

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

// --- realtime surface (fx_rt_*) ---------------------------------------------------------------
//
// The streaming counterpart of `fx_process`: lower a loaded graph once (designing coefficients at a
// fixed sample rate, certifying stability, pre-sizing every scratch buffer), then filter mono blocks
// in a hot loop with state carried across calls. One `FxRtGraph` is one mono stream — for
// multichannel audio, build one per channel. Handles are not thread-safe: drive a given handle from
// one thread at a time (`fx_rt_set_coeffs` between `fx_rt_process` calls, or from the same thread).

/// Stability verdict code (see `fx_rt_new`): strictly inside the stable region.
pub const FX_VERDICT_CERTIFIED_STABLE: c_int = 0;
/// Stability verdict code: on / within f32-tolerance of the stability boundary.
pub const FX_VERDICT_MARGINALLY_STABLE: c_int = 1;
/// Stability verdict code: could not be evaluated (non-finite coefficients).
pub const FX_VERDICT_INDETERMINATE: c_int = 2;
/// Stability verdict code: no certificate is available for this construct.
pub const FX_VERDICT_NOT_CERTIFIED: c_int = 3;
/// Stability verdict code: provably outside the stable region (refused by `fx_rt_new`).
pub const FX_VERDICT_UNSTABLE: c_int = 4;

/// An owned realtime executor for one mono stream. Opaque to C: construct with [`fx_rt_new`], pass
/// by pointer, release with [`fx_rt_free`].
pub struct FxRtGraph {
    rt: RtGraph,
    max_block: usize,
}

fn verdict_code(v: Verdict) -> c_int {
    match v {
        Verdict::CertifiedStable => FX_VERDICT_CERTIFIED_STABLE,
        Verdict::MarginallyStable => FX_VERDICT_MARGINALLY_STABLE,
        Verdict::Indeterminate => FX_VERDICT_INDETERMINATE,
        Verdict::NotCertified => FX_VERDICT_NOT_CERTIFIED,
        Verdict::Unstable => FX_VERDICT_UNSTABLE,
    }
}

/// Lower `graph` to a realtime executor at sample rate `fs`, sized for blocks up to `max_block`
/// samples. Returns an owning handle, or NULL on error (see [`fx_last_error`]); free with
/// [`fx_rt_free`].
///
/// The graph's coefficients are designed once at `fs` and certified: if non-NULL, `verdict_out`
/// receives the `FX_VERDICT_*` code and `margin_out` the stability margin (`1 − spectral radius`;
/// NaN if indeterminate) — written whenever certification ran, including when construction is then
/// refused. An `FX_VERDICT_UNSTABLE` graph is refused. Graphs containing an op with no realtime
/// lowering (normalize / fade / reverse / the modulated effects / feedback) are refused too.
///
/// All allocation happens here; [`fx_rt_process`] is allocation-free afterwards.
///
/// # Safety
/// `graph` must be NULL or a live handle from [`fx_graph_load_fxg`]. `verdict_out` and `margin_out`
/// must each be NULL or valid for a write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fx_rt_new(
    graph: *const FxGraph,
    fs: u32,
    max_block: usize,
    verdict_out: *mut c_int,
    margin_out: *mut f32,
) -> *mut FxRtGraph {
    clear_last_error();
    if graph.is_null() {
        set_last_error("fx_rt_new: graph is NULL");
        return ptr::null_mut();
    }
    if fs == 0 || max_block == 0 {
        set_last_error("fx_rt_new: fs and max_block must be > 0");
        return ptr::null_mut();
    }
    let built = catch_unwind(AssertUnwindSafe(|| {
        let g = unsafe { &(*graph).graph };
        let cert = fluxion_backend::certify_graph(g, fs);
        let rt = if cert.verdict == Verdict::Unstable {
            Err(format!(
                "fx_rt_new: graph is unstable at fs={fs} (margin {:.3e})",
                cert.margin
            ))
        } else {
            match fluxion_backend::to_rt_graph(g, fs) {
                Some(mut rt) => {
                    rt.prepare(max_block);
                    Ok(rt)
                }
                None => Err("fx_rt_new: graph contains an op with no realtime lowering \
                     (normalize / fade / reverse / a modulated effect / feedback)"
                    .to_string()),
            }
        };
        (cert, rt)
    }));
    match built {
        Ok((cert, rt)) => {
            if !verdict_out.is_null() {
                unsafe { verdict_out.write(verdict_code(cert.verdict)) };
            }
            if !margin_out.is_null() {
                unsafe { margin_out.write(cert.margin) };
            }
            match rt {
                Ok(rt) => Box::into_raw(Box::new(FxRtGraph { rt, max_block })),
                Err(msg) => {
                    set_last_error(msg);
                    ptr::null_mut()
                }
            }
        }
        Err(_) => {
            set_last_error("fx_rt_new: panic while lowering graph");
            ptr::null_mut()
        }
    }
}

/// Filter one mono block: read `len` `f32`s from `input`, write `len` filtered `f32`s to `output`.
/// `len` must be ≤ the `max_block` given to [`fx_rt_new`]; the two buffers must **not** overlap
/// (in-place processing is rejected as [`FX_ERR_INVALID_ARG`]). Filter state is carried across
/// calls — streaming a signal in chunks equals filtering it whole. Returns [`FX_OK`] or a negative
/// `FX_ERR_*`.
///
/// Allocation-free and lock-free: safe to call from an audio callback. (The one caveat: entering
/// any `fx_*` call clears a *stale* error message left by a previous failed call on the same
/// thread, which frees that string — in a callback loop that isn't erroring, this is a no-op.)
///
/// # Safety
/// `rt` must be NULL or a live handle from [`fx_rt_new`], driven by one thread at a time. If
/// `len > 0`, `input` must point to `len` readable `f32`s and `output` to `len` writable `f32`s.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fx_rt_process(
    rt: *mut FxRtGraph,
    input: *const f32,
    output: *mut f32,
    len: usize,
) -> c_int {
    clear_last_error();
    if rt.is_null() || input.is_null() || output.is_null() {
        set_last_error("fx_rt_process: NULL rt, input or output pointer");
        return FX_ERR_NULL_ARG;
    }
    if len == 0 {
        return FX_OK; // nothing to process
    }
    let handle = unsafe { &mut *rt };
    if len > handle.max_block {
        set_last_error("fx_rt_process: len exceeds the max_block given to fx_rt_new");
        return FX_ERR_INVALID_ARG;
    }
    // Overlapping in/out would alias a `&[f32]` with a `&mut [f32]` (UB in Rust) — reject.
    let bytes = len * size_of::<f32>(); // len ≤ max_block, far from overflow
    let (i0, o0) = (input as usize, output as usize);
    if i0 < o0 + bytes && o0 < i0 + bytes {
        set_last_error("fx_rt_process: input and output buffers must not overlap");
        return FX_ERR_INVALID_ARG;
    }
    let ran = catch_unwind(AssertUnwindSafe(|| {
        let x = unsafe { slice::from_raw_parts(input, len) };
        let y = unsafe { slice::from_raw_parts_mut(output, len) };
        handle.rt.process(x, y);
    }));
    match ran {
        Ok(()) => FX_OK,
        Err(_) => {
            set_last_error("fx_rt_process: panic during processing");
            FX_ERR_PANIC
        }
    }
}

/// Swap the `node`-th filter's coefficients (depth-first order over the graph's filter nodes, see
/// [`fx_rt_filter_count`]) to a new cascade of `n_sections` second-order sections, equal-power
/// crossfaded over `fade_samples` (0 = swap at the next sample). `sections` points to
/// `n_sections * 5` `f32`s, `[b0, b1, b2, a1, a2]` per section (`a0` normalised to 1).
///
/// The incoming cascade is certified first; unstable sections are refused. Returns [`FX_OK`] or a
/// negative `FX_ERR_*` (out-of-range `node` is [`FX_ERR_INVALID_ARG`]).
///
/// This is a **control-plane** call: it allocates (and may reallocate the incoming-coefficient
/// storage when the section count changes), so call it from a control thread between
/// [`fx_rt_process`] calls — not from the audio callback.
///
/// # Safety
/// `rt` must be NULL or a live handle from [`fx_rt_new`], driven by one thread at a time.
/// `sections` must be NULL or point to `n_sections * 5` readable `f32`s.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fx_rt_set_coeffs(
    rt: *mut FxRtGraph,
    node: usize,
    sections: *const f32,
    n_sections: usize,
    fade_samples: u32,
) -> c_int {
    clear_last_error();
    if rt.is_null() || sections.is_null() {
        set_last_error("fx_rt_set_coeffs: NULL rt or sections pointer");
        return FX_ERR_NULL_ARG;
    }
    let Some(len) = n_sections.checked_mul(5).filter(|_| n_sections > 0) else {
        set_last_error(
            "fx_rt_set_coeffs: n_sections must be > 0 (and n_sections * 5 must not overflow)",
        );
        return FX_ERR_INVALID_ARG;
    };
    let handle = unsafe { &mut *rt };
    let ran = catch_unwind(AssertUnwindSafe(|| {
        let flat = unsafe { slice::from_raw_parts(sections, len) };
        let sos: Vec<Biquad> = flat
            .chunks_exact(5)
            .map(|c| Biquad {
                b0: c[0],
                b1: c[1],
                b2: c[2],
                a1: c[3],
                a2: c[4],
            })
            .collect();
        let cert = certify_sos(&sos);
        if cert.verdict == Verdict::Unstable {
            return Err(format!(
                "fx_rt_set_coeffs: incoming sections are unstable (margin {:.3e})",
                cert.margin
            ));
        }
        if !handle.rt.set_coeffs(node, &sos, fade_samples) {
            return Err(format!(
                "fx_rt_set_coeffs: node {node} out of range (filter count = {})",
                handle.rt.filter_count()
            ));
        }
        Ok(())
    }));
    match ran {
        Ok(Ok(())) => FX_OK,
        Ok(Err(msg)) => {
            set_last_error(msg);
            FX_ERR_INVALID_ARG
        }
        Err(_) => {
            set_last_error("fx_rt_set_coeffs: panic during swap");
            FX_ERR_PANIC
        }
    }
}

/// Reset all filter state and delay lines to silence (as if freshly built) — call on seek or
/// source switch. Returns [`FX_OK`] or a negative `FX_ERR_*`.
///
/// # Safety
/// `rt` must be NULL or a live handle from [`fx_rt_new`], driven by one thread at a time.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fx_rt_reset(rt: *mut FxRtGraph) -> c_int {
    clear_last_error();
    if rt.is_null() {
        set_last_error("fx_rt_reset: rt is NULL");
        return FX_ERR_NULL_ARG;
    }
    let handle = unsafe { &mut *rt };
    match catch_unwind(AssertUnwindSafe(|| handle.rt.reset())) {
        Ok(()) => FX_OK,
        Err(_) => {
            set_last_error("fx_rt_reset: panic during reset");
            FX_ERR_PANIC
        }
    }
}

/// Number of addressable filter (SOS) nodes in the executor, depth-first — the valid `node` range
/// for [`fx_rt_set_coeffs`]. Returns the count (≥ 0) or a negative `FX_ERR_*`.
///
/// # Safety
/// `rt` must be NULL or a live handle from [`fx_rt_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fx_rt_filter_count(rt: *const FxRtGraph) -> c_int {
    clear_last_error();
    if rt.is_null() {
        set_last_error("fx_rt_filter_count: rt is NULL");
        return FX_ERR_NULL_ARG;
    }
    let handle = unsafe { &*rt };
    match catch_unwind(AssertUnwindSafe(|| handle.rt.filter_count() as c_int)) {
        Ok(n) => n,
        Err(_) => {
            set_last_error("fx_rt_filter_count: panic");
            FX_ERR_PANIC
        }
    }
}

/// Free a realtime handle returned by [`fx_rt_new`]. NULL is a no-op; double-free is undefined.
///
/// # Safety
/// `rt` must be NULL or a pointer previously returned by [`fx_rt_new`] and not yet freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fx_rt_free(rt: *mut FxRtGraph) {
    clear_last_error(); // a successful free must not leave a stale prior error visible
    if !rt.is_null()
        && catch_unwind(AssertUnwindSafe(|| drop(unsafe { Box::from_raw(rt) }))).is_err()
    {
        set_last_error("fx_rt_free: panic during drop");
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

    // --- realtime surface ---------------------------------------------------------------------

    /// Boxed graph handle for driving the ABI without a `.fxg` file on disk.
    fn graph_handle(graph: Graph) -> *mut FxGraph {
        Box::into_raw(Box::new(FxGraph { graph }))
    }

    #[test]
    fn rt_streaming_matches_batch() {
        // lowpass | gain streamed in blocks through fx_rt_process must match the one-shot batch
        // path (fx_process) — the SosStream chunk-invariance, observed through the C ABI.
        let g = Graph::op(OpKind::Lowpass, [1_000.0, 4.0]) | Graph::op(OpKind::Gain, [0.5]);
        let gh = graph_handle(g);

        let (mut verdict, mut margin) = (-1 as c_int, f32::NAN);
        let rt = unsafe { fx_rt_new(gh, 48_000, 256, &mut verdict, &mut margin) };
        assert!(!rt.is_null(), "fx_rt_new returned NULL");
        assert_eq!(verdict, FX_VERDICT_CERTIFIED_STABLE);
        assert!(margin > 0.0);
        assert_eq!(unsafe { fx_rt_filter_count(rt) }, 1);

        let x: Vec<f32> = (0..1000)
            .map(|i| (0.05 * i as f32).sin() + 0.3 * (0.27 * i as f32).sin())
            .collect();
        let mut streamed = vec![0.0f32; x.len()];
        for (xc, yc) in x.chunks(128).zip(streamed.chunks_mut(128)) {
            let st = unsafe { fx_rt_process(rt, xc.as_ptr(), yc.as_mut_ptr(), xc.len()) };
            assert_eq!(st, FX_OK);
        }

        let mut batch = x.clone(); // mono: interleaved == planar
        let st = unsafe { fx_process(gh, batch.as_mut_ptr(), batch.len(), 1, 48_000) };
        assert_eq!(st, FX_OK);
        for (i, (a, b)) in streamed.iter().zip(&batch).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "sample {i}: streamed {a} vs batch {b}"
            );
        }

        unsafe { fx_rt_free(rt) };
        unsafe { fx_graph_free(gh) };
    }

    #[test]
    fn rt_new_rejects_unstable_and_reports_verdict() {
        // A raw biquad with poles outside the unit circle (a2 = 1.5 ⇒ radius √1.5 > 1).
        let gh = graph_handle(Graph::op(OpKind::Biquad, [1.0, 0.0, 0.0, 0.0, 1.5]));
        let (mut verdict, mut margin) = (-1 as c_int, f32::NAN);
        let rt = unsafe { fx_rt_new(gh, 48_000, 256, &mut verdict, &mut margin) };
        assert!(rt.is_null(), "an unstable graph must be refused");
        assert_eq!(verdict, FX_VERDICT_UNSTABLE);
        assert!(margin < 0.0);
        assert!(!fx_last_error().is_null());
        unsafe { fx_graph_free(gh) };
    }

    #[test]
    fn rt_new_rejects_non_realtime_op() {
        // Normalize needs the whole signal's peak — no realtime lowering.
        let gh = graph_handle(Graph::op(OpKind::Normalize, [1.0]));
        let rt = unsafe { fx_rt_new(gh, 48_000, 256, ptr::null_mut(), ptr::null_mut()) };
        assert!(rt.is_null());
        assert!(!fx_last_error().is_null());
        unsafe { fx_graph_free(gh) };
    }

    #[test]
    fn rt_process_guards() {
        assert_eq!(
            unsafe { fx_rt_process(ptr::null_mut(), ptr::null(), ptr::null_mut(), 8) },
            FX_ERR_NULL_ARG
        );

        let gh = graph_handle(Graph::op(OpKind::Gain, [1.0]));
        let rt = unsafe { fx_rt_new(gh, 48_000, 64, ptr::null_mut(), ptr::null_mut()) };
        assert!(!rt.is_null());

        let x = [0.0f32; 128];
        let mut y = [0.0f32; 128];
        // A block larger than max_block is rejected, not clipped.
        assert_eq!(
            unsafe { fx_rt_process(rt, x.as_ptr(), y.as_mut_ptr(), 128) },
            FX_ERR_INVALID_ARG
        );
        // In-place (overlapping) processing would alias &[f32] with &mut [f32] — rejected.
        assert_eq!(
            unsafe { fx_rt_process(rt, y.as_ptr(), y.as_mut_ptr(), 64) },
            FX_ERR_INVALID_ARG
        );
        // len == 0 is a no-op success.
        assert_eq!(
            unsafe { fx_rt_process(rt, x.as_ptr(), y.as_mut_ptr(), 0) },
            FX_OK
        );

        unsafe { fx_rt_free(rt) };
        unsafe { fx_graph_free(gh) };
    }

    #[test]
    fn rt_set_coeffs_swaps_and_guards() {
        // Start from a low-pass (passes DC), swap node 0 to a designed high-pass: DC must die out.
        let gh = graph_handle(Graph::op(OpKind::Lowpass, [1_000.0, 4.0]));
        let rt = unsafe { fx_rt_new(gh, 48_000, 256, ptr::null_mut(), ptr::null_mut()) };
        assert!(!rt.is_null());

        let hp = fluxion_ops::butterworth_highpass(4, 1_000.0, 48_000);
        let flat: Vec<f32> = hp
            .iter()
            .flat_map(|b| [b.b0, b.b1, b.b2, b.a1, b.a2])
            .collect();
        assert_eq!(
            unsafe { fx_rt_set_coeffs(rt, 0, flat.as_ptr(), hp.len(), 0) },
            FX_OK
        );

        let dc = [1.0f32; 256];
        let mut out = [0.0f32; 256];
        for _ in 0..16 {
            assert_eq!(
                unsafe { fx_rt_process(rt, dc.as_ptr(), out.as_mut_ptr(), 256) },
                FX_OK
            );
        }
        let tail = out.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(
            tail < 1e-3,
            "DC through a swapped-in high-pass should decay, got {tail}"
        );

        // Out-of-range node and unstable sections are refused with a message.
        assert_eq!(
            unsafe { fx_rt_set_coeffs(rt, 7, flat.as_ptr(), hp.len(), 0) },
            FX_ERR_INVALID_ARG
        );
        let unstable = [1.0f32, 0.0, 0.0, 0.0, 1.5];
        assert_eq!(
            unsafe { fx_rt_set_coeffs(rt, 0, unstable.as_ptr(), 1, 0) },
            FX_ERR_INVALID_ARG
        );
        assert!(!fx_last_error().is_null());

        // Reset clears state (a fresh DC step behaves like a fresh executor).
        assert_eq!(unsafe { fx_rt_reset(rt) }, FX_OK);

        unsafe { fx_rt_free(rt) };
        unsafe { fx_graph_free(gh) };
    }
}

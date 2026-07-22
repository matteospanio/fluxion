/* Fluxion C ABI. Generated from crates/fluxion-ffi by cbindgen — do not edit by hand. */

#ifndef FLUXION_H
#define FLUXION_H

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * Success.
 */
#define FX_OK 0

/**
 * A required pointer argument was NULL.
 */
#define FX_ERR_NULL_ARG -1

/**
 * An argument was structurally invalid (e.g. `channels == 0`, or `frames * channels` overflows).
 */
#define FX_ERR_INVALID_ARG -2

/**
 * A panic was caught at the FFI boundary and stopped from unwinding into C.
 */
#define FX_ERR_PANIC -3

/**
 * Stability verdict code (see `fx_rt_new`): strictly inside the stable region.
 */
#define FX_VERDICT_CERTIFIED_STABLE 0

/**
 * Stability verdict code: on / within f32-tolerance of the stability boundary.
 */
#define FX_VERDICT_MARGINALLY_STABLE 1

/**
 * Stability verdict code: could not be evaluated (non-finite coefficients).
 */
#define FX_VERDICT_INDETERMINATE 2

/**
 * Stability verdict code: no certificate is available for this construct.
 */
#define FX_VERDICT_NOT_CERTIFIED 3

/**
 * Stability verdict code: provably outside the stable region (refused by `fx_rt_new`).
 */
#define FX_VERDICT_UNSTABLE 4

/**
 * An owned, reified DSP graph. Opaque to C: construct with [`fx_graph_load_fxg`], pass by pointer,
 * release with [`fx_graph_free`].
 */
typedef struct FxGraph FxGraph;

/**
 * An owned realtime executor for one mono stream. Opaque to C: construct with [`fx_rt_new`], pass
 * by pointer, release with [`fx_rt_free`].
 */
typedef struct FxRtGraph FxRtGraph;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * Return this thread's most recent error message as a NUL-terminated C string, or NULL if the last
 * fluxion call on this thread succeeded.
 *
 * The pointer is owned by fluxion and valid only until the **next** fluxion call on this thread;
 * copy the string if you need to keep it. Never call `free` on it.
 *
 * # Safety
 * None to uphold on the caller side, but the returned pointer must be treated as borrowed (see above).
 */
const char *fx_last_error(void);

/**
 * Load a reified graph from a `.fxg` file. Returns an owning handle, or NULL on error (see
 * [`fx_last_error`]). Free the handle with [`fx_graph_free`].
 *
 * # Safety
 * `path` must be either NULL or a valid pointer to a NUL-terminated C string.
 */
FxGraph *fx_graph_load_fxg(const char *path);

/**
 * Free a graph handle returned by [`fx_graph_load_fxg`]. NULL is a no-op; double-free is undefined.
 *
 * # Safety
 * `graph` must be NULL or a pointer previously returned by [`fx_graph_load_fxg`] and not yet freed.
 */
void fx_graph_free(FxGraph *graph);

/**
 * Process an interleaved audio buffer in place through `graph`.
 *
 * **Contract.** `data` points to `frames * channels` `f32`s in **interleaved (frame-major)** order —
 * `[c0f0, c1f0, …, c0f1, c1f1, …]`. `fs` is the sample rate in Hz. The graph runs on the CPU batch
 * engine; the result (same shape — the op set is length- and channel-preserving) is written back
 * over `data`. Returns [`FX_OK`] or a negative `FX_ERR_*` (see [`fx_last_error`] for a message).
 *
 * This is the offline/batch path, not the realtime engine — it allocates; do not call it from an
 * audio callback.
 *
 * # Safety
 * `graph` must be a live handle from [`fx_graph_load_fxg`]. If `frames * channels > 0`, `data` must
 * point to at least that many writable, initialized `f32`s.
 */
int fx_process(const FxGraph *graph,
               float *data,
               uintptr_t frames,
               uintptr_t channels,
               uint32_t fs);

/**
 * ABI smoke test: builds a trivial `gain | gain` graph and returns its leaf count (`2`), or
 * [`FX_ERR_PANIC`] if the graph construction panicked.
 *
 * Exists so the C ABI links and round-trips with no arguments or state.
 *
 * # Safety
 * None — takes no pointers and has no preconditions.
 */
int fx_abi_smoke(void);

/**
 * Lower `graph` to a realtime executor at sample rate `fs`, sized for blocks up to `max_block`
 * samples. Returns an owning handle, or NULL on error (see [`fx_last_error`]); free with
 * [`fx_rt_free`].
 *
 * The graph's coefficients are designed once at `fs` and certified: if non-NULL, `verdict_out`
 * receives the `FX_VERDICT_*` code and `margin_out` the stability margin (`1 − spectral radius`;
 * NaN if indeterminate) — written whenever certification ran, including when construction is then
 * refused. An `FX_VERDICT_UNSTABLE` graph is refused. Graphs containing an op with no realtime
 * lowering (normalize / fade / reverse / the modulated effects / feedback) are refused too.
 *
 * All allocation happens here; [`fx_rt_process`] is allocation-free afterwards.
 *
 * # Safety
 * `graph` must be NULL or a live handle from [`fx_graph_load_fxg`]. `verdict_out` and `margin_out`
 * must each be NULL or valid for a write.
 */
FxRtGraph *fx_rt_new(const FxGraph *graph,
                     uint32_t fs,
                     uintptr_t max_block,
                     int *verdict_out,
                     float *margin_out);

/**
 * Filter one mono block: read `len` `f32`s from `input`, write `len` filtered `f32`s to `output`.
 * `len` must be ≤ the `max_block` given to [`fx_rt_new`]; the two buffers must **not** overlap
 * (in-place processing is rejected as [`FX_ERR_INVALID_ARG`]). Filter state is carried across
 * calls — streaming a signal in chunks equals filtering it whole. Returns [`FX_OK`] or a negative
 * `FX_ERR_*`.
 *
 * Allocation-free and lock-free: safe to call from an audio callback. (The one caveat: entering
 * any `fx_*` call clears a *stale* error message left by a previous failed call on the same
 * thread, which frees that string — in a callback loop that isn't erroring, this is a no-op.)
 *
 * # Safety
 * `rt` must be NULL or a live handle from [`fx_rt_new`], driven by one thread at a time. If
 * `len > 0`, `input` must point to `len` readable `f32`s and `output` to `len` writable `f32`s.
 */
int fx_rt_process(FxRtGraph *rt, const float *input, float *output, uintptr_t len);

/**
 * Swap the `node`-th filter's coefficients (depth-first order over the graph's filter nodes, see
 * [`fx_rt_filter_count`]) to a new cascade of `n_sections` second-order sections, equal-power
 * crossfaded over `fade_samples` (0 = swap at the next sample). `sections` points to
 * `n_sections * 5` `f32`s, `[b0, b1, b2, a1, a2]` per section (`a0` normalised to 1).
 *
 * The incoming cascade is certified first; unstable sections are refused. Returns [`FX_OK`] or a
 * negative `FX_ERR_*` (out-of-range `node` is [`FX_ERR_INVALID_ARG`]).
 *
 * This is a **control-plane** call: it allocates (and may reallocate the incoming-coefficient
 * storage when the section count changes), so call it from a control thread between
 * [`fx_rt_process`] calls — not from the audio callback.
 *
 * # Safety
 * `rt` must be NULL or a live handle from [`fx_rt_new`], driven by one thread at a time.
 * `sections` must be NULL or point to `n_sections * 5` readable `f32`s.
 */
int fx_rt_set_coeffs(FxRtGraph *rt,
                     uintptr_t node,
                     const float *sections,
                     uintptr_t n_sections,
                     uint32_t fade_samples);

/**
 * Reset all filter state and delay lines to silence (as if freshly built) — call on seek or
 * source switch. Returns [`FX_OK`] or a negative `FX_ERR_*`.
 *
 * # Safety
 * `rt` must be NULL or a live handle from [`fx_rt_new`], driven by one thread at a time.
 */
int fx_rt_reset(FxRtGraph *rt);

/**
 * Number of addressable filter (SOS) nodes in the executor, depth-first — the valid `node` range
 * for [`fx_rt_set_coeffs`]. Returns the count (≥ 0) or a negative `FX_ERR_*`.
 *
 * # Safety
 * `rt` must be NULL or a live handle from [`fx_rt_new`].
 */
int fx_rt_filter_count(const FxRtGraph *rt);

/**
 * Free a realtime handle returned by [`fx_rt_new`]. NULL is a no-op; double-free is undefined.
 *
 * # Safety
 * `rt` must be NULL or a pointer previously returned by [`fx_rt_new`] and not yet freed.
 */
void fx_rt_free(FxRtGraph *rt);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  /* FLUXION_H */

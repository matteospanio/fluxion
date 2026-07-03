/* Fluxion C ABI. Generated from crates/fluxion-ffi by cbindgen — do not edit by hand. */

#ifndef FLUXION_H
#define FLUXION_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/** Success. */
#define FX_OK 0
/** A required pointer argument was NULL. */
#define FX_ERR_NULL_ARG -1
/** An argument was structurally invalid (e.g. channels == 0, or frames * channels overflows). */
#define FX_ERR_INVALID_ARG -2
/** A panic was caught at the FFI boundary and stopped from unwinding into C. */
#define FX_ERR_PANIC -3

/**
 * An owned, reified DSP graph. Opaque to C: construct with fx_graph_load_fxg, pass by pointer,
 * release with fx_graph_free.
 */
typedef struct FxGraph FxGraph;

/**
 * Return this thread's most recent error message as a NUL-terminated C string, or NULL if the last
 * fluxion call on this thread succeeded.
 *
 * The pointer is owned by fluxion and valid only until the next fluxion call on this thread; copy
 * the string if you need to keep it. Never call free() on it.
 */
const char *fx_last_error(void);

/**
 * Load a reified graph from a `.fxg` file. Returns an owning handle, or NULL on error (see
 * fx_last_error). Free the handle with fx_graph_free. `path` must be NULL or a NUL-terminated string.
 */
FxGraph *fx_graph_load_fxg(const char *path);

/**
 * Free a graph handle returned by fx_graph_load_fxg. NULL is a no-op; double-free is undefined.
 */
void fx_graph_free(FxGraph *graph);

/**
 * Process an interleaved audio buffer in place through `graph`.
 *
 * `data` points to frames * channels floats in interleaved (frame-major) order
 * [c0f0, c1f0, ..., c0f1, c1f1, ...]; `fs` is the sample rate in Hz. The result (same shape) is
 * written back over `data`. Returns FX_OK or a negative FX_ERR_* (see fx_last_error). This is the
 * offline/batch path — it allocates; do not call it from an audio callback.
 */
int fx_process(const FxGraph *graph,
               float *data,
               uintptr_t frames,
               uintptr_t channels,
               uint32_t fs);

/**
 * ABI smoke test: builds a trivial `gain | gain` graph and returns its leaf count (2). Exists so
 * the C ABI links and round-trips with no arguments or state.
 */
int fx_abi_smoke(void);

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#endif /* FLUXION_H */

/* Minimal C smoke test for the Fluxion C ABI (plan tasks K1-K3).
 *
 * Proves the staticlib links and that the boundary reports errors as a status + message instead of
 * crashing / unwinding into C. Build against the staticlib, e.g.:
 *
 *   cargo build -p fluxion-ffi
 *   cc examples/smoke.c -I crates/fluxion-ffi/include \
 *      -L target/debug -lfluxion_ffi -lpthread -lm -ldl -o smoke && ./smoke
 */

#include <stddef.h>
#include <stdio.h>

#include "fluxion.h"

int main(void) {
    /* Links + round-trips with no state. */
    if (fx_abi_smoke() != 2) {
        fprintf(stderr, "fx_abi_smoke: expected 2\n");
        return 1;
    }

    /* A NULL graph/data must be reported (not crash) and leave a last-error message. */
    int st = fx_process(NULL, NULL, 4, 2, 48000);
    if (st != FX_ERR_NULL_ARG) {
        fprintf(stderr, "fx_process(NULL): expected FX_ERR_NULL_ARG (%d), got %d\n",
                FX_ERR_NULL_ARG, st);
        return 1;
    }
    /* The error pointer is only valid until the NEXT fluxion call on this thread — use it now. */
    const char *err = fx_last_error();
    if (err == NULL) {
        fprintf(stderr, "fx_last_error: expected a message after an error\n");
        return 1;
    }
    printf("fx_abi_smoke()=2, fx_process(NULL)=%d (\"%s\")\n", st, err);

    /* The realtime surface links and guards its arguments the same way. */
    FxRtGraph *rt = fx_rt_new(NULL, 48000, 256, NULL, NULL);
    if (rt != NULL || fx_last_error() == NULL) {
        fprintf(stderr, "fx_rt_new(NULL): expected NULL and a last-error message\n");
        return 1;
    }
    printf("fx_rt_new(NULL)=NULL (\"%s\")\n", fx_last_error());
    int rst = fx_rt_process(NULL, NULL, NULL, 8);
    if (rst != FX_ERR_NULL_ARG) {
        fprintf(stderr, "fx_rt_process(NULL): expected FX_ERR_NULL_ARG (%d), got %d\n",
                FX_ERR_NULL_ARG, rst);
        return 1;
    }
    printf("fx_rt_process(NULL)=%d (\"%s\")\n", rst, fx_last_error());
    fx_rt_free(NULL); /* NULL free is a safe no-op */

    printf("fluxion C ABI smoke OK\n");
    return 0;
}

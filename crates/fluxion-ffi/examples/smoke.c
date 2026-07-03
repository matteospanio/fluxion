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
    const char *err = fx_last_error();
    if (err == NULL) {
        fprintf(stderr, "fx_last_error: expected a message after an error\n");
        return 1;
    }

    printf("fluxion C ABI smoke OK: fx_abi_smoke()=2, fx_process(NULL)=%d (\"%s\")\n", st, err);
    return 0;
}

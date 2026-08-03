/* The C quickstart. CI compiles and runs this on Linux, macOS and Windows.
 *
 * Ten code lines is the budget (docs/interfaces.md); comments and blanks do not count. Build it
 * against the static library:
 *
 *   cargo build -p fluxion-ffi
 *   cc examples/quickstart.c -I include target/debug/libfluxion_ffi.a -lpthread -lm -ldl -o qs
 */
#include <stdio.h>
#include "fluxion.h"

int main(void) {
    float block[480] = {0};
    char text[128];
    FxGraph *graph = fx_chain_from_text("highpass(80, 4) | gain(-3dB)");
    if (!graph) { fprintf(stderr, "%s\n", fx_last_error()); return 1; }
    fx_graph_to_text(graph, text, sizeof text);
    if (fx_process(graph, block, 480, 1, 48000) != FX_OK) { return 1; }
    printf("ok: %s\n", text);
    fx_graph_free(graph);
}

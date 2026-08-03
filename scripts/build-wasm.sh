#!/bin/sh
# Build the browser bindings: compile crates/fluxion-wasm to wasm32, then run wasm-bindgen to emit
# the JavaScript glue and the .d.ts alongside it.
#
#   ./scripts/build-wasm.sh            release (what ships)
#   ./scripts/build-wasm.sh --debug    faster to build, much larger
#
# Output lands in crates/fluxion-wasm/js/pkg/, which the package in crates/fluxion-wasm/js/ wraps.
#
# Requirements: the wasm32-unknown-unknown target (rust-toolchain.toml installs it) and
# wasm-bindgen-cli at the same version as the wasm-bindgen crate — they are a matched pair and a
# mismatch fails with a confusing schema error:
#
#   cargo install --locked wasm-bindgen-cli@$(scripts/build-wasm.sh --print-version)
#
# `--target web` produces one artifact that works in a browser and in Node 18+ (Node reads the
# .wasm with readFileSync and hands the bytes to init), so there is a single thing to publish.
set -e

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
out="$root/crates/fluxion-wasm/js/pkg"

# The wasm-bindgen crate version the lockfile resolved — the CLI must match it exactly.
wb_version() {
    cargo tree -p fluxion-wasm -e normal --depth 1 --manifest-path "$root/Cargo.toml" 2>/dev/null |
        sed -n 's/.*wasm-bindgen v\([0-9.]*\).*/\1/p' | head -1
}

if [ "$1" = "--print-version" ]; then
    wb_version
    exit 0
fi

profile=release
profile_flag=--release
if [ "$1" = "--debug" ]; then
    profile=debug
    profile_flag=
fi

echo "building fluxion-wasm ($profile)"
cargo build -p fluxion-wasm --target wasm32-unknown-unknown $profile_flag --manifest-path "$root/Cargo.toml"

wasm="$root/target/wasm32-unknown-unknown/$profile/fluxion_wasm.wasm"
[ -f "$wasm" ] || { echo "no wasm at $wasm" >&2; exit 1; }

echo "running wasm-bindgen -> $out"
wasm-bindgen "$wasm" --out-dir "$out" --target web --omit-default-module-path

# A second binding of the *same* .wasm as a classic script. An AudioWorklet has no module loader
# and no fetch, so the ES-module glue above cannot be used inside one; `no-modules` produces a
# script the page can read as text and hand to `addModule` alongside the processor.
wasm-bindgen "$wasm" --out-dir "$out/no-modules" --target no-modules --out-name fluxion_wasm

ls -l "$out"

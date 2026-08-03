#!/usr/bin/env bash
# Set the version across every package in the repository.
#
#   scripts/set-version.sh 0.2.0     set it
#   scripts/set-version.sh --check   verify every file already agrees (used by CI)
#
# The version lives in more places than it looks like it does, and they must agree exactly or
# `cargo publish` refuses the workspace:
#
#   Cargo.toml                        [workspace.package] version — every crate inherits it
#   crates/*/Cargo.toml               intra-workspace path deps carry `version = "…"` too, because
#                                     a published crate cannot depend on a bare path
#   crates/fluxion-py/Cargo.toml      its own workspace, so it does not inherit
#   crates/fluxion-wasm/js/package.json   the npm package
#   Cargo.lock, crates/fluxion-py/Cargo.lock   refreshed from the above
#
# Not touched, on purpose:
#   crates/fluxion-py/pyproject.toml  `dynamic = ["version"]` — maturin reads Cargo.toml
#   CHANGELOG.md                      prose; this script reminds you rather than writing it
#   git tag                           releasing is a decision, not a side effect of an edit
set -euo pipefail

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"

usage() {
    sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//; $d'
    exit "${1:-0}"
}

# In-place edit that works with both GNU and BSD sed (`sed -i` differs between them).
edit() {
    local file=$1 script=$2
    sed "$script" "$file" >"$file.tmp"
    mv "$file.tmp" "$file"
}

current() {
    sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1
}

# Every file that names the version, and how to read it out — the single list this script and its
# --check mode both work from, so a new package cannot be half-added.
report() {
    printf '  %-42s %s\n' "Cargo.toml [workspace.package]" "$(current)"
    printf '  %-42s %s\n' "crates/fluxion-py/Cargo.toml" \
        "$(sed -n 's/^version = "\(.*\)"/\1/p' crates/fluxion-py/Cargo.toml | head -1)"
    printf '  %-42s %s\n' "crates/fluxion-wasm/js/package.json" \
        "$(sed -n 's/^  "version": "\(.*\)",/\1/p' crates/fluxion-wasm/js/package.json | head -1)"
    printf '  %-42s %s\n' "intra-workspace path deps" \
        "$(grep -ho 'path = "\.\./fluxion-[a-z]*", version = "[^"]*"' crates/*/Cargo.toml |
            sed 's/.*version = "\(.*\)"/\1/' | sort -u | tr '\n' ' ')"
}

# Fail if any file disagrees with `Cargo.toml`. This is what CI runs.
verify() {
    local want=$1 bad=0
    while read -r found; do
        [ "$found" = "$want" ] || { echo "path dep at $found, expected $want" >&2; bad=1; }
    done < <(grep -ho 'path = "\.\./fluxion-[a-z]*", version = "[^"]*"' crates/*/Cargo.toml |
        sed 's/.*version = "\(.*\)"/\1/' | sort -u)

    local py npm
    py=$(sed -n 's/^version = "\(.*\)"/\1/p' crates/fluxion-py/Cargo.toml | head -1)
    npm=$(sed -n 's/^  "version": "\(.*\)",/\1/p' crates/fluxion-wasm/js/package.json | head -1)
    [ "$py" = "$want" ] || { echo "crates/fluxion-py/Cargo.toml is $py, expected $want" >&2; bad=1; }
    [ "$npm" = "$want" ] || { echo "js/package.json is $npm, expected $want" >&2; bad=1; }
    return $bad
}

case "${1:---help}" in
--help | -h) usage ;;
--check)
    want=$(current)
    echo "checking every package is at $want"
    report
    if verify "$want"; then
        echo "ok: all packages agree"
    else
        echo >&2
        echo "Run scripts/set-version.sh $want to bring them into line." >&2
        exit 1
    fi
    exit 0
    ;;
esac

new=$1
# X.Y.Z, optionally with a pre-release tag (`0.2.0-rc.1`). No `+build`: PEP 440 treats a local
# version specially and PyPI refuses to publish one, so a wheel built from it could not be released.
if ! printf '%s' "$new" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$'; then
    echo "not a version: '$new' (expected X.Y.Z or X.Y.Z-pre)" >&2
    exit 1
fi

old=$(current)
if [ "$old" = "$new" ]; then
    echo "already at $new"
else
    echo "$old -> $new"

    # The workspace version every member crate inherits. Anchored to the first `version = ` line,
    # which is inside [workspace.package]; dependency versions are never at column 0.
    edit Cargo.toml "0,/^version = \"$old\"/s//version = \"$new\"/"

    # Path dependencies between our own crates. Matched together with the `path = "../fluxion-…"`
    # so a third-party dependency that happens to be at the same version is left alone.
    for manifest in crates/*/Cargo.toml; do
        edit "$manifest" \
            "s|\(path = \"\.\./fluxion-[a-z]*\", version = \)\"$old\"|\1\"$new\"|g"
    done

    # fluxion-py is its own workspace (PyO3's extension-module linking breaks `cargo test
    # --workspace`), so it does not inherit and needs setting directly.
    edit crates/fluxion-py/Cargo.toml "0,/^version = \"$old\"/s//version = \"$new\"/"

    # The npm package.
    edit crates/fluxion-wasm/js/package.json \
        "0,/^  \"version\": \"$old\",/s//  \"version\": \"$new\",/"
fi

echo "refreshing lockfiles"
cargo update --workspace --quiet
(cd crates/fluxion-py && cargo update --workspace --quiet)

echo
report
echo
if ! verify "$new"; then
    echo "something did not update — fix it before committing" >&2
    exit 1
fi

leftover=$(grep -rl "\"$old\"" --include=Cargo.toml --include=package.json . 2>/dev/null |
    grep -v node_modules || true)
if [ -n "$leftover" ]; then
    echo "note: '$old' still appears in:" >&2
    echo "$leftover" | sed 's/^/  /' >&2
    echo "  (fine if it is an unrelated dependency's version)" >&2
    echo
fi

echo "set to $new. Still to do by hand:"
if ! grep -q "^## \[$new\]" CHANGELOG.md 2>/dev/null; then
    echo "  - CHANGELOG.md: move [Unreleased] under '## [$new] - $(date +%F)' and add the link ref"
fi
echo "  - commit, then tag: git tag v$new"
echo "  - the tag triggers .github/workflows/wheels.yml (PyPI); crates.io is still manual"

# AGENTS.md

Guidance for AI agents (and humans) working in this repository. Keep it short — the full design
lives in [PROJECT.md](PROJECT.md), the crate layout in [README.md](README.md).

## What this is

**Fluxion** — a differentiable, cross-vendor, framework-agnostic audio DSP library with a functional
graph API, a SoX-replacement CLI, and a hard-real-time engine. Rust core, bound to anything
(Python / C / JS / WASM). **Status: scaffold.** `PROJECT.md` is the source of truth for design and
roadmap; only `fluxion-core` has real logic today (the graph algebra).

## Commands

```bash
cargo build                  # whole workspace (builds offline — no network deps yet)
cargo test                   # unit tests + doctests
cargo clippy --all-targets   # lint (workspace lint: clippy::all = warn)
cargo fmt                    # format
cargo run -p fluxion-cli     # the CLI (binary name: fluxion)
cargo doc --no-deps          # API docs
```

## Workspace

Cargo workspace, members under `crates/*`, shared metadata via `[workspace.package]`, edition 2024,
MSRV 1.85. Package `fluxion` (dir `crates/fluxion-facade`) is the facade users depend on;
`fluxion-cli` builds the binary named `fluxion`. The other crates are placeholders — see the table
in `README.md`.

## Conventions — don't break these

- **Operator algebra:** `|` = series, `+` = parallel (outputs summed), matching the `Graph` IR in
  `fluxion-core` (and torchfx semantics).
- **Two engines, never one.** The differentiable/GPU **batch** path and the **real-time** CPU path
  are separate. Never run autograd or GPU dispatch inside an audio callback (no alloc, no locks,
  bounded time). PROJECT.md §5.
- **Own the backward, rent the graph.** Each DSP op owns its analytic forward + backward (VJP);
  autograd is rented from Burn / torch / jax. Do not build a general autograd engine. PROJECT.md §2, §4.3.
- **Coefficients are computed in an explicit design/compile stage, not lazily inside `forward`** —
  the lazy idiom blocks AOT/scripting (the nnAudio2 lesson). PROJECT.md §3.
- **Heavy deps are wired per-crate, only when that crate is implemented.** The backbone builds
  offline; do not add `burn` / `cubecl` / `pyo3` / `clap` / `symphonia` to placeholder crates until
  you implement them. Keep placeholder `[dependencies]` empty.
- **Naming:** `Lo`/`Hi` prefix for low/high-pass variants (e.g. `LoButterworth`); base types
  unprefixed. Frequencies in Hz; sample rate is `fs`, never `sample_rate`.
- **Style:** keep `cargo clippy` and `cargo fmt` clean. Mark deliberate shortcuts with a
  `// ponytail:` comment that names the ceiling and the upgrade path.
- **Tests:** non-trivial logic leaves one runnable check (`#[test]` or a doctest); use exact
  asserts for the algebra.

## License

**MIT** (see [LICENSE](LICENSE)). Do **not** copy code from the GPL reference projects (AudioNoise,
SoX, sox-extended) — reuse ideas/math only (e.g. the RBJ cookbook). PROJECT.md §12.

## Etiquette

Commit or push only when asked. Update `PROJECT.md` when a design decision changes, and the crate
table in `README.md` when a crate changes state.

# Contributing to Fluxion

Thanks for helping build Fluxion. This page is the source of truth for the project's conventions;
the architecture is described in the [README](README.md) and the companion paper it cites.

## Getting started

```bash
cargo build                 # whole workspace (builds offline — pure-Rust backbone)
cargo test                  # unit tests + doctests
cargo clippy --all-targets  # lint (CI denies warnings)
cargo fmt                   # format
cargo bench --no-run        # compile the Criterion benches (fluxion-ops, fluxion-backend)
cargo run -p fluxion-cli    # the CLI (binary name: fluxion)
```

`cargo fmt --all --check` and `cargo clippy --workspace --all-targets -- -D warnings` must pass, and
public items need doc comments (`missing_docs` is warn-denied in CI).

## Conventions that reviewers will hold you to

- **Operator algebra:** `|` = series, `+` = parallel (outputs summed). Match the `Graph` IR in
  `fluxion-core`.
- **Two engines, never one.** The differentiable/GPU **batch** path and the **real-time** CPU path
  are separate. No autograd, GPU dispatch, allocation, or locks inside an audio callback.
- **Own the backward, rent the graph.** Each DSP op owns its analytic forward + backward (VJP);
  autograd is rented from Burn / torch / jax. Don't build a general autograd engine.
- **Coefficients are designed in an explicit compile stage, not lazily inside `forward`.**
- **Naming:** full-word low/high-pass names with the filter family where it matters (e.g.
  `Lowpass`, `cheby1_lowpass`); base types unprefixed. Frequencies in Hz; sample rate is `fs`,
  never `sample_rate`.
- **Minimal deps.** The backbone builds offline. Heavy/alpha deps (Burn, CubeCL, PyO3, clap,
  Symphonia) are wired per-crate behind features when that crate is implemented — don't add them to
  placeholder crates.
- **Leave a runnable check.** Non-trivial logic gets a `#[test]` or a doctest; use exact asserts for
  the algebra.
- **Deliberate shortcuts** get a `// ponytail:` comment naming the ceiling and the upgrade path.

## Commits & PRs

- Keep the tree formatted and clippy-clean. Add or update a test for behavior you change.
- Update the crate table in [README.md](README.md) when a crate changes state.
- Small, focused PRs. Fill in the pull-request template.

## Licensing

Fluxion is **[MIT](LICENSE)**. By contributing you agree your work is licensed under MIT.

**Do not copy code from the GPL reference projects** (AudioNoise, SoX, sox-extended). Reuse *ideas
and math only* — e.g. the RBJ cookbook formulas — never the source.

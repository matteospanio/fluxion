<!-- Thanks for contributing! Keep PRs small and focused. See CONTRIBUTING.md. -->

## What & why

Briefly: what this changes and the motivation. Link any related issue (`Closes #123`).

## Checklist

- [ ] `cargo fmt --all --check` and `cargo clippy --workspace --all-targets -- -D warnings` pass
- [ ] `cargo test` passes; new/changed behavior has a `#[test]` or doctest with exact asserts
- [ ] Public items are documented (`missing_docs` is denied in CI)
- [ ] Follows the [AGENTS.md](../AGENTS.md) conventions (operator algebra, two engines, own-the-backward, explicit coeff design, `fs`/Hz naming)
- [ ] No new heavy deps in the offline backbone (feature-gate per crate if needed)
- [ ] Updated `PROJECT.md` (design change) / `README.md` crate table (crate state change) if applicable
- [ ] No code copied from the GPL reference projects (ideas/math only)

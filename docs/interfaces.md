# The interface contract

Fluxion has five doors. This page is the agreement between them.

**The rule: an op ships on every interface, or this document says why not.**

That is not a slogan — it is checkable, and CI checks it. What follows is who each door is for,
what it must expose, and what a pull request has to do before it counts as finished.

## Who uses what

| Interface | Who it is for | What it must expose |
|---|---|---|
| Rust (`fluxion`) | engine and app developers | everything: chain building, offline run, realtime, freeze/load, analysis taps |
| CLI (`fluxion`) | terminal users, scripts, CI | file in → file out: every offline op as a verb, `stat` for measurements, chain syntax for combinations |
| Python (`pip install fluxion`) | data/ML people, notebooks, batch jobs | offline processing on arrays (NumPy/torch, zero-copy) and on `Wave`, every op as a class plus the chain API, measurements as dicts |
| C header (`fluxion.h`) | C/C++/Swift hosts, plugin shells | the stable core: build chain from text, run on buffers, freeze/load; small on purpose |
| JS/wasm (npm) | web apps | offline render + AudioWorklet playback, chain from the same text syntax, typed API |

## One name, and how ops reach each door

There is one canonical name per op — `lowpass`, `cheby1_lowpass`, `lowshelf` — declared once in the
`ops!` table in [`crates/fluxion-core/src/op.rs`](../crates/fluxion-core/src/op.rs). Every interface
derives its spelling from that row; none of them keeps its own list.

- **Rust** spells it as a function: `fluxion::prelude::lowpass`.
- **The CLI** spells it as a pipeline token: `fluxion in.wav out.wav lowpass --cutoff 800`.
- **Python** spells it as a class named after the Rust variant: `fluxion.filter.Lowpass`.
- **C and JS** do **not** get one symbol per op, deliberately. They build chains from
  [text](chain-syntax.md): `fx_chain_from_text("lowpass(800, 4) | gain(-3dB)")`,
  `Chain.fromText(...)`. The C header stays small on purpose, and a per-op ABI symbol is a
  commitment that never expires. JS additionally gets the catalog as *data* — `fluxion/ops`, with
  `OpName` as a TypeScript literal union — because a bundler can drop it if unused, which is not
  true of anything compiled into the wasm module. The coverage check for these two is
  `every_registry_op_parses_from_its_bare_name` in
  [`crates/fluxion-core/tests/parse_roundtrip.rs`](../crates/fluxion-core/tests/parse_roundtrip.rs):
  if an op is not reachable from its bare name, it is not on those interfaces, whatever
  [ops.md](ops.md) claims.

[ops.md](ops.md) is the generated cross-name table. Do not edit it; run
`python scripts/gen_interfaces.py`.

## The project rate

Ops never change the sample rate — that is what lets `|` and `+` compose — so a host picks one rate
and converts on the way in. Each door spells that one way (roadmap R2):

- **Rust** — `fluxion::transform::ensure_fs(signal, rate)`.
- **The CLI** — `fluxion --rate 48000 a.wav b.wav out.wav`.
- **Python** — `fx.Wave.from_file(path, fs=48_000)`, or `wave.ensure_fs(48_000)` later.
- **JS** — `ensureFs(samples, fromFs, toFs)`, for the gap between the file's rate and the
  `AudioContext`'s.

All four run the one converter in `fluxion_ops::resample`, whether the signal arrives whole or a
block at a time, so a file imported in the CLI and the same file streamed through the worklet come
out as the same samples. Input already at the rate is not touched at all. The frame count is exactly
`round(frames · to/from)` — a host computes that number for itself, so the converter has to land
on it rather than near it.

C is the exception, for the reason its whole surface is small: see "Not yet".

## Side inputs

Most chains carry one signal; a gate opened by a different microphone needs two. `side(0)` reads
the first extra signal handed to the chain and `<` says which signal drives a keyed op — one
grammar, so `gate(-35, 40) < side(0)` is the same string everywhere (roadmap S1). What differs is
only how the extra signal is handed over:

- **Rust** — `process_with(&graph, &input, &[&key])`.
- **The CLI** — `fluxion --side key.wav --chain "gate(-35, 40) < side(0)" in.wav out.wav`.
- **Python** — `chain(x, fs, sides=[key])`.
- **JS** — `chain.processWith(samples, fs, [key])`.

C is again the exception, and for the same reason: see "Not yet".

## Definition of done

A pull request that adds or changes an op is finished when all of these are true:

- [ ] The op is one row in the `ops!` table, with its doc comment (which becomes rustdoc, the
      Python docstring, and its entry in [ops.md](ops.md) — written once, used three times).
- [ ] `cargo test --workspace` passes, including the conformance tests that compare each interface
      against the registry.
- [ ] `python scripts/gen_interfaces.py` produces no diff.
- [ ] The op is reachable from Rust, the CLI, Python, and the chain text. If one of those is
      genuinely impossible, say so in "Not yet" below — with the reason, not just the gap.
- [ ] The check that would have caught the bug exists and was red before the fix
      ([CONTRIBUTING.md](../CONTRIBUTING.md): test first).
- [ ] `CHANGELOG.md` has an entry; a rename or a signature change is marked **Breaking**.

## Quickstarts

Every interface has a quickstart that CI runs, so none of them can rot:

| Interface | Source |
|---|---|
| Rust | a doctest in `crates/fluxion-facade/src/lib.rs` |
| CLI | `crates/fluxion-cli/tests/quickstart.sh` |
| Python | `crates/fluxion-py/tests/quickstart.py` |
| C | `crates/fluxion-ffi/examples/quickstart.c` |
| JS | `crates/fluxion-wasm/js/quickstart.mjs` |

**A quickstart may not exceed ten code lines.** A code line is one that is neither blank nor
starting with `//`, `#`, `/*`, `*` or `--`. The count is enforced in one place —
`code_lines()` in [`scripts/gen_interfaces.py`](../scripts/gen_interfaces.py) — so all five are
measured the same way.

If a quickstart grows past ten lines, **that is a bug in the API, not in the quickstart.** Fix the
API. This is the one rule in this document whose whole purpose is to push back on the library.

## Not yet

Honest gaps, with the reason. Each is a thing the rule above would otherwise require.

**The AudioWorklet half of the JS package.** `Chain` renders offline in the browser today — enough
for waveforms, previews and bounces, and checked against native in CI across all 27 ops and five
chain topologies (roadmap W6), 26 of 32 cases bit-identical. Live
playback needs the worklet, the lock-free ring and the no-allocation guarantee (roadmap W4), and
loading a frozen `.fxg` in the browser is W3. The offline `Chain` is the same class both extend, so
these are methods to add rather than a design to redo.

**The geometry stages** — `trim`, `pad`, `rate`, `speed`, `stretch`, `repeat`, `silence`,
`channels`, `remix`
— are CLI-only. They are not in the op registry because they *cannot* be: every graph op is
per-channel, length-preserving and rate-preserving, which is exactly what makes `|` and `+` compose
without bookkeeping, and every one of these changes the frame count, the sample rate or the channel
layout. They live in `fluxion_ops::transform` and run between graph passes. Giving them a home on
every interface is a design round of its own, not a missing row in a table.

**`ensure_fs` in C.** The other four doors have it; the C header does not. Its contract is the
stable core — build a chain from text, run it on a buffer — and a rate converter is not a chain op,
so it would be a new pair of symbols with its own buffer-sizing rules, and every ABI symbol is a
commitment that never expires. A C host that needs a project rate today can run the chain at each
source's own rate, or convert with whatever its platform already provides. If one asks, the shape to
add is `fx_ensure_fs(in, n, from_fs, to_fs, out, cap)` returning the frames written, the same
`snprintf` convention as `fx_graph_to_text`.

**Side inputs in C.** `fx_process` takes one buffer, and a second would mean a second entry point
with its own length and channel rules. The gap is the same shape as `ensure_fs` above and the answer
is the same: the C surface is the small stable core, and a host that needs a key today can run the
key chain itself and multiply. `fx_process_with(graph, buf, sides, n_sides, frames, channels, fs)`
is the signature to add if one asks.

**`.to(device)` in Python.** torchfx has it because its arrays are torch tensors. Fluxion's Python
API is an Array-API *consumer* over NumPy, and its GPU path is in the batch backend, not in the
array type — so there is no device to move a `Wave` to. Mirroring the method would be a lie about
what the object can do.

**Alias spellings.** There is no `fluxion.filter.LoButterworth`. Fluxion names ops by what they do
(`lowpass`) rather than by filter family, and that one name has to serve the CLI, the chain text,
the C ABI and the JS export. A second vocabulary for one interface is precisely the drift this
document exists to prevent.

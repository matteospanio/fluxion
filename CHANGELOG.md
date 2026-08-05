<!--
Release checklist (maintainers), before tagging vX.Y.Z:
  1. `cargo test --workspace` + `cargo test -p fluxion-autodiff --features burn` green.
  2. `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo deny check`.
  3. `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --lib`.
  4. Python wheel builds + `pytest` passes (CPU and, on a CUDA host, the GPU wheel).
  5. Move the `[Unreleased]` entries below under a new `## [X.Y.Z] - DATE` heading; bump the workspace
     and `fluxion-py` versions; update the link references at the bottom.
  6. Tag `vX.Y.Z`; publish crates.io in dependency order (needs `CARGO_REGISTRY_TOKEN`); PyPI publishes
     automatically from the tag via the wheels workflow (Trusted Publishing — register the GitHub
     `pypi` environment on PyPI once).
-->

# Changelog

All notable changes to fluxion are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); from 1.0.0 the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Render any region (Epic D / D4 — completes milestone F-M6)** — `render_region(graph, input,
  from, to)`, `chain.render_region` in Python, `chain.renderRegion` in the browser. The window is
  bit-identical to that window of a whole render, and not because it was tested into agreement: it
  *is* that window, since the chain runs from frame 0 and the rest is discarded. D4's check is
  taken literally — a chain of a high-pass, a parallel split, an echo and a compressor, rendered in
  ragged pieces in a scrambled order, compared with `assert_eq!` on `f32`.
  The cost is stated rather than hidden. Every op in front of a window carries state, so producing
  `[from, to)` costs `to` frames of work, not `to - from` — `frames_to_compute` returns exactly
  that so a caller can decide whether to ask. The cheap version is a state checkpoint, which is its
  own piece of work with its own cost model; this is the correctness floor it will have to match.
  Ops whose output depends on the whole signal are refused by name with what to do instead:
  `normalize` scales by the peak of what it was given, `loudnorm` measures before it changes,
  `reverse` needs to know where the end is, and the `limiter` looks ahead.

- **Automation: curves driving op parameters (Epic D / D2)** — `process_automated(graph, input,
  &automation)`, where an `Automation` is a side table of `Lane`s naming a node's `name:` label, a
  parameter *by name*, and a curve. A side table rather than part of the `Graph` on purpose: a
  graph describes a signal path and round-trips through the chain text, while automation describes
  a performance over a stretch of time. An automated render still prints, parses and freezes as
  the chain it is.
  Two application modes, because the ops genuinely differ. A **gain is a multiply**, so it is
  rendered against the curve per sample with nothing approximated — which is exactly what D2's
  check demands, and the test asserts `==` on every one of 48 000 frames. A **filter's cutoff is
  an input to a design**, so it is redesigned every 64 frames (1.33 ms) with the filter state
  carried across the change. Anything else is refused by name, with an error that lists the
  parameters the op does have; a mistyped lane fails before a sample is rendered rather than being
  silently ignored.
  D2's own check says "0 → −60 dB", and getting that right needed `Shape::geometric` — a fade
  drawn in decibels is *geometric* in amplitude, not linear. Half way through a 60 dB fade the
  level is −30 dB; a straight line in amplitude is at −6 dB there, which is a different fade
  entirely. `Curve::db_ramp` is the constructor for the one people mean.

### Fixed

- **Coefficient crossfades in the realtime engine were 3 dB loud (`fluxion-rt`)** — `SosStream`
  blended the outgoing and incoming cascades with an **equal-power** law, `cos·old + sin·new`. But
  the two branches are the *same input* through two similar filters, so their outputs are strongly
  correlated, and correlated signals add in amplitude: `cos + sin = √2` is **+3.01 dB** in the
  middle of every fade. The law is now linear, whose gains sum to 1.
  This is the same arithmetic that D1 turned up in the roadmap's own crossfade check, applied to a
  place it had been wrong since the streaming filter was written. The existing test could not see
  it — it swaps a low-pass for a high-pass on DC, where one branch is 1 and the other 0, and both
  laws slide cleanly between them. The new test crossfades a filter **to itself**, where the
  branches are identical and any law but linear shows immediately; it measures 3.01 dB against the
  old code and 0.0 against the new.
- `SosStream::set_coeffs_now` replaces coefficients while keeping the filter state, for a parameter
  that is moving smoothly rather than jumping. Crossfading two cascades cold-starts the incoming
  one, which is the wrong trade 750 times a second.

- **One curve, two engines (Epic S / S4, Epic D / D3)** — `core::automation::Curve`: breakpoint
  automation, an LFO and an ADSR are one type, because they are one thing seen three ways — a list
  of points and a rule for how time maps onto them (`Once`, `Loop`, `Sustain`). The LFO is not an
  approximation of a sine: a raised cosine over two half-cycles *is* one, exactly.
  S4 and D3 land together because the roadmap gives them the same check in the same words — "the
  same description gives identical curves in the batch and realtime engines" and "the same
  breakpoints give identical envelopes offline and in the realtime engine". Identical here means
  **bit-identical**, asserted with `==` on `f32` across five shapes and five ramps, because both
  sides call the same `segment()` in `fluxion-core` and a tolerance would only hide it if they
  stopped.
  That forced a change to `SmoothedValue`, and it is worth stating why. The obvious realtime ramp
  accumulates — `current += step`, one add per sample — and it drifts from the line it is meant to
  be: over a one-second ramp at 48 kHz it lands **6.45e-4** away from exact, with only **24 of
  48 000** samples bit-identical. Computing from the sample index instead costs one multiply and
  lands on the curve at every sample. The old arithmetic is kept in the test suite as the thing
  that would *not* have matched, so the claim stays measured rather than asserted. All 27 existing
  realtime tests pass unchanged.
  Curves are read at absolute frames rather than by accumulating phase, so an LFO an hour into a
  session is sample-for-sample the same cycle as the first one, and a render that starts in the
  middle sees what a render from the beginning saw — which is what D4 leans on.

- **Crossfade over concat (Epic D / D1)** — `transform::crossfade(&[a, b, ...], overlap_s, law)` and
  `fluxion --crossfade 0.05` on the CLI. `concat` butt-joins, which puts a step at the seam; this
  overlaps each adjacent pair and fades across it. Output length is exactly the sum of the frames
  less each seam's overlap, with the overlap clamped to what the two sides actually have — so
  joining a 10 ms clip with a 1 s overlap asked for gives 10 ms rather than failing.
  **The roadmap's own check for this task names the wrong law, and the code says so.** D1 asks that
  "an equal-power crossfade of a signal with itself leaves the level unchanged (±1e-6)". It does
  not: equal-power's gains are `cos(tπ/2)` and `sin(tπ/2)`, which *square* to 1, so on identical —
  fully correlated — material they sum to `√2` and the seam is **+3.01 dB**. The law that leaves
  correlated material untouched is **linear**, whose gains sum to 1 exactly. Both properties are
  now tested in the units they belong to: linear holds a constant signal to within 1e-6 across the
  seam, equal-power holds white noise's RMS to 0.35 dB where linear digs the classic -3.01 dB hole,
  and a third test pins both failure modes at 3.01 dB so the two laws cannot quietly be swapped
  back. Pick by what the material is, not by taste — `CrossfadeLaw`'s own docs carry the table.

- **Observer taps: analysis that reads the chain and never touches it (Epic A / A1, A2, A3 —
  completes milestone F-M5)** — `meter` and `spectrum(2048, 0.5)` sit in a chain like anything else
  and measure what flows past. Invisible to the audio is **structural**, not promised: a tap is a
  different kind of node from an op, the executor hands it the buffer to borrow, and the buffer that
  carries on is the one that arrived — there is no code path by which it could return anything else.
  The check compares a chain with six taps against the same chain without them, bit for bit.
  The spectrum is a Hann-windowed FFT averaged over frames, scaled so a partial reads its own
  amplitude rather than a number proportional to it — the part an analyser gets wrong quietly.
  Checked against an independently written SciPy rfft over four size/overlap combinations: worst
  disagreement 1.8e-7, which is f32 noise. The meter reports peak, RMS and the loudest 3 s window,
  the last straight from M1's BS.1770 code. All three in decibels, because a meter is a decibel
  instrument and a reading that mixes units is how a caller ends up drawing a linear number on a dB
  scale; silence reads `-inf` rather than 0, which on that scale would mean full scale.
  Readings come back in chain order, labelled by the nearest enclosing `name:`. Rust, Python and
  the browser can read them (`process_taps`, `chain.taps`, `chain.processTaps`); the CLI and C can
  build a tapped chain but have nowhere to put the numbers, and `docs/interfaces.md` says what each
  would need.

- **Side inputs, and the gate that proves they work (Epic S / S1, S3)** — a chain could only ever
  carry one signal, which is the one thing standing between fluxion and a ducker, a keyed gate or
  any other two-input effect. Two additions to the algebra: `side(0)` reads a second signal handed
  to the chain, and `<` says which signal drives a keyed op — `gate(-35, 40) < side(0)`. The same
  string on every interface, with only the delivery differing: `process_with` in Rust, `--side` on
  the CLI, `sides=[...]` in Python, `processWith` in the browser. C is deliberately left out and
  `docs/interfaces.md` says why and what the signature would be.
  The two halves are checked separately because they fail differently. Alignment: two identical
  tones, one inverted, summed through `id + side(0)` cancel to under 1e-6 — and the same test slips
  the side signal by a single frame to confirm it would have noticed (6.5 % of the amplitude left
  over at 1 kHz). Transparency: keying a chain of ops that do not read a key produces the identical
  samples, and every algebra test in the repo passes unchanged.
  `gate` is the first op to declare a key input. Below the threshold it drops the signal by exactly
  `range` dB — measured within 0.1 dB at 6, 20 and 60 — and `hold` rides over a dip so it does not
  chatter. A silent key shuts it on loud material; a loud key holds it open on quiet material. An
  unconnected `side(n)` is silence, so a keyed gate handed no key **closes**: a key that went
  missing should shut the gate rather than quietly fall back to opening it.

- **Envelope follower (Epic S / S2)** — `Follower`, one pole with separate attack and release, peak
  or RMS. The block under gates, duckers and meters; not an op, because what comes out is a control
  signal. Checked two ways, because the two halves need different oracles: with attack and release
  equal it is a plain one-pole and is compared sample-for-sample against SciPy's `lfilter` (worst
  disagreement 1.2e-6 across four cases), and the asymmetric case — which has no LTI reference at
  all — is checked against the closed-form curves, a step following `1 - a^n` and silence decaying
  as `r^n`, both within 1e-4. `CompandCoeffs::design` now takes its coefficient from the same
  place, so an attack time means one thing across the crate.

- **Realtime varispeed (Epic R / R5)** — `varispeed::Varispeed`: playback speed that moves while it
  is playing, for scrubbing and tape effects. Pull rather than push, because a callback needs
  exactly the block it asked for and how much input that takes is the part that varies —
  `process(input, output)` fills the output and reports what it swallowed. Anti-aliased up to the
  `max_speed` it was built for: a 15 kHz tone played at 4× comes out quiet rather than folded back
  down as a loud tone in the wrong place. Zero allocations across five seconds of scrubbing with
  the speed moved every block, at 1.1 % of real time (`Fast`); the widest kernel it can be asked
  for — `Hq` at 2× — is 3.6 %.
  The clicking check took two attempts and the first one was wrong: a speed change does *not* step
  the output, since the read position stays continuous either way, so the largest sample-to-sample
  jump does not move at all when the smoothing is removed and a test written on it passes whatever
  the code does. What a speed change puts in the signal is a **corner** — the position's slope
  jumps in one sample — so the check is on the second difference. Slamming the speed between 1.0
  and 1.6 every 20 blocks measures 0.00084 against a steady-speed baseline of 0.00089; with the
  20 ms ramp taken out it is 0.0076, eight and a half times the baseline.

- **One project rate, set once (Epic R / R2)** — `transform::ensure_fs(signal, rate)` and its
  spelling on each door: `fluxion --rate 48000` in the CLI, `fx.Wave.from_file(path, fs=48_000)` /
  `wave.ensure_fs(48_000)` in Python, `ensureFs(samples, fromFs, toFs)` in the browser. A signal
  already at the rate is handed straight back, not run through a filter for nothing — which is why
  the Rust one takes the signal by value. The frame count is exactly `round(frames · to/from)`,
  because that is the number a host computed for itself before asking.
  There is now **one** sample-rate conversion in fluxion rather than two: `resample`, `speed` and
  `ensure_fs` all run the streaming `Resampler` from R1, so a file imported whole and the same file
  streamed through the worklet come out as the same samples. That also made the offline path about
  **11× faster** (10 s of 48 k → 44.1 k in 42 ms, from 471 ms) — the old one evaluated a `sin()`
  per tap per output sample where the streaming converter reads a precomputed table.
  Alignment is the part worth stating: a whole-signal conversion has no run-up to compensate, so
  `Resampler::align_to_input` puts output frame 0 on input frame 0 exactly. Rounding that
  compensation to the nearest output frame instead — the obvious first version — left a fifth of a
  frame of slip, visible as 0.022 of amplitude error on a 1 kHz tone. Aligned properly, a converted
  1 kHz tone matches the sine its new rate implies to 1.8e-6, which is f32 noise. `Resampler` also
  takes a ratio directly now (`with_ratio`), so a speed factor or a pitch interval is no longer
  rounded into a pair of integer rates on the way in.

- **Time tools: streaming resampler, time-stretch, pitch-shift (Epic R / R1, R3, R4 — completes
  milestone F-M4)** — three things a host needs and the offline converter could not give it.
  `Resampler` converts sample rates a block at a time with every buffer allocated in `new` and
  none afterwards, in two qualities: `Hq` is the offline filter, `Fast` is a quarter of the taps
  for scrubbing. A first attempt accumulated the read position per block and drifted, so 11025
  frames arrived as 11026 when the block size changed; the position is now computed from the
  input and output frame counters, which cannot drift. Checked against
  `scipy.signal.resample_poly` band by band across five signals — worst disagreement 0.05 dB of a
  1 dB bar — and on the failure that matters, folding: it pushes a 23 kHz tone down by 23.6 dB
  converting 48 k to 44.1 k, where `resample_poly` manages 11.1 dB.
  `pitchshift` is a chain op in cents, so like the mastering set it arrived on Rust, the CLI,
  Python, C and the browser at once. `stretch` is a CLI stage rather than an op, because it
  changes the frame count and length-preserving is what lets `|` and `+` compose.
  Underneath both is a phase vocoder with **peak-locked phases**: spectral peaks advance their own
  phase and every bin near a peak takes that peak's phase plus its current offset from it, which
  is what keeps the partials of one note in step instead of smearing. `docs/time-stretch.md` is
  the study roadmap R3 asks for — the reason it is written rather than bound to a C++ library
  (which would end the wasm build) or ported from one (which would be a fork to maintain), and
  the reason transients are the next thing to do rather than a thing already done.
  The oracle is Rubber Band, through ffmpeg. Scoring against its *output* was the first attempt
  and it measured the wrong thing — on a pure 440 Hz tone Rubber Band puts -41 dB of sideband at
  350 Hz where we put -88, so a test built that way fails us for being 46 dB cleaner. The ground
  truth is the **source**: a stretcher changes how long the material lasts and nothing else, so
  both are scored on how closely the output spectrum tracks the input's. Ours is 0.00 dB out on a
  sustained chord where Rubber Band is 0.15-0.53, and 0.85-1.69 on a sweep where it is 1.65-4.39;
  on band-limited noise it is about 0.7 dB better than us, which is the transient handling we do
  not have, and the test says so with a stated margin rather than hiding it. Duration is exact for
  every ratio — Rubber Band's is not, landing on 93566 frames where 96000 was asked for.


## [0.2.0] - 2026-08-03

First release of the host-engine push: [Epic I](ROADMAP.md) — one op registry behind every
interface, one text form for a chain, and a quickstart per interface that CI runs — plus the
WebAssembly build and the browser chain API (roadmap W1–W2).

### Changed

- **CLI help and errors that help (Epic I / I4)** — `fluxion --help` used to list ten global flags
  and nothing else: no verbs, no ops, no chain syntax. It now fits one screen (39 lines, 80
  columns — asserted) and names the verbs, the two self-describing commands and the chain syntax.
  Running `fluxion` bare or `fluxion help` prints that same screen and exits 0 instead of a usage
  error; `fluxion help <op>` describes the op. `fluxion lowpass` used to fail with `probing
  'lowpass': No such file or directory` because the op name was treated as an input file — it now
  describes the op. Errors suggest a fix: `unknown effect or stage 'hipass' — did you mean
  'highpass'?`, `effect 'lowpass' has no parameter '--cutof' — did you mean '--cutoff'?`, and a
  missing input says `no such file 'nope.wav'` rather than leaking a probe error. New `--chain
  "highpass(80, 4) | gain(-3dB)"` accepts the shared text form, and `--dry-run` prints the chain
  that would run — in canonical form, so it can be pasted straight back into `--chain`, Python or
  the browser (asserted). Long files get a progress readout, off unless stderr is a terminal so it
  can never contaminate a log. Sixteen committed snapshots pin the help screen and the ten common
  mistakes; `UPDATE_EXPECT=1 cargo test -p fluxion-cli` accepts a deliberate change. No snapshot
  library was added.
- **Breaking: the Python API is now torchfx-shaped (Epic I / I3)** — `fluxion.Wave` carries the
  sample rate, so `fs` leaves user code entirely: `wave | fx.filter.Highpass(80, order=4) |
  fx.effect.Gain(fx.db(-3))`, then `.save()`. `|` is series and `+` is parallel, the same algebra
  as the Rust library and the CLI. Piping is deferred — `w | a | b | c` accumulates one chain and
  runs it in a single fused pass rather than three. `Wave.from_file` / `.save` go through
  `fluxion-io`, so the wheel gained file I/O with no new Python dependency. Arrays still work
  everywhere a `Wave` does, and a chain is now directly callable: `chain(x, fs=48_000)`.
  The eighteen hand-written per-op functions are gone. Every op is a class in `fluxion.filter` or
  `fluxion.effect`, **generated** from the registry along with its typed stub and its docstring —
  coverage goes from 18 of 27 ops to all 27, and `fade`, `tremolo`, `overdrive`, `compand`,
  `reverse`, `biquad`, `chorus`, `flanger` and `phaser` reach Python for the first time. Names and
  parameter names are the registry's, so `low_shelf`/`high_shelf` become `LowShelf`/`HighShelf`,
  `delay(seconds=…)` becomes `Delay(time=…)` and `gain(value=…)` becomes `Gain(gain=…)`.
  `fluxion.chain("highpass(80, 4) | gain(-3dB)")` parses the shared text form and `str(chain)`
  prints it back. Errors name the class, the parameter and its range, and suggest a fix on a typo.
  A conformance test compares `fluxion.filter` / `fluxion.effect` against `fluxion.ops_table()` in
  both directions, so neither a missing class nor a stale generated file can survive CI. New
  helper: `fluxion.db(-3)` for the linear ratio the gain-like ops take. CI now builds the wheel
  and runs the ten-line quickstart from a *fresh* virtualenv on Linux, macOS and Windows on every
  pull request — previously the Python job was Linux-only and wheels were built on tags alone,
  so "pip install works" was never actually tested before a release.
- **Breaking: one op name on every interface (Epic I / I2)** — the four Chebyshev ops were spelled
  `cheby1low` / `cheby1high` / `cheby2low` / `cheby2high` in the CLI and chain text but
  `cheby1_lowpass` / … in Rust and Python. The registry name is now the long form everywhere, which
  is also what CONTRIBUTING's naming rule asks for. `.fxg` files are unaffected — they key on the
  Rust variant name, not this one. In the Rust prelude, `lowpass`/`highpass` now take
  `(cutoff, order)` like every other builder, and the `lowpass_n`/`highpass_n` synonyms are gone;
  the three ops the prelude was silently missing (`reverb`, `cheby2_lowpass`, `cheby2_highpass`)
  are added. A conformance test in each of `fluxion` and `fluxion-cli` now fails the build if the
  prelude or the `effects` listing drifts from the registry, which is how those three gaps were
  found.
- **`Graph`'s text rendering is now canonical (Epic I / I2)** — `Display` used to print
  `Series(Series(a,b),c)` and `Series(a,Series(b,c))` identically, and a `Named` node wrapping a
  composite lost track of where the label ended. Both are now bracketed: a same-kind child is
  parenthesized when it is the *right* operand (the operators parse left-associative, so only that
  side is ambiguous), and `name:` brackets a composite child. Mixed nesting is unchanged, and so is
  every string the crate printed before — this only adds parens where two different graphs used to
  render alike. That makes the rendering a canonical form, which is what lets the chain-text parser
  be its exact inverse.
- **One declarative op table (Epic I / I2)** — the op catalog in `fluxion-core` is now declared
  once, in a single `ops!` table: one row per op carrying its doc comment, Rust variant, stable
  text name, catalogue group and parameter schema. `OpKind::name`, `group`, `params` and `all` are
  generated from that row, replacing four hand-kept parallel lists and twenty `static X_PARAMS`
  tables that nothing kept in sync. Adding an op is one edit. `OpKind` stays a real
  `#[non_exhaustive]` enum, so the backend's exhaustive matching and the serde representation are
  unchanged. New: `OpKind::group() -> Group` (`filter` / `effect`), the catalogue split that drives
  `fluxion.filter` vs `fluxion.effect` in Python and the sections of the generated op docs.
  `.fxg` keys on the Rust variant identifier, not the text name, so a committed
  `tests/fixtures/all_ops_v1.fxg` holding one op of every kind now guards the format against a
  variant rename — the one refactor that would silently orphan saved graphs. New registry
  invariants are asserted too: op names are unique and identifier-shaped, parameter names are
  unique per op and identifier-shaped, and every default sits inside its own bounds.
- **CPU batch kernel: set-spreading tile** — `sos_filter_batch`'s AVX2 8-row path now
  bounces each time block through a small padded-pitch scratch instead of loading the
  planar rows at their raw stride. Power-of-two row strides (e.g. the 64×524k paper
  workload's 2 MB) map all sixteen row streams to a single L1/L2 set on 8-way parts
  and cap the kernel at LLC speed; the tile restores cache-resident loads for two
  L2-resident copies. Measured on the i9-10900KF: 1.66 → 2.60 Gsamples/s multi-thread
  (+57%, past TorchFX's OpenMP kernel at 1.89) for ≈5% single-thread; per-row outputs
  stay bit-identical (asserted across tile boundaries and scalar tails).

### Added

- **AudioWorklet playback (roadmap W4, W5, W7 — completes milestone F-M2, and I6 with it)** —
  `attachWorklet(context, chain)` runs a chain on the audio thread, 128 frames at a time, fed
  through the lock-free SPSC ring and the allocation-free block executor `fluxion-rt` already had.
  The claims are measured rather than asserted. Five seconds of playback allocates **nothing** on
  the wasm side — the module counts its own allocations through a wrapping global allocator, and
  the test requires the count not to move; a 40 dB gain change arrives as a ramp whose largest
  per-sample step is 1e-3, against the 0.99 a step would give; the module is 193 KB gzipped of the
  1.5 MB budget; and a five-filter mastering chain costs 2.8 us per block, 0.11% of its 2.67 ms
  deadline. Block-by-block playback is bit-identical to the offline render, checked in Node and
  again inside a real browser — which is the concrete form of "preview and export come from the
  same DSP".
  Two things a browser makes you find out the hard way. `AudioWorkletGlobalScope` has no
  `TextDecoder` or `TextEncoder`, and wasm-bindgen's glue builds one at its top level, so the
  worklet script dies before `registerProcessor` and the only symptom is a processor that "is not
  defined"; a small UTF-8 shim is prepended for that reason. And an `OfflineAudioContext` can
  finish rendering before a `postMessage` is delivered, so audio that must be present for the first
  block arrives through `processorOptions` instead.
  A chain that cannot run in a callback — `reverse`, `normalize`, `limiter`, `loudnorm` all need
  the whole signal — is refused when the player is built, with a message saying which and what to
  use instead, rather than failing part-way through playback.
- **The mastering set (Epic M / M1-M4 — completes milestone F-M3)** — a mastering chain needed
  four things the dynamics module did not have. `stat` now reports integrated loudness, loudness
  range and true peak; `limiter` and `loudnorm` are chain ops, so they arrived on Rust, the CLI,
  Python, C and the browser at once by being one row each in the registry.
  Loudness is ITU-R BS.1770: K-weighting designed analytically (so it is correct at any sample
  rate, and reproduces the standard's tabulated 48 kHz coefficients to machine precision), the
  two-pass gate, and EBU Tech 3342 loudness range. Checked against pyloudnorm and ffmpeg's
  `ebur128` over 11 signals covering the K-curve, channel summing and gating — worst disagreement
  0.058 LU against a 0.1 LU bar, where the two references disagree with *each other* by up to
  0.055.
  True peak oversamples 4x through a 96-tap polyphase interpolator whose length was chosen by
  measurement against signals with analytically known peaks. M2 asked for 0.1 dB of ffmpeg; that
  is not achievable and measurement says why — ffmpeg reports 10 kHz and 19 kHz sines of amplitude
  0.5 as -5.2 dBTP, above their own mathematical maximum, and reads the canonical inter-sample-peak
  fixture 0.62 dB high. The test says so, and pins our accuracy against the truth instead.
  The limiter computes its gain from the reconstructed waveform, not the samples, applies one gain
  curve across all channels so the image does not wander, and holds its ceiling on *any* input —
  including full-scale noise and square waves, where a first attempt did not, because a gain that
  moves sample-to-sample creates the very inter-sample peaks it is removing. Loudness normalize is
  measure-apply-verify, keeping the closest attempt rather than the last: material with enough
  crest factor cannot reach a loud target under a strict ceiling, and chasing it would hand back
  something quieter than it started.
- **The wasm-vs-native suite, over every op (roadmap W6 — completes milestone F-M1)** — W2 compared
  one chain; this compares all 27 ops plus five chain topologies (series, parallel, nested, a
  labelled node, and feedback — the one construct a series/parallel tree cannot encode), 4096
  frames each, on one bit-exact input. **26 of the 32 cases are bit-identical to native.** The six
  that are not are exactly the ops that call a transcendental *per sample*, where wasm's `libm` and
  the platform's differ in the last bit: `overdrive` (`tanh`) and `phaser` (`sin` LFO) at 2.4e-7,
  `compand` (`exp`/`ln`) at 1.2e-7, `fade` at 6.0e-8, `tremolo` at 4.5e-8. f32's epsilon is 1.2e-7,
  so the worst of those is two ULP. Every *designed* filter is bit-identical, which says the two
  libms agree on the trigonometry the coefficient design calls; only the per-sample calls diverge.
  The tolerance is 1e-6 with the measurements tabulated beside it, and the suite prints what it
  actually measured on every run, so drift toward the bound is visible instead of hidden behind a
  pass. A coverage test fails the build if an op has no case, so a new op cannot skip the
  comparison, and each case must demonstrably change the signal — parameters that happen to be a
  no-op would otherwise pass while proving nothing.
  The native reference is now **generated at test time rather than committed**. The wasm job builds
  the module, so it has a Rust toolchain by definition; comparing against current native costs
  nothing and removes both the staleness risk and the second, looser tolerance the committed
  fixture needed to survive the difference between glibc's and Apple's `sin`.
- **The JS package and five executed quickstarts (Epic I / I6, I7)** — `crates/fluxion-wasm/js`
  is an npm package: `import init, { Chain } from "fluxion"`. TypeScript types are generated from
  the registry, including `OpName` as a literal union, so a misspelled op is a compile error rather
  than a runtime throw; the catalog also ships as data at `fluxion/ops` (a bundler can drop it,
  which is not true of anything compiled into the wasm module). The AudioWorklet half of I6 waits
  on W4 and is recorded as such — the offline `Chain` is the class it will extend.
  Each of the five interfaces now has a quickstart that CI **runs**: a Rust doctest, a shell script,
  a Python script in a fresh virtualenv on three systems, a C program compiled on three systems,
  and a Node script. None exceeds ten code lines (6, 5, 8, 10, 8), and the budget is enforced by
  one function in the generator so all five are measured the same way. If one grows past ten, the
  generator fails the build — the roadmap's stated signal that the API, not the quickstart, is
  wrong.
- **The chain API in the browser (roadmap W2)** — `Chain.fromText("highpass(80, 4) | gain(-3dB)")`,
  `.process(samples, fs)`, `.toText()`, plus `ops()` and `version()`. The same graph, from the same
  string, as the native library — and the parity check that says so is committed: a fixture holds a
  native-rendered reference which `cargo test` re-verifies against native, and a Node script feeds
  the same input through the built wasm module and compares. It matches **bit for bit** (worst
  difference 0.0, against a 1e-6 tolerance), which is the concrete form of "preview and export come
  from the same DSP". The Node side reads a committed fixture, so it needs no Rust toolchain. A
  misspelled op throws with the caret rendering and the suggestion, exactly as it does everywhere
  else.
- **WebAssembly build (roadmap W1)** — `crates/fluxion-wasm` was a seven-line scaffold with no
  dependencies. It now compiles to wasm32 with wasm-bindgen, and `scripts/build-wasm.sh` emits the
  module plus its JavaScript glue and `.d.ts`. A `wasm` CI job builds it on every PR and loads it
  in Node, so the browser path cannot quietly stop working. The facade's dependency tree turned out
  to be wasm-clean as-is — rustfft included — because it does not pull `fluxion-io` and CubeCL sits
  behind the `cuda` feature. `rust-toolchain.toml` now installs the `wasm32-unknown-unknown` target,
  so a fresh checkout can build it without a separate `rustup target add`. Panics route through
  `console_error_panic_hook`, since the default is an unhelpful `unreachable executed`.
- **C: build a chain from text (Epic I / I5)** — `fx_chain_from_text("highpass(80, 4) |
  gain(-3dB)")` reaches the whole op catalog from C, and `fx_graph_to_text` prints a graph back in
  the same form (snprintf semantics: caller-owned buffer, returns the length needed). Before this,
  the only way into the C door was loading a `.fxg` file produced elsewhere — zero of the 27 ops
  were reachable. Two symbols rather than one per op is deliberate: the roadmap's contract for C is
  "the stable core … small on purpose", and every ABI symbol is permanent. A test asserts every
  registry op parses from C, which is what the "via chain text" column in `docs/ops.md` means.
  Errors, including the caret rendering and the did-you-mean suggestion, come through
  `fx_last_error`. The `ffi` CI job is now a Linux/macOS/Windows matrix — it was Linux-only, and
  its link line used `-l:libfluxion_ffi.a`, a GNU-ld extension that Apple's ld64 rejects and MSVC
  does not have; passing the archive as a plain file path works everywhere. A new step regenerates
  `fluxion.h` with a pinned cbindgen and diffs it, so the committed header can no longer drift from
  the source it claims to describe.
- **The interface contract, in `docs/` (Epic I / I1)** — a new `docs/` directory carrying the
  agreement between Fluxion's five doors. [`docs/interfaces.md`](docs/interfaces.md) states the
  rule the epic exists to enforce — *an op ships on every interface, or this document says why
  not* — plus who each interface is for, the definition of done a PR is held to, the ten-line
  quickstart rule and how a line is counted, and an honest "Not yet" section for the gaps that
  remain (the AudioWorklet half of the JS package, the geometry stages, `.to(device)`, alias
  spellings) with the reason for each rather than just the absence.
  [`docs/chain-syntax.md`](docs/chain-syntax.md) documents the shared text form.
  [`docs/ops.md`](docs/ops.md) is **generated** — every op, what it does, its parameters, ranges
  and its name on each interface — by `scripts/gen_interfaces.py`, which reads the registry through
  the new `fluxion effects --json`. A `contract` CI job regenerates and runs `git diff
  --exit-code`, so a hand edit or an op added without regenerating fails the build. The registry
  gained `OpKind::variant()` (the Rust identifier, which is also the Python class name) and
  `OpKind::doc()` (the catalog's doc comment), so an op's documentation is written once and used
  three times: rustdoc, the generated table, and the Python docstring.
- **The chain text syntax (Epic I / I2)** — `"highpass(80, 4) | gain(-3dB)".parse::<Graph>()`. One
  grammar, in `fluxion-core`, that is the exact inverse of what `Graph` prints: whatever the library
  renders parses back to the identical graph, asserted over a corpus covering every op, both
  associativities of `|` and `+`, labels around every node kind, feedback nested either way, and the
  numeric edges (`inf`, signed zero, exponents). That is what lets the CLI's `--chain`, Python's
  `fluxion.chain()`, C's `fx_chain_from_text` and the browser describe a chain with one string
  instead of four dialects. `+` binds tighter than `|`, matching Rust and Python, so
  `a | b + c | d` needs no parentheses. Ergonomics on the way in, canonical form on the way out:
  trailing parameters fall back to their defaults (`highpass(80)`), arguments may be named
  (`compand(threshold=-24, ratio=8)`), there is a `name=v1,v2` shorthand, and the suffixes `k`
  (×1000) and `dB` are accepted — `gain(-3dB)` is an actual 3 dB cut, which `gain(-3)` is not, since
  that parameter is a linear ratio. A suffix a parameter cannot take is refused rather than
  misread. Errors carry a byte offset, a message and an optional fix, and render with a caret:
  `unknown op 'hipass'` under `^^^^^^ did you mean 'highpass'?`. The suggestion helper
  (`fluxion_core::suggest`) counts a transposition as one edit, so `gian` finds `gain`; it adds no
  dependency.
- **Realtime bindings (Python + C ABI)** — the `fluxion-rt` streaming engine is now reachable
  from both binding surfaces, closing the "batch-only bindings" gap. Python: `fluxion.RtChain`
  (`from_chain` / `from_sections`) lowers a `Chain` once at a fixed `fs`, certifies it on the
  stability ladder (an `unstable` verdict is refused), pre-sizes every scratch buffer, then
  `process(input, output)` filters caller-provided numpy blocks in place — allocation-free,
  filter state carried across calls (chunked streaming ≡ whole-signal, asserted), GIL released
  while the Rust kernel runs; `set_coeffs(node, sections, fade_samples)` swaps a filter live with
  the equal-power crossfade (incoming sections certified too), and `filter_count` / `fs` /
  `max_block` / `verdict` / `margin` expose the executor's state. C:
  `fx_rt_new` / `fx_rt_process` / `fx_rt_set_coeffs` / `fx_rt_reset` / `fx_rt_filter_count` /
  `fx_rt_free` plus the `FX_VERDICT_*` codes in the regenerated `include/fluxion.h` — the same
  certification gate, panic-safe at the boundary, with an allocation-free process path that is
  safe inside an audio callback (in/out aliasing and oversized blocks are rejected, not UB).
  This lets a C/C++ host stream a loaded `.fxg` block-by-block; a round-trip test provisions a
  FIR + gain graph to disk, loads it through the C ABI, and asserts the streamed output matches
  the batch path.
- **`.fxg` provisioning from Python** — `Chain.save_fxg(path)` serializes the whole chain
  (biquads, FIR taps, gain, delay) to a sample-rate-agnostic `.fxg` graph a C/C++ host loads with
  `fx_graph_load_fxg` (the pre-existing `save_biquad_fxg` only covered raw SOS sections), and
  `Chain.certify(fs)` returns the `(verdict, margin)` stability certificate for fail-fast
  per-channel checks in a provisioning script.
- **Checkpoint import (goal #6 / J13, full slice)** — run DDSP filters trained in other
  frameworks: `fluxion import ckpt.safetensors model.fxg` (CLI) and
  `fluxion.interop.import_checkpoint(...)` (Python; also parses `.pt` and `.onnx` and torchfx
  compiled artifacts) replay the exact param→coefficient math of FLAMO
  (`SOSFilter`/`SVF` all filter types/`Biquad`, realised `b`/`a`, RBJ tables) and torchfx.ddsp
  (learnable lowpass/highpass/peaking/parametric-EQ) into raw `biquad` sections, **certify** them
  on the stability ladder (E8; `--project-stable` Jury-clamps unstable checkpoints), and write a
  standard `.fxg` that splices into any pipeline, plays realtime, and hot-swaps. Rust converter in
  `fluxion-io::checkpoint` (feature `checkpoint`, pure-Rust `safetensors` reader); golden-tested
  against 15 real FLAMO/torchfx checkpoints. SISO only; MIMO banks and FIR taps are rejected with
  clear errors.
- **Filters & effects** — Butterworth and Chebyshev I/II low/high-pass; RBJ biquads (peaking,
  low/high shelf, notch, band-pass, all-pass) and a raw-coefficient `biquad`; FIR (plus FFT
  convolution); gain, normalize, delay (integer + fractional), echo, and Schroeder–Moorer reverb;
  plus a SoX-parity effect batch — `fade`, `tremolo`, `overdrive`, `compand` (feed-forward
  compressor, realtime-playable), `reverse`, and the modulated `chorus` / `flanger` / `phaser` — all
  as composable graph ops, designed from closed forms with no SciPy at runtime.
- **Geometry transforms** — whole-`Signal` verbs that change frame/channel count or sample rate
  (deliberately outside the graph algebra): `trim`, `pad`, `repeat`, `silence_trim`, a real
  windowed-sinc `resample` (the SoX `rate` replacement, anti-aliased) and `speed`, `remix` /
  `channels` (energy-preserving), and the `concat` / `mix` multi-input primitives.
- **Functional graph algebra** — `|` (series) and `+` (parallel) composition, node identity
  (`Graph::Named`, addressable by name), and the `~` feedback operator (`Graph::feedback`).
- **Differentiable DSP** — hand-derived analytic VJPs for every op; whole-graph reverse-mode autodiff
  through Burn (`fluxion::diff_process`); trainable filter coefficients and *design parameters*
  ("learn a cutoff") and FIR taps; an in-loop Jury-triangle stability projection; and torch
  (`SosModule`, `torch.autograd.Function`) + `jax.custom_vjp` adapters.
- **GPU** — CubeCL SOS forward + backward kernels (validated on CUDA) and a split CPU/GPU Python wheel.
- **Real-time engine** — allocation-free, lock-free block executor (SOS cascade, general
  series/parallel graph, reverb, FIR, delay/echo, fractional delay, compand); click-free parameter
  automation with an equal-power coefficient crossfade; a lock-free SPSC command queue; and a CPAL
  audio backend. Reachable from the `fluxion` facade via its `realtime` feature (re-exporting
  `RtGraph` / `RtEngine` / `SosStream` / `SmoothedValue`, `freeze` / `to_rt_graph`, and `FrozenSos`).
- **CLI (`fluxion`)** — a SoX substitute with named effects and long flags: a stage pipeline mixing
  filter passes with geometry stages (`trim`, `pad`, `rate`, `speed`, `repeat`, `silence`,
  `channels`, `remix`); multi-input concatenation and `--mix`; `--db` and SI-suffix (`1k`) parsing;
  output encoding control (`--bits 16|24|32`, `--float`, `--no-dither`); verbs `info`/`soxi` (all
  formats via Symphonia probe), `stat`, `effects` (self-describing op catalog), `synth`, `compile`
  (→ `.fxg`), `batch`, stdin/stdout (`-`) / null-sink (`-n`); realtime `play`/`record`
  (feature `realtime`).
- **Audio IO** — WAV read/write via hound with output encoding options (16/24/32-bit integer PCM
  with TPDF dither on by default, or 32-bit float) and decode + header-only `probe` of
  FLAC/MP3/OGG/AAC/… via Symphonia (pure Rust, no libsndfile/ffmpeg). Bounded-memory streaming
  readers (`read_wav_blocks` / `decode_blocks`) yield fixed-size `Signal` chunks for large files,
  and columnar dataset IO (`Signal` ↔ Arrow `RecordBatch` ↔ Parquet) sits behind an optional
  `parquet` feature for the augmentation workflow.
- **Python bindings** — torchaudio-style eager `Chain` API accepting 1-D `(T,)` and 2-D `(C, T)`
  input plus a batched `Chain.process_batch((B, T))`, zero-copy DLPack interop with NumPy /
  PyTorch / JAX, Array-API consumer conformance, `fluxion.augment` (`Compose`, `RandomChain`)
  for stochastic data augmentation, `fluxion.dataset` (Parquet audio-dataset IO — the same schema
  as the Rust side, streaming both ways; extra `fluxion[dataset]`), and
  `fluxion.interop.load_flamo_sos` for importing FLAMO-style SISO biquad checkpoints
  (`safetensors`).
- **C ABI (`fluxion-ffi`)** — a minimal panic-safe C surface (`fx_graph_load_fxg`, `fx_process`
  interleaved in-place, `fx_last_error`) with a checked-in `include/fluxion.h` and a C smoke test.
- **Quality gates** — SciPy/RBJ golden-vector oracle tests pinning every filter design's impulse
  response (32 cases, no runtime SciPy); Criterion benchmarks (`cargo bench`); CI jobs for
  benches, the C ABI, and a CUDA compile check; PyPI wheels for Linux x86_64/aarch64,
  macOS Intel/Apple-Silicon, and Windows, published on tag via Trusted Publishing.
- **Serialization** — versioned `.fxg` graph and `FrozenSos` plan envelopes
  (`{version, kind, fs, payload}`), rejecting incompatible/old files with a clear error.
- **Stability certification** — a pole-based + small-gain verdict ladder over a graph's frozen
  coefficients, gating `.fxg` export / realtime freeze.

### Notes

- Pre-1.0: the public Rust/Python API and the `.fxg` on-disk format are not yet stable.

[Unreleased]: https://github.com/matteospanio/fluxion/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/matteospanio/fluxion/releases/tag/v0.2.0

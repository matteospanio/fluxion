# Time-stretch: the short study ROADMAP R3 asks for

R3 says "Signalsmith-Stretch class. Short study first: pure-Rust port vs binding, keeping the
reference implementation as a test-only oracle". This is that study and the decision it reached.

## The three options

**Bind to a C++ library** (Signalsmith Stretch, Rubber Band, SoundTouch). Best quality for the least
DSP work, and openDAW's `signalsmith` crate shows it is workable. But fluxion's backbone is a
dependency-light crate that builds offline, and every interface in `docs/interfaces.md` has to keep
working: a C++ dependency means a C++ toolchain on every contributor's machine and in all three CI
systems, and `wasm32-unknown-unknown` — which W1–W7 just finished wiring up — has no C++ standard
library. That last point is not a preference, it is the JS interface disappearing. Rejected.

**Port Signalsmith Stretch to Rust.** Legally fine (it is MIT), and it would land the quality
immediately. Rejected on maintenance: a port is a fork, and a fork of somebody else's evolving DSP
is a debt that comes due every time upstream changes. It also sits badly with how this repo already
works — `CONTRIBUTING.md` takes ideas and math from reference projects, never source, and every
filter here is derived from published formulas rather than transcribed.

**Write the algorithm, keep the references as oracles.** Chosen. A phase vocoder with peak-locked
phases is well-documented published DSP (Laroche & Dolson, *Improved Phase Vocoder Time-Scale
Modification of Audio*, 1999), it is the class of algorithm Signalsmith Stretch belongs to, and it
is about 200 lines on top of the `rustfft` this crate already depends on. Correctness comes from the
oracles rather than from the source: `ffmpeg -af rubberband` and `librosa.effects.time_stretch`
generate the fixtures in `scripts/gen_stretch_golden.py`, exactly as ffmpeg and pyloudnorm do for
the loudness work in `crates/fluxion-ops/tests/loudness_golden.rs`.

## What is implemented

`fluxion_ops::stretch` — STFT with a ~85 ms Hann window at 75 % overlap, per-bin instantaneous
frequency from the phase deviation between frames, and **identity phase locking**: spectral peaks
advance their own phase, and every bin near a peak takes that peak's phase plus its current offset
from it. Locking is what separates this from a textbook phase vocoder — without it the partials of
one note drift out of step with each other and the result is the familiar smeared, "phasey" sound.

`pitch_shift` is R4 and comes free from R3 plus R1: stretch by `2^(cents/1200)`, then play the
result back at that same factor with the streaming resampler. Tempo cancels, pitch does not.

## What is not implemented

Transient handling. Signalsmith Stretch and Rubber Band both detect transients and reset phases
across the whole spectrum so a drum hit stays sharp; here a transient is smeared over the window.
`stretch.rs` marks it `// ponytail:`. It is the single biggest quality gap and the obvious next
change, but it is a quality refinement on a correct converter rather than something F-M4 needs.

Formant preservation, and stretch ratios far from 1. The tests pin 0.5×–2×; outside that the window
should adapt and does not.

## How it is checked

`crates/fluxion-ops/tests/stretch_golden.rs`:

- duration is exact, for every ratio in the fixture set — the roadmap's own wording;
- pitch does not move: a 440 Hz tone stretched 0.5×–2× still peaks at 440 Hz;
- the long-term spectrum matches `rubberband`'s band by band, within a written tolerance.

The third is the interesting one, and it is spectral for the same reason R1's is: two stretchers
running different algorithms produce completely different waveforms on purpose. What they must agree
on is *what the signal contains*, which is what a listener hears and what a sample-wise comparison
cannot see.

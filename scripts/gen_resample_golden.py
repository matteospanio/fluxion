#!/usr/bin/env python3
"""Generate the resampler oracle vectors for `crates/fluxion-ops/tests/resample_golden.rs`.

`scipy.signal.resample_poly` is the independent reference ROADMAP task R1 names. It is a different
filter design from ours — a Kaiser-windowed polyphase FIR against our Blackman-windowed sinc — so
this is not a bit-comparison; it is a check that two well-built converters agree on what the signal
*is*, to a tolerance the test writes down.

Signals are described by parameters and rebuilt identically on the Rust side; only the reference
output is stored, and only for a short window, so the generated file stays readable.

    python scripts/gen_resample_golden.py

Conventions: frequencies in Hz, sample rate is `fs`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.signal as ss

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "crates/fluxion-ops/tests/resample_golden_data.rs"

FROM_FS = 48_000
TO_FS = 44_100
# 147/160 is 44100/48000 exactly, which is what resample_poly wants.
UP, DOWN = 147, 160

# Output frames analysed per case. A power of two, for the FFT.
KEEP = 8192
# Skip past both converters' start-up before the analysed window.
SKIP_OUT = 4096
# Bins of the compared spectrum. Averaging the FFT down to this many keeps the generated file
# readable and removes the bin-to-bin noise that says nothing about either converter.
BANDS = 128
# The band compared. Above this the two filter designs are comparing their transition bands with
# each other, which is not a statement about either being correct — that is what the alias test is
# for.
BAND_LO, BAND_HI = 30.0, 15_000.0

CASES = [
    # (name, kind, freq)
    ("tone_1k", "tone", 1000.0),
    ("tone_5k", "tone", 5000.0),
    # Close to the transition band, where two different filter designs disagree most.
    ("tone_15k", "tone", 15000.0),
    ("sweep", "sweep", 0.0),
    ("noise", "noise", 0.0),
]

SECONDS = 1.0


def signal(kind: str, freq: float) -> np.ndarray:
    """Build one case. Must match `signal()` in crates/fluxion-ops/tests/resample_golden.rs."""
    n = int(SECONDS * FROM_FS)
    t = np.arange(n) / FROM_FS
    if kind == "tone":
        return np.sin(2 * np.pi * freq * t)
    if kind == "sweep":
        # Stops well below Nyquist: above it the two designs are comparing their transition bands
        # with each other, which says nothing about either being right.
        return ss.chirp(t, 20.0, SECONDS, 15000.0)
    if kind == "noise":
        # The same integer LCG the Rust side uses, band-limited so the comparison is about
        # conversion rather than about who rolls off where.
        out = np.empty(n)
        state = np.uint32(0x1234_5678)
        for i in range(n):
            state = np.uint32(
                (np.uint64(state) * np.uint64(1_664_525) + np.uint64(1_013_904_223))
                & np.uint64(0xFFFF_FFFF)
            )
            out[i] = (int(state) >> 8) / 16_777_216.0 * 2.0 - 1.0
        sos = ss.butter(8, 15000, "low", fs=FROM_FS, output="sos")
        return ss.sosfilt(sos, out) * 0.5
    raise ValueError(f"unknown kind {kind!r}")


def spectrum(x: np.ndarray) -> np.ndarray:
    """Magnitude spectrum in dB, averaged into BANDS bins across the compared band.

    Comparing spectra rather than samples is deliberate. Two converters can be equally correct and
    still differ by a fraction of a sample in delay, and at 15 kHz a sixth of a sample is already a
    third of full scale — a sample-wise comparison would be measuring alignment, not conversion.
    """
    windowed = x * np.hanning(len(x))
    mag = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(x), 1 / TO_FS)
    edges = np.geomspace(BAND_LO, BAND_HI, BANDS + 1)
    out = np.empty(BANDS)
    for i in range(BANDS):
        band = (freqs >= edges[i]) & (freqs < edges[i + 1])
        out[i] = np.sqrt(np.mean(mag[band] ** 2)) if band.any() else 0.0
    return 20 * np.log10(np.maximum(out, 1e-9))


def f32(v: float) -> str:
    """Shortest literal that round-trips to the nearest f32 (see scripts/gen_golden.py)."""
    x = np.float32(v)
    text = repr(float(x))
    for digits in range(1, 10):
        candidate = "%.{}g".format(digits) % float(x)
        if np.float32(float(candidate)) == x:
            text = candidate
            break
    if "." not in text and "e" not in text:
        text += ".0"
    return text


def main() -> int:
    rows = []
    print(f"{'case':10} {'ref rms':>10} {'alias rejection':>17}")
    for name, kind, freq in CASES:
        x = signal(kind, freq)
        # resample_poly compensates its own filter delay, so its output lines up with the input.
        y = ss.resample_poly(x, UP, DOWN)
        window = y[SKIP_OUT : SKIP_OUT + KEEP]
        print(f"{name:10} {np.sqrt(np.mean(window**2)):10.5f}")
        rows.append((name, kind, freq, spectrum(window)))

    # Alias rejection: a tone above the output Nyquist has nowhere to go but into the band, and a
    # converter that lets it through is broken in the way that actually matters.
    alias_freq = 23_000.0
    t = np.arange(int(SECONDS * FROM_FS)) / FROM_FS
    alias_out = ss.resample_poly(np.sin(2 * np.pi * alias_freq * t), UP, DOWN)
    alias_rms = float(np.sqrt(np.mean(alias_out[SKIP_OUT:-SKIP_OUT] ** 2)))
    alias_db = 20 * np.log10(max(alias_rms, 1e-12) / (1 / np.sqrt(2)))
    print(f"{'alias 23k':10} {alias_rms:10.6f} {alias_db:16.1f} dB")

    lines = [
        "// @generated by scripts/gen_resample_golden.py — do not edit.",
        f"// scipy {ss.__name__} {__import__('scipy').__version__}, numpy {np.__version__}",
        "",
        "//! Resampler oracle vectors for `resample_golden.rs`.",
        "//!",
        "//! `scipy.signal.resample_poly` converting 48 kHz to 44.1 kHz. Signals are described by",
        "//! parameters and rebuilt identically by `signal()` in the test; only the reference output",
        "//! is stored, for a window of `KEEP` frames starting `SKIP_OUT` in.",
        "#![allow(dead_code)]",
        "",
        "/// Output frames analysed per case.",
        f"pub const KEEP: usize = {KEEP};",
        "/// Bins of the compared spectrum.",
        f"pub const BANDS: usize = {BANDS};",
        "/// The band compared, in Hz.",
        f"pub const BAND_LO: f32 = {f32(BAND_LO)};",
        "/// The top of the band compared, in Hz.",
        f"pub const BAND_HI: f32 = {f32(BAND_HI)};",
        "/// Output frames skipped before the compared window, so both converters have started up.",
        f"pub const SKIP_OUT: usize = {SKIP_OUT};",
        "/// Seconds of input per case.",
        f"pub const SECONDS: f32 = {f32(SECONDS)};",
        "/// How far `resample_poly` pushes a 23 kHz tone down, in dB. Ours must do at least as well.",
        f"pub const ALIAS_REJECTION_DB: f32 = {f32(alias_db)};",
        "",
        "/// One oracle case.",
        "pub struct ResampleCase {",
        "    /// Case name; the join key with the test's own list.",
        "    pub name: &'static str,",
        "    /// Which shape `signal()` should build.",
        "    pub kind: &'static str,",
        "    /// Tone frequency, Hz (ignored by sweep and noise).",
        "    pub freq: f32,",
        "    /// The magnitude spectrum of what resample_poly produced, in dB: `BANDS` bins",
        "    /// spaced geometrically across `BAND_LO`..`BAND_HI`.",
        "    pub expected_db: &'static [f32],",
        "}",
        "",
    ]
    for name, kind, freq, window in rows:
        values = ", ".join(f32(v) for v in window)
        lines.append(f"#[rustfmt::skip]")
        lines.append(f"const {name.upper()}: &[f32] = &[{values}];")
    lines += ["", "/// Every oracle case, in the order the generator emitted them.", "pub const RESAMPLE_CASES: &[ResampleCase] = &["]
    for name, kind, freq, _ in rows:
        lines.append(
            f'    ResampleCase {{ name: "{name}", kind: "{kind}", freq: {f32(freq)}, '
            f"expected_db: {name.upper()} }},"
        )
    lines += ["];", ""]

    OUT.write_text("\n".join(lines))
    print(f"\nwrote {len(rows)} cases to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

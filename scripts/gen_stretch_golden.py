#!/usr/bin/env python3
"""Generate the time-stretch oracle vectors for `crates/fluxion-ops/tests/stretch_golden.rs`.

The reference is **Rubber Band**, through `ffmpeg -af rubberband` — an independent, well-regarded
implementation of the same class of algorithm, kept test-only exactly as ROADMAP R3 asks. ffmpeg is
already the oracle for the loudness work (`scripts/gen_loudness_golden.py`), and as there the
vectors are committed, so nothing at test time needs ffmpeg installed.

    python scripts/gen_stretch_golden.py

What is compared is the **spectral shape of the whole output**, normalized to its own peak.

Two things force that choice. Sample-wise is meaningless: two stretchers reconstruct completely
different waveforms on purpose, and only agree about what the signal contains. And a *window* of the
output is no good either, because Rubber Band's duration is not exact — a 1 s tone stretched 2x
comes back as 93566 samples where ours is 96000 — so "the middle 16384 samples" is a different part
of a sweep for each. The whole signal is the one region both agree on.

Conventions: frequencies in Hz, sample rate is `fs`.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as ss

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "crates/fluxion-ops/tests/stretch_golden_data.rs"

FS = 48_000
SECONDS = 1.0

# Bins of the compared spectrum, geometric across the band that matters.
BANDS = 64
BAND_LO, BAND_HI = 50.0, 15_000.0

# (name, kind, freq)
SIGNALS = [
    ("tone", "tone", 440.0),
    ("chord", "chord", 0.0),
    ("sweep", "sweep", 0.0),
    ("noise", "noise", 0.0),
]
# Output duration over input duration. `tempo` in Rubber Band is the reciprocal: it is a speed.
RATIOS = [0.5, 0.75, 1.5, 2.0]


def signal(kind: str, freq: float) -> np.ndarray:
    """Build one case. Must match `signal()` in crates/fluxion-ops/tests/stretch_golden.rs."""
    n = int(SECONDS * FS)
    t = np.arange(n) / FS
    if kind == "tone":
        return np.sin(2 * np.pi * freq * t) * 0.5
    if kind == "chord":
        # A3 major triad: three partials close enough that a vocaoder without phase locking smears
        # them into each other, which is the whole point of testing it.
        return sum(np.sin(2 * np.pi * f * t) for f in (220.0, 277.183, 329.628)) * 0.2
    if kind == "sweep":
        return ss.chirp(t, 100.0, SECONDS, 8000.0) * 0.5
    if kind == "noise":
        out = np.empty(n)
        state = np.uint32(0x1234_5678)
        for i in range(n):
            state = np.uint32(
                (np.uint64(state) * np.uint64(1_664_525) + np.uint64(1_013_904_223))
                & np.uint64(0xFFFF_FFFF)
            )
            out[i] = (int(state) >> 8) / 16_777_216.0 * 2.0 - 1.0
        sos = ss.butter(8, 12000, "low", fs=FS, output="sos")
        return ss.sosfilt(sos, out) * 0.5
    raise ValueError(f"unknown kind {kind!r}")


def spectrum(x: np.ndarray) -> np.ndarray:
    """Band spectrum of the whole signal in dB, normalized so the loudest band reads 0.

    Normalizing by the peak is what makes this comparable across two stretchers that do not agree
    about output gain or output length. What is left is the shape, which is what a listener hears.

    Must match `spectrum()` in crates/fluxion-ops/tests/stretch_golden.rs.
    """
    n = 1 << int(np.ceil(np.log2(len(x))))
    windowed = x * np.hanning(len(x))
    mag = np.abs(np.fft.rfft(windowed, n))
    freqs = np.fft.rfftfreq(n, 1 / FS)
    edges = np.geomspace(BAND_LO, BAND_HI, BANDS + 1)
    out = np.empty(BANDS)
    for i in range(BANDS):
        band = (freqs >= edges[i]) & (freqs < edges[i + 1])
        out[i] = np.sqrt(np.mean(mag[band] ** 2)) if band.any() else 0.0
    db = 20 * np.log10(np.maximum(out, 1e-12))
    return db - db.max()


def rubberband(x: np.ndarray, ratio: float, tmp: Path) -> np.ndarray:
    """Stretch to `ratio` times the duration with Rubber Band, via ffmpeg."""
    src, dst = tmp / "in.wav", tmp / "out.wav"
    wavfile.write(src, FS, x.astype(np.float32))
    subprocess.run(
        # fmt: off
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
         "-af", f"rubberband=tempo={1.0 / ratio}", "-c:a", "pcm_f32le", str(dst)],
        # fmt: on
        check=True,
    )
    _, y = wavfile.read(dst)
    return y.astype(np.float64)


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
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        print("ffmpeg is required (with --enable-librubberband)", file=sys.stderr)
        return 1

    rows = []
    print(f"{'case':18} {'in':>8} {'rubberband':>12} {'exact':>8}")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for name, kind, freq in SIGNALS:
            x = signal(kind, freq)
            for ratio in RATIOS:
                y = rubberband(x, ratio, tmp)
                exact = round(len(x) * ratio)
                case = f"{name}_{str(ratio).replace('.', 'p')}"
                print(f"{case:18} {len(x):8d} {len(y):12d} {exact:8d}")
                rows.append((case, name, kind, freq, ratio, spectrum(y)))

    lines = [
        "// @generated by scripts/gen_stretch_golden.py — do not edit.",
        f"// numpy {np.__version__}, ffmpeg -af rubberband",
        "",
        "//! Time-stretch oracle vectors for `stretch_golden.rs`.",
        "//!",
        "//! Rubber Band, through ffmpeg, stretching each signal to `ratio` times its duration.",
        "//! Signals are described by parameters and rebuilt identically by `signal()` in the test;",
        "//! what is stored is the band spectrum of Rubber Band's whole output, normalized so its",
        "//! loudest band reads 0 dB.",
        "#![allow(dead_code)]",
        "",
        "/// Bins of the compared spectrum.",
        f"pub const BANDS: usize = {BANDS};",
        "/// The band compared, in Hz.",
        f"pub const BAND_LO: f32 = {f32(BAND_LO)};",
        "/// The top of the band compared, in Hz.",
        f"pub const BAND_HI: f32 = {f32(BAND_HI)};",
        "/// Seconds of input per case.",
        f"pub const SECONDS: f32 = {f32(SECONDS)};",
        "/// Sample rate of every case, Hz.",
        f"pub const FS: u32 = {FS};",
        "",
        "/// One oracle case.",
        "pub struct StretchCase {",
        "    /// Case name; signal and ratio, for failure messages.",
        "    pub name: &'static str,",
        "    /// Which shape `signal()` should build.",
        "    pub kind: &'static str,",
        "    /// Tone frequency, Hz (ignored by chord, sweep and noise).",
        "    pub freq: f32,",
        "    /// Output duration over input duration.",
        "    pub ratio: f32,",
        "    /// Rubber Band's band spectrum, dB relative to its own loudest band.",
        "    pub expected_db: &'static [f32],",
        "}",
        "",
    ]
    for case, _, _, _, _, spec in rows:
        values = ", ".join(f32(v) for v in spec)
        lines.append("#[rustfmt::skip]")
        lines.append(f"const {case.upper()}: &[f32] = &[{values}];")
    lines += [
        "",
        "/// Every oracle case, in the order the generator emitted them.",
        "pub const STRETCH_CASES: &[StretchCase] = &[",
    ]
    for case, _, kind, freq, ratio, _ in rows:
        lines.append(
            f'    StretchCase {{ name: "{case}", kind: "{kind}", freq: {f32(freq)}, '
            f"ratio: {f32(ratio)}, expected_db: {case.upper()} }},"
        )
    lines += ["];", ""]

    OUT.write_text("\n".join(lines))
    print(f"\nwrote {len(rows)} cases to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

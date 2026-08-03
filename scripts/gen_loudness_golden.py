#!/usr/bin/env python3
"""Generate the loudness oracle vectors for `crates/fluxion-ops/tests/loudness_golden.rs`.

ITU-R BS.1770 is a specification, so the honest way to check an implementation of it is against
*other* implementations, independently written:

    pyloudnorm  (MIT)  — integrated loudness
    ffmpeg      (LGPL) — integrated loudness, loudness range, true peak, via the ebur128 filter

Both are consulted here, offline, and the numbers they agree on are baked into a generated Rust
file of `const` arrays — so the Rust test needs neither Python nor ffmpeg to run, exactly like
`scripts/gen_golden.py` does for the SciPy filter-design vectors.

Signals are described by *parameters*, not samples, and regenerated identically on the Rust side.
The comparison tolerance is 0.1 LU, which is millions of times larger than any float difference
between the two generators, so this needs none of the bit-exactness the wasm fixtures do.

    python scripts/gen_loudness_golden.py

Conventions: frequencies in Hz, sample rate is `fs`.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    import pyloudnorm as pyln
except ImportError:  # pragma: no cover
    sys.exit("pyloudnorm is required: pip install pyloudnorm")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "crates/fluxion-ops/tests/loudness_golden_data.rs"
FS = 48_000

# The signal shapes the Rust side knows how to rebuild. Keep this in step with `signal()` in
# crates/fluxion-ops/tests/loudness_golden.rs — the case name is the join key, and the test asserts
# the two lists match, so a signal added on one side and not the other fails rather than silently
# skipping.
CASES = [
    # (name, kind, freq, amp, seconds, channels)
    ("sine_1k_-20dbfs_rms", "sine", 1000.0, 0.1 * np.sqrt(2), 10.0, 1),
    ("sine_1k_quiet", "sine", 1000.0, 0.02, 10.0, 1),
    ("sine_1k_loud", "sine", 1000.0, 0.8, 10.0, 1),
    # Across the K-curve: the shelf and the RLB high-pass have to be in the right places.
    ("sine_40hz", "sine", 40.0, 0.5, 10.0, 1),
    ("sine_100hz", "sine", 100.0, 0.5, 10.0, 1),
    ("sine_5khz", "sine", 5000.0, 0.5, 10.0, 1),
    ("sine_10khz", "sine", 10000.0, 0.5, 10.0, 1),
    # Broadband, where every part of the weighting contributes at once.
    ("noise", "noise", 0.0, 0.3, 10.0, 1),
    ("noise_stereo", "noise", 0.0, 0.3, 10.0, 2),
    # Gating: a loud passage followed by a quiet one, and by silence.
    ("stepped", "stepped", 1000.0, 0.5, 20.0, 1),
    ("tone_then_silence", "tone_then_silence", 1000.0, 0.5, 20.0, 1),
]


def lcg(n: int) -> np.ndarray:
    """The same integer LCG the Rust side uses, so both build the identical noise."""
    out = np.empty(n, dtype=np.float64)
    state = np.uint32(0x1234_5678)
    for i in range(n):
        state = np.uint32((np.uint64(state) * np.uint64(1_664_525) + np.uint64(1_013_904_223)) & np.uint64(0xFFFF_FFFF))
        out[i] = (int(state) >> 8) / 16_777_216.0 * 2.0 - 1.0
    return out


def signal(kind: str, freq: float, amp: float, seconds: float, channels: int) -> np.ndarray:
    """Build one case. Returns `(frames, channels)`, the layout soundfile and pyloudnorm want."""
    n = int(seconds * FS)
    t = np.arange(n) / FS
    if kind == "sine":
        mono = amp * np.sin(2 * np.pi * freq * t)
    elif kind == "noise":
        mono = amp * lcg(n)
    elif kind == "stepped":
        # 10 s at `amp`, then 10 s 20 dB down: a programme with real loudness range.
        half = n // 2
        mono = amp * np.sin(2 * np.pi * freq * t)
        mono[half:] *= 0.1
    elif kind == "tone_then_silence":
        half = n // 2
        mono = amp * np.sin(2 * np.pi * freq * t)
        mono[half:] = 0.0
    else:
        raise ValueError(f"unknown signal kind {kind!r}")

    if channels == 1:
        return mono.reshape(-1, 1)
    # A second channel at a different level, so the channel sum is not a trivial doubling.
    return np.stack([mono, mono * 0.5], axis=1)


def ffmpeg_ebur128(samples: np.ndarray) -> dict[str, float]:
    """Integrated loudness, loudness range and true peak, from ffmpeg's ebur128 filter."""
    import soundfile as sf

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "case.wav"
        sf.write(path, samples, FS, subtype="FLOAT")
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
             "-filter_complex", "ebur128=peak=true", "-f", "null", "-"],
            capture_output=True, text=True, check=True,
        ).stderr

    # The Summary block at the end is the integrated result; earlier lines are the running meter.
    summary = out[out.rindex("Summary:"):]

    def field(label: str) -> float:
        match = re.search(rf"{label}:\s*(-?[\d.]+|-inf)", summary)
        if not match:
            raise RuntimeError(f"no '{label}' in ffmpeg summary:\n{summary}")
        return float("-inf") if match.group(1) == "-inf" else float(match.group(1))

    return {"integrated": field("I"), "lra": field("LRA"), "true_peak": field("Peak")}


def version(package: str) -> str:
    """`pyloudnorm 0.2.0`, without assuming the module exposes __version__."""
    from importlib.metadata import PackageNotFoundError, version as dist_version

    try:
        return f"{package} {dist_version(package)}"
    except PackageNotFoundError:  # pragma: no cover
        return f"{package} (version unknown)"


def rust_float(v: float) -> str:
    """Shortest decimal literal that round-trips to the nearest f32.

    The same trick `scripts/gen_golden.py` uses: emitting full f64 precision for an f32 literal
    trips clippy's `excessive_precision`, and the extra digits mean nothing anyway.
    """
    if v == float("-inf"):
        return "f32::NEG_INFINITY"
    x = np.float32(v)
    text = repr(float(x))
    for digits in range(1, 10):
        candidate = "%.{}g".format(digits) % float(x)
        if np.float32(float(candidate)) == x:
            text = candidate
            break
    if "." not in text and "e" not in text and "inf" not in text:
        text += ".0"
    return f"{text}f32"


def main() -> int:
    if shutil.which("ffmpeg") is None:
        return int(print("ffmpeg is required", file=sys.stderr) or 1)

    rows = []
    print(f"{'case':24} {'pyloudnorm':>12} {'ffmpeg I':>10} {'spread':>8} {'ffmpeg LRA':>11} {'true peak':>10}")
    for name, kind, freq, amp, seconds, channels in CASES:
        samples = signal(kind, freq, amp, seconds, channels)
        meter = pyln.Meter(FS)
        pyln_i = meter.integrated_loudness(samples if channels > 1 else samples[:, 0])
        ff = ffmpeg_ebur128(samples)

        spread = abs(pyln_i - ff["integrated"]) if np.isfinite(pyln_i) else 0.0
        print(f"{name:24} {pyln_i:12.3f} {ff['integrated']:10.3f} {spread:8.3f} "
              f"{ff['lra']:11.3f} {ff['true_peak']:10.3f}")
        rows.append((name, kind, freq, amp, seconds, channels, pyln_i, ff))

    lines = [
        "// @generated by scripts/gen_loudness_golden.py — do not edit.",
        "//",
        "// Loudness oracle vectors: what pyloudnorm and ffmpeg's ebur128 independently measure for",
        "// each signal. Baked in as consts so the Rust test needs neither of them installed.",
        f"// {version('pyloudnorm')}, "
        f"{subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True).stdout.splitlines()[0]}",
        "",
        "//! Loudness oracle vectors for `loudness_golden.rs`.",
        "//!",
        "//! pyloudnorm and ffmpeg's `ebur128` are the independent references for ITU-R BS.1770.",
        "//! Signals are described by parameters and rebuilt identically by `signal()` in the test;",
        "//! the case name is the join key.",
        "#![allow(dead_code)]",
        "",
        "/// One oracle case: how to rebuild the signal, and what the two references measured.",
        "pub struct LoudnessCase {",
        "    /// Case name; the join key with the test's own list.",
        "    pub name: &'static str,",
        "    /// Which shape `signal()` should build.",
        "    pub kind: &'static str,",
        "    /// Tone frequency, Hz (ignored by the noise cases).",
        "    pub freq: f32,",
        "    /// Peak amplitude, linear.",
        "    pub amp: f32,",
        "    /// Duration, seconds.",
        "    pub seconds: f32,",
        "    /// Channel count.",
        "    pub channels: usize,",
        "    /// Integrated loudness, LUFS, per pyloudnorm.",
        "    pub pyloudnorm: f32,",
        "    /// Integrated loudness, LUFS, per ffmpeg ebur128.",
        "    pub ffmpeg: f32,",
        "    /// Loudness range, LU, per ffmpeg.",
        "    pub ffmpeg_lra: f32,",
        "    /// True peak, dBTP, per ffmpeg.",
        "    pub ffmpeg_true_peak: f32,",
        "}",
        "",
        "/// Every oracle case, in the order the generator emitted them.",
        "#[rustfmt::skip]",
        "pub const LOUDNESS_CASES: &[LoudnessCase] = &[",
    ]
    for name, kind, freq, amp, seconds, channels, pyln_i, ff in rows:
        lines.append(
            f'    LoudnessCase {{ name: "{name}", kind: "{kind}", freq: {rust_float(freq)}, '
            f"amp: {rust_float(amp)}, seconds: {rust_float(seconds)}, channels: {channels}, "
            f"pyloudnorm: {rust_float(pyln_i)}, ffmpeg: {rust_float(ff['integrated'])}, "
            f"ffmpeg_lra: {rust_float(ff['lra'])}, "
            f"ffmpeg_true_peak: {rust_float(ff['true_peak'])} }},"
        )
    lines += ["];", ""]

    OUT.write_text("\n".join(lines))
    print(f"\nwrote {len(rows)} cases to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

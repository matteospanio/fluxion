#!/usr/bin/env python3
"""Generate golden-vector oracle data for fluxion's filter designs.

SciPy is used as an *independent* reference oracle: for the Butterworth and Chebyshev
families we take SciPy's own `output='sos'` design, and for the RBJ biquads we evaluate the
Robert Bristow-Johnson Audio-EQ-Cookbook formulas directly here (SciPy has no RBJ designer).
For every case we emit (a) the reference SOS coefficient rows and (b) the first 64 samples of
the unit-impulse response computed with `scipy.signal.sosfilt` in float64.

The output is a generated Rust file, `crates/fluxion-ops/tests/golden_data.rs`, containing
`const` arrays only — so the Rust oracle test in `golden.rs` runs with **no** SciPy/Python
dependency. The case names here must stay in lock-step with `fluxion_cases()` in `golden.rs`;
that shared name is the join key the test asserts on.

Run:  python scripts/gen_golden.py      (needs scipy + numpy in the environment)

Conventions mirrored from AGENTS.md: frequencies in Hz, sample rate is `fs`.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import scipy
from scipy import signal

FS = 48_000
IR_LEN = 64


# --- RBJ Audio-EQ-Cookbook biquads (independent reference for the rbj_* family) -------------
# Transcribed from the canonical RBJ cookbook. fluxion uses the Q form `alpha = sin(w0)/(2Q)`
# and the shelf term `t = 2*sqrt(A)*alpha`; we reproduce exactly that here so the oracle pins
# fluxion's chosen cookbook variant (e.g. bandpass = constant 0 dB peak gain).

def _pre(freq, q, fs):
    w0 = 2.0 * math.pi * freq / fs
    return math.cos(w0), math.sin(w0), math.sin(w0) / (2.0 * q)


def _norm(b0, b1, b2, a0, a1, a2):
    """Normalize by a0, return a SciPy-style sos row [b0, b1, b2, 1, a1, a2]."""
    return [b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]


def rbj_peaking(freq, gain_db, q, fs):
    cw, _, alpha = _pre(freq, q, fs)
    a = 10.0 ** (gain_db / 40.0)
    return _norm(1 + alpha * a, -2 * cw, 1 - alpha * a,
                 1 + alpha / a, -2 * cw, 1 - alpha / a)


def rbj_low_shelf(freq, gain_db, q, fs):
    cw, _, alpha = _pre(freq, q, fs)
    a = 10.0 ** (gain_db / 40.0)
    t = 2.0 * math.sqrt(a) * alpha
    return _norm(a * ((a + 1) - (a - 1) * cw + t),
                 2 * a * ((a - 1) - (a + 1) * cw),
                 a * ((a + 1) - (a - 1) * cw - t),
                 (a + 1) + (a - 1) * cw + t,
                 -2 * ((a - 1) + (a + 1) * cw),
                 (a + 1) + (a - 1) * cw - t)


def rbj_high_shelf(freq, gain_db, q, fs):
    cw, _, alpha = _pre(freq, q, fs)
    a = 10.0 ** (gain_db / 40.0)
    t = 2.0 * math.sqrt(a) * alpha
    return _norm(a * ((a + 1) + (a - 1) * cw + t),
                 -2 * a * ((a - 1) + (a + 1) * cw),
                 a * ((a + 1) + (a - 1) * cw - t),
                 (a + 1) - (a - 1) * cw + t,
                 2 * ((a - 1) - (a + 1) * cw),
                 (a + 1) - (a - 1) * cw - t)


def rbj_notch(freq, q, fs):
    cw, _, alpha = _pre(freq, q, fs)
    return _norm(1.0, -2 * cw, 1.0, 1 + alpha, -2 * cw, 1 - alpha)


def rbj_bandpass(freq, q, fs):
    # BPF with constant 0 dB peak gain (matches fluxion::bandpass).
    cw, _, alpha = _pre(freq, q, fs)
    return _norm(alpha, 0.0, -alpha, 1 + alpha, -2 * cw, 1 - alpha)


def rbj_allpass(freq, q, fs):
    cw, _, alpha = _pre(freq, q, fs)
    return _norm(1 - alpha, -2 * cw, 1 + alpha, 1 + alpha, -2 * cw, 1 - alpha)


# --- the case matrix ------------------------------------------------------------------------
# Each case: (name, sos-rows) where sos-rows is an (n_sections, 6) float64 array with a0 == 1.
# `name` is the join key shared with golden.rs; keep the two in lock-step.

def cases():
    out = []

    # Butterworth low/high-pass, orders {2,4,6}, cutoffs {200,1000,8000} Hz.
    for order in (2, 4, 6):
        for fc in (200, 1000, 8000):
            out.append((f"butter_lp_o{order}_fc{fc}",
                        signal.butter(order, float(fc), "lowpass", fs=FS, output="sos")))
            out.append((f"butter_hp_o{order}_fc{fc}",
                        signal.butter(order, float(fc), "highpass", fs=FS, output="sos")))

    # Chebyshev I (equiripple passband), order 4, 1 dB ripple.
    for fc in (1000, 8000):
        out.append((f"cheby1_lp_o4_rp1_fc{fc}",
                    signal.cheby1(4, 1.0, float(fc), "lowpass", fs=FS, output="sos")))
        out.append((f"cheby1_hp_o4_rp1_fc{fc}",
                    signal.cheby1(4, 1.0, float(fc), "highpass", fs=FS, output="sos")))

    # Chebyshev II (equiripple stopband), order 4, 40 dB stopband; fc = stopband edge.
    for fc in (2000, 8000):
        out.append((f"cheby2_lp_o4_rs40_fc{fc}",
                    signal.cheby2(4, 40.0, float(fc), "lowpass", fs=FS, output="sos")))
        out.append((f"cheby2_hp_o4_rs40_fc{fc}",
                    signal.cheby2(4, 40.0, float(fc), "highpass", fs=FS, output="sos")))

    # RBJ cookbook biquads at 1 kHz (single-section cascades).
    out.append(("rbj_peaking_f1000_g6_q1", np.array([rbj_peaking(1000.0, 6.0, 1.0, FS)])))
    out.append(("rbj_lowshelf_f1000_g6_q0707", np.array([rbj_low_shelf(1000.0, 6.0, 0.707, FS)])))
    out.append(("rbj_highshelf_f1000_g6_q0707", np.array([rbj_high_shelf(1000.0, 6.0, 0.707, FS)])))
    out.append(("rbj_notch_f1000_q5", np.array([rbj_notch(1000.0, 5.0, FS)])))
    out.append(("rbj_bandpass_f1000_q1", np.array([rbj_bandpass(1000.0, 1.0, FS)])))
    out.append(("rbj_allpass_f1000_q0707", np.array([rbj_allpass(1000.0, 0.707, FS)])))

    return out


# --- Rust emission --------------------------------------------------------------------------

def f32(v):
    """Shortest decimal literal that round-trips to the f32 nearest `v`.

    Uses the fewest significant digits that still cast back to the same f32, so the emitted
    literals do not trip clippy's `excessive_precision`. Always carries a '.' or 'e' so Rust
    parses it as `f32`, never an integer literal.
    """
    x = np.float32(v)
    s = repr(float(x))
    for p in range(1, 10):
        cand = f"%.{p}g" % float(x)
        if np.float32(float(cand)) == x:
            s = cand
            break
    if not any(c in s for c in ".eE"):
        s += ".0"
    return s


def sos_rows_literal(sos):
    rows = []
    for row in sos:
        rows.append("[" + ", ".join(f32(c) for c in row) + "]")
    return "&[" + ", ".join(rows) + "]"


def ir_literal(ir):
    return "[" + ", ".join(f32(v) for v in ir) + "]"


def main():
    impulse = np.zeros(IR_LEN, dtype=np.float64)
    impulse[0] = 1.0

    entries = []
    for name, sos in cases():
        sos = np.asarray(sos, dtype=np.float64)
        ir = signal.sosfilt(sos, impulse)
        entries.append(
            f"    GoldenCase {{\n"
            f'        name: "{name}",\n'
            f"        sos: {sos_rows_literal(sos)},\n"
            f"        ir: &{ir_literal(ir)},\n"
            f"    }},"
        )

    body = "\n".join(entries)
    header = (
        "// generated by scripts/gen_golden.py — do not hand-edit\n"
        f"// scipy {scipy.__version__}, numpy {np.__version__}\n"
        "\n"
        "//! Golden filter-design oracle vectors for `golden.rs`.\n"
        "//!\n"
        "//! SciPy (Butterworth/Chebyshev) and the RBJ Audio-EQ-Cookbook (`rbj_*`) are the independent\n"
        "//! references. `sos` rows are `[b0, b1, b2, a0, a1, a2]` (a0 == 1); `ir` is the first 64\n"
        "//! samples of the unit-impulse response from `scipy.signal.sosfilt` (computed in f64, stored\n"
        "//! as f32). Case names are the join key with `fluxion_cases()` in `golden.rs`.\n"
        "#![allow(dead_code)]\n"
        "\n"
        "/// One golden filter case emitted by scripts/gen_golden.py.\n"
        "pub struct GoldenCase {\n"
        "    /// Case name — the join key shared with `fluxion_cases()` in golden.rs.\n"
        "    pub name: &'static str,\n"
        "    /// Reference SOS rows `[b0, b1, b2, a0, a1, a2]` with `a0 == 1` (documentation only).\n"
        "    pub sos: &'static [[f32; 6]],\n"
        "    /// First 64 samples of the reference unit-impulse response.\n"
        "    pub ir: &'static [f32; 64],\n"
        "}\n"
        "\n"
    )
    # rustfmt::skip keeps the wide generated coefficient/IR rows one-per-case (they round-trip to
    # this exact text) instead of being rewrapped, so `cargo fmt --check` stays clean on regen.
    text = (
        header
        + "/// Every golden case: SciPy/RBJ reference designs and their impulse responses.\n"
        + "#[rustfmt::skip]\n"
        + "pub const CASES: &[GoldenCase] = &[\n"
        + body
        + "\n];\n"
    )

    root = Path(__file__).resolve().parent.parent
    out_path = root / "crates" / "fluxion-ops" / "tests" / "golden_data.rs"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    print(f"wrote {out_path} ({len(text)} bytes, {len(entries)} cases)")


if __name__ == "__main__":
    main()

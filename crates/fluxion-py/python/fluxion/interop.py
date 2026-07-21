"""Import filter checkpoints from other differentiable-DSP frameworks into fluxion SOS coefficients.

Currently: FLAMO / DDSP-style **SISO biquad cascades**. FLAMO stores a filter as a set of named
per-section parameter tensors; this module reads such a checkpoint (a ``dict`` of tensors/arrays, or
a ``.safetensors`` file) and returns second-order-section (SOS) coefficients you can drop straight
into fluxion's differentiable filter path.

Supported layouts (auto-detected from the parameter names; trailing name after the last ``.`` is
used, so a module prefix like ``biquad.b`` is fine):

* **coefficients** — keys ``b`` and ``a``, each shape ``(N, 3)`` (numerator / denominator per
  section). This is the portable layout: export FLAMO's realised ``Biquad.b`` / ``Biquad.a``.
* **RBJ peaking params** — keys ``freq`` (Hz, ``(N,)``), ``gain_db`` (``(N,)``), ``Q`` (``(N,)``)
  and a scalar ``fs``. The per-section coefficients are replayed with the RBJ cookbook peaking-EQ
  formulas (the layout of FLAMO's SISO ``Biquad`` in peaking/PEQ mode).

For the **full import path** — FLAMO ``SOSFilter``/``SVF``/``Biquad`` raw-parameter layouts,
torchfx.ddsp learnable filters, ``.pt`` / ``.onnx`` / ``.safetensors`` artifacts, stability
certification, and writing a runnable ``.fxg`` — use :func:`import_checkpoint`, which replays the
exact source-library math in Rust (the same converter behind the ``fluxion import`` CLI verb).

``load_flamo_sos`` (below) is the original pure-NumPy first slice: realised ``b``/``a`` and RBJ
peaking tables only; an SVF cascade is rejected there (use :func:`import_checkpoint` instead).

``load_flamo_sos`` returns ``float32`` SOS with the denominator normalised to ``a0 = 1``:

* ``layout="sos"`` (default) → shape ``(N, 6)`` in scipy order ``[b0, b1, b2, a0, a1, a2]``.
* ``layout="fluxion"`` → shape ``(N, 5)`` ``[b0, b1, b2, a1, a2]`` (``a0`` dropped) — fluxion's own
  per-section layout.

Feeding the result into fluxion's differentiable SOS path::

    >>> import numpy as np, fluxion
    >>> from fluxion.interop import load_flamo_sos
    >>> sd = {"b": np.array([[0.3, 0.5, 0.2]]), "a": np.array([[1.0, -0.2, 0.05]])}
    >>> coeffs = load_flamo_sos(sd, layout="fluxion").ravel()   # flat [b0,b1,b2,a1,a2]·N
    >>> x = np.zeros(8, np.float32); x[0] = 1.0
    >>> _ = fluxion.sos_forward(x, coeffs)                      # or fluxion.torch.SosModule(coeffs)
"""

from __future__ import annotations

import os
from typing import Any, Mapping, NamedTuple, Union

import numpy as np

__all__ = ["ImportResult", "import_checkpoint", "load_flamo_sos"]

StateDict = Mapping[str, Any]


def load_flamo_sos(
    state_dict_or_path: Union[StateDict, str, "os.PathLike[str]"],
    *,
    layout: str = "sos",
) -> np.ndarray:
    """Load a FLAMO/DDSP biquad-cascade checkpoint as SOS coefficients (see the module docstring).

    ``state_dict_or_path`` is either a mapping of parameter name → tensor/array, or a path to a
    ``.safetensors`` file (``safetensors`` is imported lazily; install the ``interop`` extra).
    ``layout`` is ``"sos"`` → ``(N, 6)`` scipy order, or ``"fluxion"`` → ``(N, 5)``.
    """
    if layout not in ("sos", "fluxion"):
        raise ValueError(f"layout must be 'sos' or 'fluxion', got {layout!r}")

    sd = _load_state_dict(state_dict_or_path)
    # Index by trailing name so a module prefix (e.g. 'filters.b') resolves to 'b' — but refuse
    # ambiguity: per-section prefixes like 'sections.0.b'/'sections.1.b' would silently collapse
    # to the last section and load a wrong (single-section) cascade.
    named: dict = {}
    for key, val in sd.items():
        name = key.rsplit(".", 1)[-1]
        if name in named:
            raise ValueError(
                f"ambiguous checkpoint: multiple keys share the trailing name {name!r} "
                "(per-section prefixes?). Export one stacked tensor per parameter instead "
                "('b'/'a', each (N, 3))."
            )
        named[name] = val

    if "b" in named and "a" in named:
        ba = _sos_from_coeffs(named["b"], named["a"])
    elif {"freq", "gain_db", "Q"} <= named.keys():
        ba = _sos_from_rbj_peaking(named)
    elif _looks_like_svf(named):
        raise ValueError(
            "state-variable-filter (SVF) cascade import is not supported; its param→coeff map is "
            "not implemented. Export the realised biquad coefficients ('b'/'a', each (N, 3)) instead."
        )
    else:
        raise ValueError(
            "unrecognised FLAMO layout: expected biquad coefficients (keys 'b' and 'a', each "
            "(N, 3)) or RBJ peaking params (keys 'freq', 'gain_db', 'Q' and scalar 'fs'); got keys "
            f"{sorted(named)}"
        )

    if layout == "fluxion":
        ba = ba[:, [0, 1, 2, 4, 5]]  # drop a0 (== 1) → [b0, b1, b2, a1, a2]
    return np.ascontiguousarray(ba, dtype=np.float32)


def _load_state_dict(src: Union[StateDict, str, "os.PathLike[str]"]) -> StateDict:
    """Return a name→array mapping, loading a ``.safetensors`` file lazily if given a path."""
    if isinstance(src, (str, os.PathLike)):
        try:
            from safetensors.numpy import load_file
        except ImportError as exc:  # pragma: no cover - exercised only without the extra installed
            raise ImportError(
                "loading a .safetensors path needs the 'safetensors' package; install the interop "
                "extra: pip install 'fluxion[interop]'"
            ) from exc
        return load_file(os.fspath(src))
    return src


def _to_2d(v: Any, cols: int, name: str) -> np.ndarray:
    """Coerce a tensor/array to a float64 ``(N, cols)`` matrix (accepts torch tensors)."""
    arr = np.asarray(v.detach().cpu().numpy() if hasattr(v, "detach") else v, dtype=np.float64)
    if arr.ndim == 1 and arr.shape[0] == cols:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != cols:
        raise ValueError(f"{name!r} must have shape (N, {cols}); got {arr.shape}")
    return arr


def _to_1d(v: Any, name: str) -> np.ndarray:
    """Coerce a tensor/array to a flat float64 vector (accepts torch tensors)."""
    arr = np.asarray(v.detach().cpu().numpy() if hasattr(v, "detach") else v, dtype=np.float64)
    return arr.reshape(-1)


def _sos_from_coeffs(b: Any, a: Any) -> np.ndarray:
    """Assemble ``(N, 6)`` SOS from ``(N, 3)`` numerator/denominator, normalised to ``a0 = 1``."""
    b2, a2 = _to_2d(b, 3, "b"), _to_2d(a, 3, "a")
    if b2.shape[0] != a2.shape[0]:
        raise ValueError(f"'b' and 'a' must have the same N; got {b2.shape[0]} vs {a2.shape[0]}")
    a0 = a2[:, :1]
    if np.any(a0 == 0.0):
        raise ValueError("denominator leading coefficient a0 must be non-zero")
    return np.concatenate([b2 / a0, a2 / a0], axis=1)


def _sos_from_rbj_peaking(named: StateDict) -> np.ndarray:
    """Replay the RBJ cookbook peaking-EQ biquad for each section → ``(N, 6)`` SOS (``a0 = 1``)."""
    if "fs" not in named:
        raise ValueError("RBJ peaking layout requires a scalar 'fs' (sample rate in Hz)")
    fs = float(_to_1d(named["fs"], "fs")[0])
    freq = _to_1d(named["freq"], "freq")
    gain_db = _to_1d(named["gain_db"], "gain_db")
    q = _to_1d(named["Q"], "Q")
    if not (freq.shape == gain_db.shape == q.shape):
        raise ValueError(
            f"'freq', 'gain_db', 'Q' must share shape; got {freq.shape}, {gain_db.shape}, {q.shape}"
        )
    # RBJ Audio-EQ cookbook, peaking EQ.
    amp = np.power(10.0, gain_db / 40.0)
    w0 = 2.0 * np.pi * freq / fs
    alpha = np.sin(w0) / (2.0 * q)
    cos_w0 = np.cos(w0)
    b0 = 1.0 + alpha * amp
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * amp
    a0 = 1.0 + alpha / amp
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / amp
    sos = np.stack([b0, b1, b2, a0, a1, a2], axis=1)
    return sos / a0[:, None]  # normalise to a0 = 1


def _looks_like_svf(named: StateDict) -> bool:
    """Heuristic: a state-variable-filter cascade (TPT/Zavalishin mixing coefficients)."""
    keys = {k.lower() for k in named}
    svf_markers = {"m_lp", "m_bp", "m_hp", "mlp", "mbp", "mhp", "svf"}
    return bool(keys & svf_markers) or ("r" in keys and "f" in keys)


# ── full import path (Rust-backed): FLAMO / torchfx / .pt / .onnx → certified sections ─────────


class ImportResult(NamedTuple):
    """What :func:`import_checkpoint` produced."""

    sections: np.ndarray
    """``(n_sections, 5)`` float32 ``[b0, b1, b2, a1, a2]`` rows (``a0`` normalised to 1)."""
    verdict: str
    """Stability ladder verdict on the (possibly projected) sections: ``certified-stable``,
    ``marginally-stable``, ``indeterminate``, ``not-certified`` or ``unstable``."""
    margin: float
    """``1 - spectral_radius``: positive inside the stable region, ~0 on the boundary."""
    fs: int | None
    """Sample rate embedded in the artifact itself, when present (else the ``fs`` you passed)."""


def import_checkpoint(
    src: Union[StateDict, str, "os.PathLike[str]"],
    out: Union[str, "os.PathLike[str]", None] = None,
    *,
    fs: int | None = None,
    kind: str = "auto",
    svf_type: str = "general",
    biquad_type: str = "lowpass",
    eq_f_lo: float = 40.0,
    eq_f_hi: float = 16_000.0,
    eq_max_gain_db: float = 18.0,
    project_stable: bool = False,
    force: bool = False,
) -> ImportResult:
    """Import a DDSP checkpoint trained elsewhere into fluxion's certified freeze pipeline.

    ``src`` is a mapping of name → tensor/array, or a path: ``.safetensors`` (lazy ``safetensors``),
    ``.pt``/``.pth``/``.ckpt`` (lazy ``torch``; plain state-dicts, Lightning-style dicts with a
    ``state_dict`` key, and torchfx compiled ``.fxg`` artifacts ``{"version", "fs", "nodes"}`` are
    all understood), or ``.onnx`` (lazy ``onnx``; initializer tensors are read — note FLAMO models
    are generally *not* ONNX-exportable, prefer the state-dict path).

    Supported module layouts (mirrors ``fluxion-io::checkpoint``): FLAMO ``SOSFilter`` /
    ``SVF`` (every ``filter_type``, pass ``svf_type``) / ``Biquad`` (pass ``biquad_type``),
    realised ``b``/``a`` pairs, RBJ peaking tables, and torchfx.ddsp learnable filters (pass
    ``kind="ddsp-lowpass"``/``"ddsp-highpass"`` for the ambiguous log-cutoff pair; EQ band ranges
    via ``eq_*``). SISO only — MIMO banks and FIR taps are rejected.

    The sections are **stability-certified** (fluxion's verdict ladder, plan task E8);
    ``project_stable=True`` first clamps each section into the Jury stability triangle — use it
    for checkpoints trained without a stability constraint. If ``out`` is given, the sections are
    chained as raw ``biquad`` ops and written as a standard ``.fxg`` graph (refused if the
    certificate is not shippable, unless ``force=True``); the file splices into any CLI pipeline:
    ``fluxion in.wav model.fxg out.wav``.

    Raw coefficients do not retune: process at the checkpoint's training sample rate.
    """
    from . import _fluxion

    embedded_fs: int | None = None
    if isinstance(src, (str, os.PathLike)):
        path = os.fspath(src)
        if path.endswith(".safetensors"):
            sd: StateDict = _load_state_dict(path)
        elif path.endswith((".pt", ".pth", ".ckpt", ".fxg")):
            sd, embedded_fs = _load_torch_artifact(path)
        elif path.endswith(".onnx"):
            sd = _load_onnx_initializers(path)
        else:
            raise ValueError(f"unrecognised checkpoint extension: {path!r}")
    else:
        sd = src

    if embedded_fs is not None and fs is None:
        fs = embedded_fs

    sd_np = {k: _to_numpy(v) for k, v in sd.items()}
    sections, verdict, margin, artifact_fs = _fluxion.import_state_dict(
        sd_np,
        kind=kind,
        fs=fs,
        svf_type=svf_type,
        biquad_type=biquad_type,
        eq_f_lo=eq_f_lo,
        eq_f_hi=eq_f_hi,
        eq_max_gain_db=eq_max_gain_db,
        project_stable=project_stable,
    )
    final_fs = artifact_fs or fs

    if out is not None:
        _fluxion.save_biquad_fxg(os.fspath(out), sections, fs=final_fs or 48_000, force=force)

    return ImportResult(sections, verdict, margin, final_fs)


def _to_numpy(v: Any) -> np.ndarray:
    """Tensor/array → float32 numpy (accepts torch tensors without importing torch)."""
    arr = v.detach().cpu().numpy() if hasattr(v, "detach") else v
    return np.ascontiguousarray(np.asarray(arr), dtype=np.float32)


def _load_torch_artifact(path: str) -> tuple[StateDict, int | None]:
    """Read a ``.pt``-family artifact with lazy torch → (state-dict, embedded fs or None).

    Understands three shapes: a plain name→tensor state-dict, a Lightning-style wrapper with a
    ``state_dict`` key, and torchfx's compiled artifact ``{"version": 1, "fs": N, "nodes": [...]}``
    whose ``{"kind": "sos"}`` nodes carry realised ``[K, 6]`` coefficients (non-filter effect
    nodes are rejected — export those separately).
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without torch installed
        raise ImportError(
            "loading a .pt checkpoint needs 'torch' (only for parsing; conversion runs in Rust). "
            "Alternatively re-save the state-dict as .safetensors."
        ) from exc

    obj = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(obj, dict):
        raise ValueError(f"expected a dict-like checkpoint, got {type(obj).__name__}")

    # torchfx compiled artifact (the `torchfx compile` CLI output).
    if "nodes" in obj and "version" in obj:
        rows = []
        for node in obj["nodes"]:
            if node.get("kind") != "sos":
                raise ValueError(
                    "torchfx artifact contains a non-filter effect node "
                    f"({node.get('name', node.get('kind'))!r}); only SOS nodes are importable"
                )
            rows.append(np.asarray(node["sos"], dtype=np.float64).reshape(-1, 6))
        sos = np.concatenate(rows, axis=0)
        # Hand the realised rows to the converter as a FLAMO-style SOS param tensor.
        return {"param": sos}, int(obj.get("fs") or 0) or None

    if "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    return obj, None


def _load_onnx_initializers(path: str) -> StateDict:
    """Read an ``.onnx`` file's float initializer tensors with lazy ``onnx``."""
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError as exc:  # pragma: no cover - exercised only without onnx installed
        raise ImportError(
            "loading a .onnx file needs the 'onnx' package (only for parsing; conversion runs "
            "in Rust)."
        ) from exc

    model = onnx.load(path)
    out: dict[str, np.ndarray] = {}
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.dtype.kind == "f":
            out[init.name] = np.asarray(arr, dtype=np.float32)
    if not out:
        raise ValueError(f"no float initializer tensors found in {path!r}")
    return out

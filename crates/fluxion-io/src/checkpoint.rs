//! Checkpoint import — DDSP filters trained in other frameworks become fluxion SOS
//! sections (plan goal #6 / task J13, full slice; feature `checkpoint`).
//!
//! Reads a `.safetensors` state-dict and replays, in `f64`, the **exact**
//! parameter→coefficient math of the source library, yielding `[b0, b1, b2, a1, a2]`
//! sections (`a0` normalised to 1) ready for the certified freeze pipeline: chain
//! into [`fluxion_core::OpKind::Biquad`] ops, `certify_graph`, then `.fxg` / `FrozenSos`.
//! Certification deliberately stays with the caller (CLI `import`, fluxion-py) so
//! this module carries no DSP dependencies.
//!
//! Supported module layouts (SISO only; MIMO banks, FIR taps, and FLAMO's
//! GEQ/PEQ/AccurateGEQ are rejected with a clear error — out of the first slice):
//!
//! - **FLAMO `SOSFilter`** — `param` `[K, 6]`, rows `[b0,b1,b2,a0,a1,a2]`.
//! - **FLAMO `SVF`** — `param` `[5, K]`, raw `[f, R, mLP, mBP, mHP]`; every
//!   `filter_type` incl. the general (`None`) one. The replay matches the *code*
//!   in `flamo/processor/dsp.py` (`map_param2svf`), not its docstring: the
//!   shelving `R = 1` assignment there is dead (the following `else` overwrites
//!   it with `R = r`), and lowpass/highpass/bandpass mixing ignores `G`.
//! - **FLAMO `Biquad`** — `param` `[K, 2]` (lowpass/highpass `[fc, gain]`) or
//!   `[K, 3]` (bandpass `[fc1, fc2, gain]`), RBJ cookbook with fixed
//!   `Q = 1/sqrt(2)`; `fc` is stored as a fraction of pi, so the realised
//!   coefficients are sample-rate-free.
//! - **Realised coefficients** — tensors `b`/`a`, `[K, 3]` each (the portable
//!   export FLAMO users are steered to).
//! - **Named RBJ peaking table** — `freq`/`gain_db`/`Q` `[K]` + optional scalar
//!   `fs` (the layout `fluxion.interop.load_flamo_sos` already accepts).
//! - **torchfx.ddsp** — `LearnableLowpass`/`LearnableHighpass`
//!   (`_log_cutoff`/`_log_q`, ambiguous → needs an explicit [`Kind`]),
//!   `LearnablePeaking` (`_log_freq`/`_log_q`/`gain_db`) and
//!   `LearnableParametricEQ` (`_fc_raw`/`_q_raw`/`gain_db_raw` with the
//!   sigmoid/exp-clamp/tanh maps and `f_lo`/`f_hi`/`max_gain_db` ranges).
//!
//! Keys may carry arbitrary module-path prefixes (`model.eq._fc_raw`,
//! `_Shell__core.0.param`); tensors are grouped by prefix and each group is
//! converted independently, then concatenated in natural (numeric-aware) prefix
//! order — the `Series` cascade semantics of both source libraries.

use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

/// A named `f32` tensor from a checkpoint (`f64` sources are narrowed on read).
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    /// Dimensions as stored (before SISO squeezing).
    pub shape: Vec<usize>,
    /// Row-major data.
    pub data: Vec<f32>,
}

/// A parsed state-dict: key → tensor, insertion-ordered by key.
pub type StateDict = BTreeMap<String, Tensor>;

/// One imported SOS section, `[b0, b1, b2, a1, a2]` with `a0 = 1`.
pub type Section = [f32; 5];

/// What a successful import produced.
#[derive(Clone, Debug, PartialEq)]
pub struct Imported {
    /// The cascade, in source order.
    pub sections: Vec<Section>,
    /// Sample rate embedded in the artifact itself (only the RBJ peaking table
    /// carries one); `None` means the caller's `--fs` is the design rate.
    pub fs: Option<u32>,
    /// Keys that were recognised as non-filter parameters and skipped
    /// (e.g. a trailing `gain` scalar) — callers should surface these.
    pub skipped: Vec<String>,
}

/// Import failure, with enough context to fix the invocation.
#[derive(Clone, Debug, PartialEq)]
pub enum ImportError {
    /// File could not be read or is not a valid `.safetensors`.
    Read(String),
    /// A tensor had a dtype other than F32/F64.
    Dtype {
        /// Offending tensor key.
        key: String,
        /// The dtype found.
        dtype: String,
    },
    /// A tensor group could not be recognised as a supported module layout.
    Unrecognized {
        /// Module prefix of the group (empty for root).
        prefix: String,
        /// Leaf parameter names in the group.
        keys: Vec<String>,
    },
    /// The layout is recognised but outside the supported slice (MIMO, FIR, …).
    Unsupported {
        /// Offending tensor key.
        key: String,
        /// What made it unsupported.
        why: String,
    },
    /// The shape matches several layouts; pass an explicit [`Kind`].
    Ambiguous {
        /// Offending tensor key.
        key: String,
        /// The layouts it could be.
        candidates: String,
    },
    /// This source needs a sample rate and none was given.
    NeedsFs {
        /// The Hz-parameterised tensor that triggered the requirement.
        key: String,
    },
    /// The checkpoint contained no filter parameters at all.
    Empty,
}

impl fmt::Display for ImportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ImportError::Read(e) => write!(f, "reading checkpoint: {e}"),
            ImportError::Dtype { key, dtype } => {
                write!(
                    f,
                    "tensor '{key}': unsupported dtype {dtype} (want F32/F64)"
                )
            }
            ImportError::Unrecognized { prefix, keys } => write!(
                f,
                "module '{}': unrecognised parameter set {:?} — supported: FLAMO \
                 SOSFilter/SVF/Biquad ('param'), realised 'b'/'a', RBJ table \
                 ('freq'/'gain_db'/'Q'), torchfx.ddsp learnable filters",
                if prefix.is_empty() { "<root>" } else { prefix },
                keys
            ),
            ImportError::Unsupported { key, why } => write!(f, "tensor '{key}': {why}"),
            ImportError::Ambiguous { key, candidates } => write!(
                f,
                "tensor '{key}': shape matches several layouts ({candidates}); pass an \
                 explicit kind (e.g. --kind flamo-sos)"
            ),
            ImportError::NeedsFs { key } => {
                write!(
                    f,
                    "'{key}' is parameterised in Hz; a sample rate is required (--fs)"
                )
            }
            ImportError::Empty => write!(f, "checkpoint contains no filter parameters"),
        }
    }
}

impl std::error::Error for ImportError {}

/// Explicit source-module kind, when auto-detection is ambiguous or wrong.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Kind {
    /// Detect from key names and shapes (the default).
    #[default]
    Auto,
    /// FLAMO `SOSFilter`: `param` `[K, 6]`.
    FlamoSos,
    /// FLAMO `SVF`: `param` `[5, K]` (see [`SvfType`]).
    FlamoSvf,
    /// FLAMO `Biquad`: `param` `[K, 2|3]` (see [`FlamoBiquadType`]).
    FlamoBiquad,
    /// torchfx `LearnableLowpass`: `_log_cutoff`/`_log_q`.
    DdspLowpass,
    /// torchfx `LearnableHighpass`: `_log_cutoff`/`_log_q`.
    DdspHighpass,
}

impl Kind {
    /// Parse a CLI-style kind name.
    pub fn from_name(s: &str) -> Option<Self> {
        Some(match s {
            "auto" => Kind::Auto,
            "flamo-sos" => Kind::FlamoSos,
            "flamo-svf" => Kind::FlamoSvf,
            "flamo-biquad" => Kind::FlamoBiquad,
            "ddsp-lowpass" => Kind::DdspLowpass,
            "ddsp-highpass" => Kind::DdspHighpass,
            _ => return None,
        })
    }
}

/// FLAMO `SVF` `filter_type` (constructor argument, not stored in the checkpoint).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SvfType {
    /// `filter_type=None` — the general SVF (mixing = raw + bias). FLAMO's default.
    #[default]
    General,
    /// `"lowpass"`.
    Lowpass,
    /// `"highpass"`.
    Highpass,
    /// `"bandpass"`.
    Bandpass,
    /// `"lowshelf"`.
    Lowshelf,
    /// `"highshelf"`.
    Highshelf,
    /// `"peaking"`.
    Peaking,
    /// `"notch"`.
    Notch,
}

impl SvfType {
    /// Parse a CLI-style name.
    pub fn from_name(s: &str) -> Option<Self> {
        Some(match s {
            "general" => SvfType::General,
            "lowpass" => SvfType::Lowpass,
            "highpass" => SvfType::Highpass,
            "bandpass" => SvfType::Bandpass,
            "lowshelf" => SvfType::Lowshelf,
            "highshelf" => SvfType::Highshelf,
            "peaking" => SvfType::Peaking,
            "notch" => SvfType::Notch,
            _ => return None,
        })
    }
}

/// FLAMO `Biquad` `filter_type` (constructor argument, not stored in the checkpoint).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FlamoBiquadType {
    /// `"lowpass"` — FLAMO's default.
    #[default]
    Lowpass,
    /// `"highpass"`.
    Highpass,
}

impl FlamoBiquadType {
    /// Parse a CLI-style name.
    pub fn from_name(s: &str) -> Option<Self> {
        Some(match s {
            "lowpass" => FlamoBiquadType::Lowpass,
            "highpass" => FlamoBiquadType::Highpass,
            _ => return None,
        })
    }
}

/// torchfx `LearnableParametricEQ` band ranges (constructor args, not in the
/// checkpoint). Defaults mirror torchfx.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EqRanges {
    /// Lowest band centre (Hz).
    pub f_lo: f64,
    /// Highest band centre (Hz).
    pub f_hi: f64,
    /// tanh gain bound (dB).
    pub max_gain_db: f64,
}

impl Default for EqRanges {
    fn default() -> Self {
        Self {
            f_lo: 40.0,
            f_hi: 16_000.0,
            max_gain_db: 18.0,
        }
    }
}

/// Everything the converter needs besides the tensors.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ImportOptions {
    /// Explicit layout, or [`Kind::Auto`].
    pub kind: Kind,
    /// Design sample rate for Hz-parameterised sources (torchfx.ddsp, RBJ table
    /// without an embedded `fs`).
    pub fs: Option<u32>,
    /// FLAMO SVF `filter_type`.
    pub svf_type: SvfType,
    /// FLAMO Biquad `filter_type` (a 3-param row is always bandpass).
    pub biquad_type: FlamoBiquadType,
    /// torchfx EQ band ranges.
    pub eq: EqRanges,
}

// `EqRanges::default()` is not const-derivable; spell the Default impl out.
impl ImportOptions {
    /// Options with a design sample rate set.
    pub fn with_fs(fs: u32) -> Self {
        Self {
            fs: Some(fs),
            ..Self::default()
        }
    }
}

// ── safetensors parsing ───────────────────────────────────────────────────────

/// Read every F32/F64 tensor of a `.safetensors` file into a [`StateDict`].
pub fn load_safetensors(path: impl AsRef<Path>) -> Result<StateDict, ImportError> {
    let bytes = std::fs::read(path.as_ref()).map_err(|e| ImportError::Read(e.to_string()))?;
    state_dict_from_safetensors(&bytes)
}

/// Parse a `.safetensors` byte buffer into a [`StateDict`].
pub fn state_dict_from_safetensors(bytes: &[u8]) -> Result<StateDict, ImportError> {
    use safetensors::tensor::Dtype;
    let st = safetensors::SafeTensors::deserialize(bytes)
        .map_err(|e| ImportError::Read(e.to_string()))?;
    let mut out = StateDict::new();
    for (name, view) in st.tensors() {
        let data = match view.dtype() {
            Dtype::F32 => view
                .data()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            Dtype::F64 => view
                .data()
                .chunks_exact(8)
                .map(|c| {
                    f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                })
                .collect(),
            other => {
                return Err(ImportError::Dtype {
                    key: name,
                    dtype: format!("{other:?}"),
                });
            }
        };
        out.insert(
            name,
            Tensor {
                shape: view.shape().to_vec(),
                data,
            },
        );
    }
    Ok(out)
}

/// Convenience: parse + convert a `.safetensors` checkpoint in one call.
pub fn import_safetensors(
    path: impl AsRef<Path>,
    opts: &ImportOptions,
) -> Result<Imported, ImportError> {
    let sd = load_safetensors(path)?;
    sections_from_state_dict(&sd, opts)
}

// ── state-dict → sections ─────────────────────────────────────────────────────

/// Convert a parsed state-dict into SOS sections (the main entry point).
pub fn sections_from_state_dict(
    sd: &StateDict,
    opts: &ImportOptions,
) -> Result<Imported, ImportError> {
    // Group keys by module prefix (everything before the final '.').
    let mut groups: BTreeMap<&str, BTreeMap<&str, &Tensor>> = BTreeMap::new();
    for (key, tensor) in sd {
        let (prefix, leaf) = match key.rfind('.') {
            Some(i) => (&key[..i], &key[i + 1..]),
            None => ("", key.as_str()),
        };
        groups.entry(prefix).or_default().insert(leaf, tensor);
    }

    let mut prefixes: Vec<&str> = groups.keys().copied().collect();
    prefixes.sort_by(|a, b| natural_cmp(a, b));

    let mut sections = Vec::new();
    let mut fs_embedded = None;
    let mut skipped = Vec::new();
    for prefix in prefixes {
        let group = &groups[prefix];
        convert_group(
            prefix,
            group,
            opts,
            &mut sections,
            &mut fs_embedded,
            &mut skipped,
        )?;
    }
    if sections.is_empty() {
        return Err(ImportError::Empty);
    }
    Ok(Imported {
        sections,
        fs: fs_embedded,
        skipped,
    })
}

/// Convert one module's parameter group, appending its sections.
fn convert_group(
    prefix: &str,
    group: &BTreeMap<&str, &Tensor>,
    opts: &ImportOptions,
    sections: &mut Vec<Section>,
    fs_embedded: &mut Option<u32>,
    skipped: &mut Vec<String>,
) -> Result<(), ImportError> {
    let leaves: Vec<&str> = group.keys().copied().collect();
    let full = |leaf: &str| {
        if prefix.is_empty() {
            leaf.to_string()
        } else {
            format!("{prefix}.{leaf}")
        }
    };

    let has = |l: &str| group.contains_key(l);

    if has("param") {
        let t = group["param"];
        sections.extend(flamo_param_sections(&full("param"), t, opts)?);
        // A FLAMO module has exactly one tensor; anything else alongside is odd
        // but harmless — record it.
        for l in &leaves {
            if *l != "param" {
                skipped.push(full(l));
            }
        }
        return Ok(());
    }

    if has("b") && has("a") {
        sections.extend(ba_sections(&full("b"), group["b"], group["a"])?);
        return Ok(());
    }

    if has("freq") && has("gain_db") && has("Q") {
        let fs = match group.get("fs") {
            Some(t) if !t.data.is_empty() => Some(t.data[0] as f64),
            _ => opts.fs.map(f64::from),
        };
        let fs = fs.ok_or(ImportError::NeedsFs { key: full("freq") })?;
        if group.contains_key("fs") && fs_embedded.is_none() {
            *fs_embedded = Some(fs as u32);
        }
        sections.extend(rbj_table_sections(
            group["freq"],
            group["gain_db"],
            group["Q"],
            fs,
        ));
        return Ok(());
    }

    if has("_log_cutoff") && has("_log_q") {
        let fs = f64::from(opts.fs.ok_or(ImportError::NeedsFs {
            key: full("_log_cutoff"),
        })?);
        let cutoff = scalar(&full("_log_cutoff"), group["_log_cutoff"])?.exp();
        let q = scalar(&full("_log_q"), group["_log_q"])?.exp();
        let sec = match opts.kind {
            Kind::DdspLowpass => rbj_lowpass(cutoff, q, fs),
            Kind::DdspHighpass => rbj_highpass(cutoff, q, fs),
            _ => {
                return Err(ImportError::Ambiguous {
                    key: full("_log_cutoff"),
                    candidates: "torchfx LearnableLowpass and LearnableHighpass share the same \
                                 state-dict keys; pass --kind ddsp-lowpass or ddsp-highpass"
                        .into(),
                });
            }
        };
        sections.push(sec);
        return Ok(());
    }

    if has("_log_freq") && has("_log_q") && has("gain_db") {
        let fs = f64::from(opts.fs.ok_or(ImportError::NeedsFs {
            key: full("_log_freq"),
        })?);
        let freq = scalar(&full("_log_freq"), group["_log_freq"])?.exp();
        let q = scalar(&full("_log_q"), group["_log_q"])?.exp();
        let gain_db = scalar(&full("gain_db"), group["gain_db"])?;
        sections.push(rbj_peaking(freq, q, gain_db, fs));
        return Ok(());
    }

    if has("_fc_raw") && has("_q_raw") && has("gain_db_raw") {
        let fs = f64::from(opts.fs.ok_or(ImportError::NeedsFs {
            key: full("_fc_raw"),
        })?);
        sections.extend(ddsp_eq_sections(
            group["_fc_raw"],
            group["_q_raw"],
            group["gain_db_raw"],
            opts.eq,
            fs,
        ));
        return Ok(());
    }

    // Lone non-filter parameters we understand enough to skip.
    if leaves == ["gain"] || leaves == ["fs"] {
        skipped.push(full(leaves[0]));
        return Ok(());
    }

    Err(ImportError::Unrecognized {
        prefix: prefix.to_string(),
        keys: leaves.iter().map(|s| s.to_string()).collect(),
    })
}

// ── FLAMO `param` dispatch ────────────────────────────────────────────────────

/// Squeeze trailing size-1 dims beyond the first two (the SISO `(…, 1, 1)` tail);
/// any trailing dim > 1 is a MIMO bank → unsupported.
fn squeeze_siso(key: &str, t: &Tensor) -> Result<Vec<usize>, ImportError> {
    let mut shape = t.shape.clone();
    while shape.len() > 2 {
        match shape.last() {
            Some(1) => {
                shape.pop();
            }
            _ => {
                return Err(ImportError::Unsupported {
                    key: key.to_string(),
                    why: format!(
                        "shape {:?} is a MIMO bank; only SISO modules are importable \
                         (first slice)",
                        t.shape
                    ),
                });
            }
        }
    }
    Ok(shape)
}

/// Convert one FLAMO `param` tensor by shape (+ explicit kind override).
fn flamo_param_sections(
    key: &str,
    t: &Tensor,
    opts: &ImportOptions,
) -> Result<Vec<Section>, ImportError> {
    let shape = squeeze_siso(key, t)?;
    if shape.len() == 1 {
        return Err(ImportError::Unsupported {
            key: key.to_string(),
            why: format!(
                "1-D param of {} values looks like FIR taps (FLAMO Filter); FIR import \
                 is out of the first slice",
                shape[0]
            ),
        });
    }
    if shape.len() != 2 {
        return Err(ImportError::Unsupported {
            key: key.to_string(),
            why: format!("unsupported param rank {:?}", t.shape),
        });
    }
    let (r, c) = (shape[0], shape[1]);

    let sos_like = c == 6;
    let svf_like = r == 5;
    let biquad_like = c == 2 || c == 3;

    match opts.kind {
        Kind::FlamoSos if sos_like => return Ok(flamo_sos_rows(t, r)),
        Kind::FlamoSvf if svf_like => return Ok(svf_sections(t, c, opts.svf_type)),
        Kind::FlamoBiquad if biquad_like => {
            return Ok(flamo_biquad_sections(t, r, c, opts.biquad_type));
        }
        Kind::Auto => {}
        _ => {
            return Err(ImportError::Unsupported {
                key: key.to_string(),
                why: format!("shape {:?} does not fit kind {:?}", t.shape, opts.kind),
            });
        }
    }

    // Auto: unique shape match or a clear ambiguity error. One documented prior:
    // a first dim of exactly 5 is the SVF signature (its 5 raw params), so
    // `[5, K∈{2,3}]` resolves to SVF — a flamo Biquad with exactly 5 sections
    // must pass `--kind flamo-biquad`. Only `[5, 6]` (SVF K=6 vs SOS K=5) stays
    // a hard ambiguity.
    let biquad_like = biquad_like && !svf_like;
    let matches = usize::from(sos_like) + usize::from(svf_like) + usize::from(biquad_like);
    if matches > 1 {
        return Err(ImportError::Ambiguous {
            key: key.to_string(),
            candidates: format!(
                "shape {:?} fits {}{}{}",
                t.shape,
                if sos_like { "flamo-sos " } else { "" },
                if svf_like { "flamo-svf " } else { "" },
                if biquad_like { "flamo-biquad" } else { "" }
            ),
        });
    }
    if sos_like {
        Ok(flamo_sos_rows(t, r))
    } else if svf_like {
        Ok(svf_sections(t, c, opts.svf_type))
    } else if biquad_like {
        Ok(flamo_biquad_sections(t, r, c, opts.biquad_type))
    } else {
        Err(ImportError::Unsupported {
            key: key.to_string(),
            why: format!(
                "param shape {:?} matches no supported FLAMO module",
                t.shape
            ),
        })
    }
}

// ── conversion math (all in f64, cast to f32 at the very end) ────────────────

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Numerically-stable `ln(1 + e^x)`.
fn softplus(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// Normalise `[b0,b1,b2]`/`[a0,a1,a2]` to an `a0 = 1` section.
fn norm_section(b: [f64; 3], a: [f64; 3]) -> Section {
    let a0 = a[0];
    [
        (b[0] / a0) as f32,
        (b[1] / a0) as f32,
        (b[2] / a0) as f32,
        (a[1] / a0) as f32,
        (a[2] / a0) as f32,
    ]
}

/// FLAMO `SOSFilter` rows `[K,6]` `[b0,b1,b2,a0,a1,a2]` → normalised sections
/// (replaying its `normalize_a0=True` map).
fn flamo_sos_rows(t: &Tensor, k: usize) -> Vec<Section> {
    (0..k)
        .map(|i| {
            let r: Vec<f64> = t.data[i * 6..i * 6 + 6]
                .iter()
                .map(|&v| f64::from(v))
                .collect();
            norm_section([r[0], r[1], r[2]], [r[3], r[4], r[5]])
        })
        .collect()
}

/// Realised `b`/`a` tensors, `[K,3]` (or `[3]`) each.
fn ba_sections(key_b: &str, b: &Tensor, a: &Tensor) -> Result<Vec<Section>, ImportError> {
    let shape_b = squeeze_siso(key_b, b)?;
    let k = if shape_b.len() == 1 { 1 } else { shape_b[0] };
    if b.data.len() != k * 3 || a.data.len() != k * 3 {
        return Err(ImportError::Unsupported {
            key: key_b.to_string(),
            why: format!(
                "b/a must be [K,3] pairs; got b {:?} and a {:?}",
                b.shape, a.shape
            ),
        });
    }
    Ok((0..k)
        .map(|i| {
            let bb = &b.data[i * 3..i * 3 + 3];
            let aa = &a.data[i * 3..i * 3 + 3];
            norm_section(
                [f64::from(bb[0]), f64::from(bb[1]), f64::from(bb[2])],
                [f64::from(aa[0]), f64::from(aa[1]), f64::from(aa[2])],
            )
        })
        .collect())
}

/// FLAMO `SVF`: raw `[5,K]` `[f, R, mLP, mBP, mHP]` → sections.
///
/// Replays `map_param2svf` + `get_poly_coeff` exactly as coded (see module docs
/// for the two doc-vs-code discrepancies).
fn svf_sections(t: &Tensor, k: usize, ty: SvfType) -> Vec<Section> {
    let at = |p: usize, i: usize| f64::from(t.data[p * k + i]);
    (0..k)
        .map(|i| {
            let f = (std::f64::consts::PI * sigmoid(at(0, i)) * 0.5).tan();
            let r = softplus(at(1, i)) / std::f64::consts::LN_2;
            let g = 10f64.powf(-softplus(at(2, i)));
            let two_r_sqrt_g = 2.0 * r * g.sqrt();
            // `map_param2svf`: only peaking swaps the denominator R; the shelving
            // `R = 1` branch is dead code upstream.
            let (r_denom, m) = match ty {
                SvfType::General => (r, [at(2, i) + 1.0, at(3, i) + 2.0, at(4, i) + 1.0]),
                SvfType::Lowpass => (r, [1.0, 0.0, 0.0]),
                SvfType::Highpass => (r, [0.0, 0.0, 1.0]),
                SvfType::Bandpass => (r, [0.0, 1.0, 0.0]),
                SvfType::Lowshelf => (r, [1.0, two_r_sqrt_g, g]),
                SvfType::Highshelf => (r, [g, two_r_sqrt_g, 1.0]),
                SvfType::Peaking => (1.0 / r, [1.0, two_r_sqrt_g, 1.0]),
                SvfType::Notch => (r, [1.0, two_r_sqrt_g, 1.0]),
            };
            let (m_lp, m_bp, m_hp) = (m[0], m[1], m[2]);
            let b = [
                f * f * m_lp + f * m_bp + m_hp,
                2.0 * f * f * m_lp - 2.0 * m_hp,
                f * f * m_lp - f * m_bp + m_hp,
            ];
            let a = [
                f * f + 2.0 * r_denom * f + 1.0,
                2.0 * f * f - 2.0,
                f * f - 2.0 * r_denom * f + 1.0,
            ];
            norm_section(b, a)
        })
        .collect()
}

/// FLAMO `Biquad`: `[K,2]` lowpass/highpass or `[K,3]` bandpass, RBJ with fixed
/// `Q = 1/sqrt(2)`; `fc` raw is a clamped fraction of pi, gain raw is linear
/// (`20*log10(|g|)` clamped to ±60 dB) — so no sample rate enters.
fn flamo_biquad_sections(t: &Tensor, k: usize, p: usize, ty: FlamoBiquadType) -> Vec<Section> {
    let at = |i: usize, j: usize| f64::from(t.data[i * p + j]);
    let clamp_fc = |x: f64| x.clamp(0.0, 1.0);
    let gain_lin = |raw: f64| {
        let db = (20.0 * raw.abs().log10()).clamp(-60.0, 60.0);
        10f64.powf(db / 20.0)
    };
    (0..k)
        .map(|i| {
            if p == 3 {
                // bandpass [fc1, fc2, gain]
                let (w1, w2) = (
                    std::f64::consts::PI * clamp_fc(at(i, 0)),
                    std::f64::consts::PI * clamp_fc(at(i, 1)),
                );
                let g = gain_lin(at(i, 2));
                let wc = (w1 + w2) / 2.0;
                let bw = (clamp_fc(at(i, 1)) / clamp_fc(at(i, 0))).log2();
                let alpha = wc.sin() * (std::f64::consts::LN_2 / 2.0 * bw * wc / wc.sin()).sinh();
                norm_section(
                    [g * alpha, 0.0, -g * alpha],
                    [1.0 + alpha, -2.0 * wc.cos(), 1.0 - alpha],
                )
            } else {
                let wc = std::f64::consts::PI * clamp_fc(at(i, 0));
                let g = gain_lin(at(i, 1));
                let alpha = wc.sin() / 2.0 * std::f64::consts::SQRT_2;
                let c = wc.cos();
                let b = match ty {
                    FlamoBiquadType::Lowpass => {
                        [g * (1.0 - c) / 2.0, g * (1.0 - c), g * (1.0 - c) / 2.0]
                    }
                    FlamoBiquadType::Highpass => {
                        [g * (1.0 + c) / 2.0, -g * (1.0 + c), g * (1.0 + c) / 2.0]
                    }
                };
                norm_section(b, [1.0 + alpha, -2.0 * c, 1.0 - alpha])
            }
        })
        .collect()
}

/// Named RBJ peaking table (`freq`/`gain_db`/`Q` per band) at `fs`.
fn rbj_table_sections(freq: &Tensor, gain_db: &Tensor, q: &Tensor, fs: f64) -> Vec<Section> {
    let n = freq.data.len().min(gain_db.data.len()).min(q.data.len());
    (0..n)
        .map(|i| {
            rbj_peaking(
                f64::from(freq.data[i]),
                f64::from(q.data[i]),
                f64::from(gain_db.data[i]),
                fs,
            )
        })
        .collect()
}

/// torchfx `LearnableParametricEQ`: sigmoid log-spaced `fc`, exp-clamped `q`,
/// tanh-bounded gain, then RBJ peaking per band.
fn ddsp_eq_sections(
    fc_raw: &Tensor,
    q_raw: &Tensor,
    gain_raw: &Tensor,
    eq: EqRanges,
    fs: f64,
) -> Vec<Section> {
    let n = fc_raw
        .data
        .len()
        .min(q_raw.data.len())
        .min(gain_raw.data.len());
    (0..n)
        .map(|i| {
            let t = sigmoid(f64::from(fc_raw.data[i]));
            let fc = eq.f_lo * (eq.f_hi / eq.f_lo).powf(t);
            let q = (std::f64::consts::SQRT_2 * f64::from(q_raw.data[i]).exp()).clamp(0.3, 8.0);
            let gain_db = eq.max_gain_db * f64::from(gain_raw.data[i]).tanh();
            rbj_peaking(fc, q, gain_db, fs)
        })
        .collect()
}

/// RBJ peaking biquad (matches torchfx `rbj_peaking_sos` and the RBJ cookbook).
fn rbj_peaking(freq: f64, q: f64, gain_db: f64, fs: f64) -> Section {
    let a_gain = 10f64.powf(gain_db / 40.0);
    let w0 = 2.0 * std::f64::consts::PI * freq / fs;
    let (sin_w0, cos_w0) = w0.sin_cos();
    let alpha = sin_w0 / (2.0 * q);
    norm_section(
        [1.0 + alpha * a_gain, -2.0 * cos_w0, 1.0 - alpha * a_gain],
        [1.0 + alpha / a_gain, -2.0 * cos_w0, 1.0 - alpha / a_gain],
    )
}

/// RBJ low-pass biquad (matches torchfx `rbj_lowpass_sos`).
fn rbj_lowpass(cutoff: f64, q: f64, fs: f64) -> Section {
    let w0 = 2.0 * std::f64::consts::PI * cutoff / fs;
    let (sin_w0, cos_w0) = w0.sin_cos();
    let alpha = sin_w0 / (2.0 * q);
    let b1 = 1.0 - cos_w0;
    norm_section(
        [b1 / 2.0, b1, b1 / 2.0],
        [1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha],
    )
}

/// RBJ high-pass biquad (matches torchfx `rbj_highpass_sos`).
fn rbj_highpass(cutoff: f64, q: f64, fs: f64) -> Section {
    let w0 = 2.0 * std::f64::consts::PI * cutoff / fs;
    let (sin_w0, cos_w0) = w0.sin_cos();
    let alpha = sin_w0 / (2.0 * q);
    let b0 = (1.0 + cos_w0) / 2.0;
    norm_section(
        [b0, -(1.0 + cos_w0), b0],
        [1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha],
    )
}

fn scalar(key: &str, t: &Tensor) -> Result<f64, ImportError> {
    if t.data.len() != 1 {
        return Err(ImportError::Unsupported {
            key: key.to_string(),
            why: format!("expected a scalar, got shape {:?}", t.shape),
        });
    }
    Ok(f64::from(t.data[0]))
}

/// Natural (numeric-aware) ordering so `core.2` sorts before `core.10`.
fn natural_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    let (ab, bb) = (a.as_bytes(), b.as_bytes());
    let (mut i, mut j) = (0, 0);
    while i < ab.len() && j < bb.len() {
        if ab[i].is_ascii_digit() && bb[j].is_ascii_digit() {
            let (si, sj) = (i, j);
            while i < ab.len() && ab[i].is_ascii_digit() {
                i += 1;
            }
            while j < bb.len() && bb[j].is_ascii_digit() {
                j += 1;
            }
            let (na, nb) = (&a[si..i], &b[sj..j]);
            let cmp = na
                .trim_start_matches('0')
                .len()
                .cmp(&nb.trim_start_matches('0').len())
                .then_with(|| na.trim_start_matches('0').cmp(nb.trim_start_matches('0')));
            if cmp != std::cmp::Ordering::Equal {
                return cmp;
            }
        } else {
            let cmp = ab[i].cmp(&bb[j]);
            if cmp != std::cmp::Ordering::Equal {
                return cmp;
            }
            i += 1;
            j += 1;
        }
    }
    ab.len().cmp(&bb.len())
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn t(shape: &[usize], data: &[f32]) -> Tensor {
        assert_eq!(shape.iter().product::<usize>(), data.len());
        Tensor {
            shape: shape.to_vec(),
            data: data.to_vec(),
        }
    }

    fn assert_close(got: &[Section], want: &[[f64; 5]]) {
        assert_eq!(got.len(), want.len(), "section count");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            for (j, (gv, wv)) in g.iter().zip(w).enumerate() {
                assert!(
                    (f64::from(*gv) - wv).abs() < 1e-6,
                    "section {i} coeff {j}: got {gv}, want {wv}"
                );
            }
        }
    }

    // Golden values computed independently in Python (f64) from the FLAMO /
    // torchfx source formulas.
    const SVF_RAW: [f32; 5] = [0.3, -0.5, 0.2, -0.1, 0.4];

    #[test]
    fn svf_general_matches_flamo() {
        let raw = t(&[5, 1], &SVF_RAW);
        assert_close(
            &svf_sections(&raw, 1, SvfType::General),
            &[[
                1.32163882,
                0.241711454,
                0.211782332,
                0.278313693,
                0.200963478,
            ]],
        );
    }

    #[test]
    fn svf_typed_variants_match_flamo() {
        let raw = t(&[5, 1], &SVF_RAW);
        assert_close(
            &svf_sections(&raw, 1, SvfType::Peaking),
            &[[
                0.52241369,
                0.191340783,
                0.303249212,
                0.191340783,
                -0.174337098,
            ]],
        );
        assert_close(
            &svf_sections(&raw, 1, SvfType::Lowpass),
            &[[
                0.369819293,
                0.739638586,
                0.369819293,
                0.278313693,
                0.200963478,
            ]],
        );
        // Shelving replays the dead-code quirk: R stays r (not 1).
        assert_close(
            &svf_sections(&raw, 1, SvfType::Lowshelf),
            &[[
                0.565926077,
                0.666209517,
                0.247141578,
                0.278313693,
                0.200963478,
            ]],
        );
    }

    #[test]
    fn flamo_biquad_types_match_flamo() {
        assert_close(
            &flamo_biquad_sections(&t(&[1, 2], &[0.25, 2.0]), 1, 2, FlamoBiquadType::Lowpass),
            &[[
                0.195262146,
                0.390524292,
                0.195262146,
                -0.942809042,
                0.333333333,
            ]],
        );
        assert_close(
            &flamo_biquad_sections(&t(&[1, 2], &[0.5, 1.0]), 1, 2, FlamoBiquadType::Highpass),
            &[[0.292893219, -0.585786438, 0.292893219, 0.0, 0.171572875]],
        );
        // 3-param rows are always bandpass, regardless of the type option.
        assert_close(
            &flamo_biquad_sections(
                &t(&[1, 3], &[0.2, 0.4, 1.0]),
                1,
                3,
                FlamoBiquadType::Lowpass,
            ),
            &[[0.251264319, 0.0, -0.251264319, -0.880191582, 0.497471361]],
        );
    }

    #[test]
    fn ddsp_filters_match_torchfx() {
        assert_close(
            &[rbj_lowpass(1000.0, 0.707, 48_000.0)],
            &[[
                0.00391607668,
                0.00783215337,
                0.00391607668,
                -1.81531792,
                0.830982222,
            ]],
        );
        assert_close(
            &[rbj_highpass(2000.0, 1.2, 44_100.0)],
            &[[
                0.877102878,
                -1.75420576,
                0.877102878,
                -1.71810898,
                0.790302526,
            ]],
        );
        assert_close(
            &[rbj_peaking(1000.0, 1.0, 6.0, 48_000.0)],
            &[[
                1.04395309,
                -1.89532072,
                0.867722285,
                -1.89532072,
                0.911675372,
            ]],
        );
        // EQ band with raw params (0, 0, 0.5) under default ranges.
        let secs = ddsp_eq_sections(
            &t(&[1], &[0.0]),
            &t(&[1], &[0.0]),
            &t(&[1], &[0.5]),
            EqRanges::default(),
            48_000.0,
        );
        assert_close(
            &secs,
            &[[1.03593681, -1.9445243, 0.919298471, -1.9445243, 0.95523528]],
        );
    }

    #[test]
    fn sos_rows_normalize_a0() {
        let raw = t(&[1, 6], &[2.0, 1.0, 0.5, 2.0, -0.6, 0.2]);
        assert_close(&flamo_sos_rows(&raw, 1), &[[1.0, 0.5, 0.25, -0.3, 0.1]]);
    }

    #[test]
    fn state_dict_dispatch_cascades_in_natural_order() {
        // A FLAMO Series: _Shell__core.{0,2,10}.param — three SOSFilter sections
        // with distinct b0 so ordering is observable; 10 must sort after 2.
        let ident = |b0: f32| [b0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let mut sd = StateDict::new();
        sd.insert("_Shell__core.0.param".into(), t(&[1, 6, 1, 1], &ident(1.0)));
        sd.insert(
            "_Shell__core.10.param".into(),
            t(&[1, 6, 1, 1], &ident(3.0)),
        );
        sd.insert("_Shell__core.2.param".into(), t(&[1, 6, 1, 1], &ident(2.0)));
        let got = sections_from_state_dict(&sd, &ImportOptions::default()).unwrap();
        let b0s: Vec<f32> = got.sections.iter().map(|s| s[0]).collect();
        assert_eq!(b0s, vec![1.0, 2.0, 3.0]);
        assert!(got.fs.is_none() && got.skipped.is_empty());
    }

    #[test]
    fn ba_and_rbj_table_layouts() {
        let mut sd = StateDict::new();
        sd.insert("b".into(), t(&[1, 3], &[2.0, 1.0, 0.5]));
        sd.insert("a".into(), t(&[1, 3], &[2.0, -0.6, 0.2]));
        let got = sections_from_state_dict(&sd, &ImportOptions::default()).unwrap();
        assert_close(&got.sections, &[[1.0, 0.5, 0.25, -0.3, 0.1]]);

        let mut sd = StateDict::new();
        sd.insert("eq.freq".into(), t(&[1], &[1000.0]));
        sd.insert("eq.gain_db".into(), t(&[1], &[6.0]));
        sd.insert("eq.Q".into(), t(&[1], &[1.0]));
        sd.insert("eq.fs".into(), t(&[1], &[48_000.0]));
        let got = sections_from_state_dict(&sd, &ImportOptions::default()).unwrap();
        assert_eq!(got.fs, Some(48_000));
        assert_close(
            &got.sections,
            &[[
                1.04395309,
                -1.89532072,
                0.867722285,
                -1.89532072,
                0.911675372,
            ]],
        );
    }

    #[test]
    fn errors_are_actionable() {
        // MIMO bank rejected.
        let mut sd = StateDict::new();
        sd.insert("param".into(), t(&[1, 6, 2, 2], &[0.0; 24]));
        assert!(matches!(
            sections_from_state_dict(&sd, &ImportOptions::default()),
            Err(ImportError::Unsupported { .. })
        ));

        // FIR taps rejected.
        let mut sd = StateDict::new();
        sd.insert("param".into(), t(&[8, 1, 1], &[0.0; 8]));
        assert!(matches!(
            sections_from_state_dict(&sd, &ImportOptions::default()),
            Err(ImportError::Unsupported { .. })
        ));

        // (5,6) is ambiguous without a kind…
        let mut sd = StateDict::new();
        sd.insert("param".into(), t(&[5, 6], &[0.1; 30]));
        assert!(matches!(
            sections_from_state_dict(&sd, &ImportOptions::default()),
            Err(ImportError::Ambiguous { .. })
        ));
        // …and resolvable with one.
        let opts = ImportOptions {
            kind: Kind::FlamoSos,
            ..ImportOptions::default()
        };
        assert_eq!(
            sections_from_state_dict(&sd, &opts).unwrap().sections.len(),
            5
        );

        // ddsp lowpass/highpass keys demand an explicit kind.
        let mut sd = StateDict::new();
        sd.insert("_log_cutoff".into(), t(&[], &[6.9]));
        sd.insert("_log_q".into(), t(&[], &[0.0]));
        let opts = ImportOptions::with_fs(48_000);
        assert!(matches!(
            sections_from_state_dict(&sd, &opts),
            Err(ImportError::Ambiguous { .. })
        ));
        let opts = ImportOptions {
            kind: Kind::DdspLowpass,
            ..ImportOptions::with_fs(48_000)
        };
        assert_eq!(
            sections_from_state_dict(&sd, &opts).unwrap().sections.len(),
            1
        );

        // Hz-parameterised sources without fs.
        let opts = ImportOptions {
            kind: Kind::DdspLowpass,
            ..ImportOptions::default()
        };
        assert!(matches!(
            sections_from_state_dict(&sd, &opts),
            Err(ImportError::NeedsFs { .. })
        ));

        // Unrecognised group lists its keys.
        let mut sd = StateDict::new();
        sd.insert("mystery.weight".into(), t(&[2], &[0.0, 1.0]));
        assert!(matches!(
            sections_from_state_dict(&sd, &ImportOptions::default()),
            Err(ImportError::Unrecognized { .. })
        ));

        // Empty checkpoint.
        assert_eq!(
            sections_from_state_dict(&StateDict::new(), &ImportOptions::default()),
            Err(ImportError::Empty)
        );
    }

    #[test]
    fn safetensors_round_trip() {
        // Serialize a state-dict with the official crate, read it back through
        // the importer.
        use safetensors::tensor::{Dtype, TensorView};
        let data: Vec<u8> = [2.0f32, 1.0, 0.5, 2.0, -0.6, 0.2]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let views = vec![(
            "core.param".to_string(),
            TensorView::new(Dtype::F32, vec![1, 6, 1, 1], &data).unwrap(),
        )];
        let bytes = safetensors::serialize(views, None).unwrap();
        let sd = state_dict_from_safetensors(&bytes).unwrap();
        let got = sections_from_state_dict(&sd, &ImportOptions::default()).unwrap();
        assert_close(&got.sections, &[[1.0, 0.5, 0.25, -0.3, 0.1]]);
    }

    #[test]
    fn natural_order_sorts_numerically() {
        let mut v = vec!["c.10", "c.2", "c.1", ""];
        v.sort_by(|a, b| natural_cmp(a, b));
        assert_eq!(v, vec!["", "c.1", "c.2", "c.10"]);
    }
}

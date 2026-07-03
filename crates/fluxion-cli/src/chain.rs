//! Effect-chain and stage-pipeline parsing for the CLI.
//!
//! A `fluxion` pipeline is a left-to-right sequence of **stages**. Adjacent [`OpKind`] effects
//! (and spliced `.fxg` graphs) coalesce into a single [`Graph`] stage — one filter pass; a
//! **geometry stage** (`trim`, `pad`, `rate`, `speed`, `repeat`, `silence`, `channels`, `remix`)
//! changes the frame count, sample rate, or channel layout and runs between graph passes.
//!
//! Grammar (all long-flag style, SI/`k` numbers, seconds/Hz/dB units):
//!
//! ```text
//! <effect|stage> [--flag value | --bool]... <effect|stage> ...
//! ```
//!
//! * Number values accept a `k`/`K` suffix (`--cutoff 1k` = 1000 Hz); see [`parse_value`].
//! * `gain --db -6` and `normalize --db -1` take decibels and convert to the linear param
//!   (`--gain` / peak still accept a linear value directly).
//! * `fir --taps 0.5,0.3,0.2` supplies the whole (variadic) tap vector.

use fluxion::{Graph, Op, OpKind, Signal, fxg, process, transform};

/// Parse a numeric CLI value: a plain float, optionally with a `k`/`K` SI suffix (`×1000`).
///
/// Covered by `parse_value_handles_si_and_floats` below.
pub(crate) fn parse_value(s: &str) -> Result<f32, String> {
    let t = s.trim();
    let (body, mult) = match t.chars().next_back() {
        Some('k' | 'K') => (&t[..t.len() - 1], 1_000.0_f32),
        _ => (t, 1.0),
    };
    body.parse::<f32>()
        .map(|v| v * mult)
        .map_err(|_| format!("invalid number: '{s}'"))
}

/// Convert decibels to a linear amplitude ratio (`10^(dB/20)`).
fn db_to_linear(db: f32) -> f32 {
    10f32.powf(db / 20.0)
}

// --- geometry-stage catalog ------------------------------------------------------------------

/// One flag of a geometry stage, for the `effects` listing and for validation.
pub(crate) struct StageFlag {
    /// Flag name without the leading `--`.
    pub flag: &'static str,
    /// Value kind / unit shown in help (`"s"`, `"Hz"`, `"int"`, `"dB"`, `"spec"`, `"flag"`).
    pub kind: &'static str,
    /// Default or requirement note.
    pub note: &'static str,
}

/// A geometry stage: name, one-line summary, and its flags.
pub(crate) struct StageDoc {
    /// Stage name (the pipeline token).
    pub name: &'static str,
    /// One-line description for the `effects` listing.
    pub summary: &'static str,
    /// The stage's flags.
    pub flags: &'static [StageFlag],
}

/// Every geometry stage — the single source of truth for the parser's stage-name/flag validation
/// and the `effects` help listing.
pub(crate) static STAGES: &[StageDoc] = &[
    StageDoc {
        name: "trim",
        summary: "keep a [start, start+len) window (or [start, end)); drops the rest",
        flags: &[
            StageFlag {
                flag: "start",
                kind: "s",
                note: "default 0",
            },
            StageFlag {
                flag: "len",
                kind: "s",
                note: "default: to end",
            },
            StageFlag {
                flag: "end",
                kind: "s",
                note: "alternative to --len",
            },
        ],
    },
    StageDoc {
        name: "pad",
        summary: "prepend/append silence",
        flags: &[
            StageFlag {
                flag: "start",
                kind: "s",
                note: "default 0",
            },
            StageFlag {
                flag: "end",
                kind: "s",
                note: "default 0",
            },
        ],
    },
    StageDoc {
        name: "rate",
        summary: "resample to a new sample rate (windowed-sinc SRC)",
        flags: &[StageFlag {
            flag: "fs",
            kind: "Hz",
            note: "required",
        }],
    },
    StageDoc {
        name: "speed",
        summary: "change speed (pitch+tempo together); keeps fs",
        flags: &[StageFlag {
            flag: "factor",
            kind: "lin",
            note: "required; >1 faster",
        }],
    },
    StageDoc {
        name: "repeat",
        summary: "concatenate the signal with itself count times",
        flags: &[StageFlag {
            flag: "count",
            kind: "int",
            note: "required",
        }],
    },
    StageDoc {
        name: "silence",
        summary: "trim leading/trailing near-silence",
        flags: &[
            StageFlag {
                flag: "threshold-db",
                kind: "dB",
                note: "default -60",
            },
            StageFlag {
                flag: "min",
                kind: "s",
                note: "guard band, default 0",
            },
            StageFlag {
                flag: "leading",
                kind: "flag",
                note: "trim the front",
            },
            StageFlag {
                flag: "trailing",
                kind: "flag",
                note: "trim the tail",
            },
        ],
    },
    StageDoc {
        name: "channels",
        summary: "up/down-mix to count channels (energy-preserving)",
        flags: &[StageFlag {
            flag: "count",
            kind: "int",
            note: "required",
        }],
    },
    StageDoc {
        name: "remix",
        summary: "rebuild channels from a 1-based map, e.g. \"1,2\" or \"1-2,3\"",
        flags: &[StageFlag {
            flag: "map",
            kind: "spec",
            note: "required",
        }],
    },
];

/// The stage doc for `name`, if `name` is a geometry stage.
pub(crate) fn stage_doc(name: &str) -> Option<&'static StageDoc> {
    STAGES.iter().find(|s| s.name == name)
}

/// The valueless (boolean) flags of a stage.
fn stage_bool_flags(name: &str) -> Vec<&'static str> {
    stage_doc(name)
        .map(|d| {
            d.flags
                .iter()
                .filter(|f| f.kind == "flag")
                .map(|f| f.flag)
                .collect()
        })
        .unwrap_or_default()
}

// --- stage model -----------------------------------------------------------------------------

/// One pipeline stage: a coalesced graph pass or a geometry transform.
pub(crate) enum Stage {
    /// A filter pass — the fused series graph of adjacent effects.
    Graph(Graph),
    /// Keep `[start, start+len)`; `len`/`end` unset means "to the end".
    Trim {
        start: f32,
        len: Option<f32>,
        end: Option<f32>,
    },
    /// Prepend `start` and append `end` seconds of silence.
    Pad { start: f32, end: f32 },
    /// Resample to `fs` Hz.
    Rate { fs: u32 },
    /// Change speed (pitch+tempo) by `factor`.
    Speed { factor: f32 },
    /// Repeat the signal `count` times.
    Repeat { count: usize },
    /// Trim near-silence below `threshold_db`, keeping `min_s` of guard band.
    Silence {
        threshold_db: f32,
        min_s: f32,
        leading: bool,
        trailing: bool,
    },
    /// Up/down-mix to `count` channels.
    Channels { count: usize },
    /// Rebuild channels from `spec` (per output channel: `(input, weight)` list).
    Remix { spec: Vec<Vec<(usize, f32)>> },
}

impl Stage {
    /// Apply this stage to a signal, producing the next signal.
    pub(crate) fn apply(&self, sig: Signal) -> Signal {
        match self {
            Stage::Graph(g) => process(g, &sig),
            Stage::Trim { start, len, end } => {
                let len_s = match (len, end) {
                    (Some(l), _) => *l,
                    (None, Some(e)) => (e - start).max(0.0),
                    (None, None) => {
                        let dur = sig.frames() as f32 / sig.fs.max(1) as f32;
                        (dur - start).max(0.0)
                    }
                };
                transform::trim(&sig, *start, len_s)
            }
            Stage::Pad { start, end } => transform::pad(&sig, *start, *end),
            Stage::Rate { fs } => transform::resample(&sig, *fs),
            Stage::Speed { factor } => transform::speed(&sig, *factor),
            Stage::Repeat { count } => transform::repeat(&sig, *count),
            Stage::Silence {
                threshold_db,
                min_s,
                leading,
                trailing,
            } => transform::silence_trim(&sig, *threshold_db, *min_s, *leading, *trailing),
            Stage::Channels { count } => transform::channels(&sig, *count),
            Stage::Remix { spec } => transform::remix(&sig, spec),
        }
    }
}

/// Run a stage pipeline left to right.
pub(crate) fn run_stages(stages: &[Stage], mut signal: Signal) -> Signal {
    for stage in stages {
        signal = stage.apply(signal);
    }
    signal
}

// --- parsing ---------------------------------------------------------------------------------

/// Parse a pipeline into an ordered list of stages (item 1: the stage-sequence model).
pub(crate) fn parse_stages(tokens: &[String]) -> Result<Vec<Stage>, String> {
    let mut stages = Vec::new();
    let mut graph = Graph::Id;
    let mut i = 0;
    while i < tokens.len() {
        let name = tokens[i].as_str();
        if stage_doc(name).is_some() {
            if !graph.is_empty() {
                stages.push(Stage::Graph(std::mem::replace(&mut graph, Graph::Id)));
            }
            let (stage, next) = parse_stage(name, tokens, i + 1)?;
            stages.push(stage);
            i = next;
        } else if let Some(kind) = OpKind::from_name(name) {
            let (node, next) = parse_op(kind, name, tokens, i + 1)?;
            graph = if graph.is_empty() { node } else { graph | node };
            i = next;
        } else if name.ends_with(".fxg") {
            let node = fxg::load(name).map_err(|e| format!("loading '{name}': {e}"))?;
            graph = if graph.is_empty() { node } else { graph | node };
            i += 1;
        } else {
            return Err(format!("unknown effect or stage '{name}'"));
        }
    }
    if !graph.is_empty() {
        stages.push(Stage::Graph(graph));
    }
    Ok(stages)
}

/// Parse a SoX-style effect chain into a single series [`Graph`], rejecting geometry stages.
///
/// Used by `compile`/`batch`/`play`/`record`, which run a pure filter graph (a `.fxg` graph cannot
/// encode frame/channel/rate changes). Returns [`Graph::Id`] for an empty chain.
pub(crate) fn parse_chain(tokens: &[String]) -> Result<Graph, String> {
    let mut graph = Graph::Id;
    for stage in parse_stages(tokens)? {
        match stage {
            Stage::Graph(node) => graph = if graph.is_empty() { node } else { graph | node },
            other => {
                return Err(format!(
                    "geometry stage '{}' can't run here — it changes frames/rate/channels; \
                     use it in the `<in> ... <out>` pipeline instead",
                    other.stage_name()
                ));
            }
        }
    }
    Ok(graph)
}

impl Stage {
    /// The stage name (for error messages).
    fn stage_name(&self) -> &'static str {
        match self {
            Stage::Graph(_) => "graph",
            Stage::Trim { .. } => "trim",
            Stage::Pad { .. } => "pad",
            Stage::Rate { .. } => "rate",
            Stage::Speed { .. } => "speed",
            Stage::Repeat { .. } => "repeat",
            Stage::Silence { .. } => "silence",
            Stage::Channels { .. } => "channels",
            Stage::Remix { .. } => "remix",
        }
    }
}

/// Parse one graph op (with its `--param value` flags), returning the node and the next token index.
///
/// Handles the `--db` synthetic flag (gain/normalize) and the variadic `fir --taps`.
fn parse_op(
    kind: OpKind,
    name: &str,
    tokens: &[String],
    start: usize,
) -> Result<(Graph, usize), String> {
    if kind == OpKind::Fir {
        return parse_fir(tokens, start);
    }
    let specs = kind.params();
    let mut params = kind.defaults();
    let mut i = start;
    while i < tokens.len() && tokens[i].starts_with("--") {
        let flag = &tokens[i][2..];
        let value = tokens
            .get(i + 1)
            .ok_or_else(|| format!("missing value for --{flag}"))?;
        // dB ergonomics: gain/normalize accept `--db`, converted to the linear param.
        if flag == "db" && matches!(kind, OpKind::Gain | OpKind::Normalize) {
            let target = if kind == OpKind::Gain { "gain" } else { "peak" };
            let idx = specs.iter().position(|s| s.name == target).unwrap();
            params[idx] = db_to_linear(parse_value(value)?);
            i += 2;
            continue;
        }
        let idx = specs
            .iter()
            .position(|s| s.name == flag)
            .ok_or_else(|| format!("effect '{name}' has no parameter '--{flag}'"))?;
        params[idx] = parse_value(value)?;
        i += 2;
    }
    Ok((
        Graph::from(Op::new(kind, params).map_err(|e| e.to_string())?),
        i,
    ))
}

/// Parse the variadic FIR op: `fir --taps 0.5,0.3,0.2` (or `--tap 0.5` for a single tap).
fn parse_fir(tokens: &[String], start: usize) -> Result<(Graph, usize), String> {
    let mut taps: Option<Vec<f32>> = None;
    let mut i = start;
    while i < tokens.len() && tokens[i].starts_with("--") {
        let flag = &tokens[i][2..];
        let value = tokens
            .get(i + 1)
            .ok_or_else(|| format!("missing value for --{flag}"))?;
        match flag {
            "taps" => {
                taps = Some(
                    value
                        .split(',')
                        .map(|t| parse_value(t.trim()))
                        .collect::<Result<Vec<_>, _>>()?,
                );
            }
            "tap" => taps = Some(vec![parse_value(value)?]),
            _ => return Err(format!("effect 'fir' has no parameter '--{flag}'")),
        }
        i += 2;
    }
    let taps = taps.unwrap_or_else(|| OpKind::Fir.defaults());
    let node = Graph::from(Op::new(OpKind::Fir, taps).map_err(|e| e.to_string())?);
    Ok((node, i))
}

/// One parsed flag: name (without `--`) and its value (`None` for a valueless boolean flag).
type Flag<'a> = (&'a str, Option<&'a str>);

/// Scan a run of `--flag value` (and valueless `--bool`) tokens for a geometry stage, stopping at
/// the first token that is not a flag. Returns the collected flags and the next token index.
fn scan_flags<'a>(
    tokens: &'a [String],
    start: usize,
    bools: &[&str],
) -> Result<(Vec<Flag<'a>>, usize), String> {
    let mut out = Vec::new();
    let mut i = start;
    while i < tokens.len() {
        let tok = tokens[i].as_str();
        if !tok.starts_with("--") {
            break;
        }
        let flag = &tok[2..];
        if bools.contains(&flag) {
            out.push((flag, None));
            i += 1;
        } else {
            let value = tokens
                .get(i + 1)
                .ok_or_else(|| format!("missing value for --{flag}"))?;
            out.push((flag, Some(value.as_str())));
            i += 2;
        }
    }
    Ok((out, i))
}

type Flags<'a> = [Flag<'a>];

/// Optional numeric flag value (errors if the flag is present without a value).
fn opt_num(flags: &Flags, name: &str) -> Result<Option<f32>, String> {
    match flags.iter().find(|(k, _)| *k == name) {
        Some((_, Some(v))) => Ok(Some(parse_value(v)?)),
        Some((_, None)) => Err(format!("--{name} needs a value")),
        None => Ok(None),
    }
}

/// Required numeric flag value.
fn req_num(flags: &Flags, name: &str, stage: &str) -> Result<f32, String> {
    opt_num(flags, name)?.ok_or_else(|| format!("stage '{stage}' requires --{name}"))
}

/// Required non-negative integer flag value (parsed via [`parse_value`], so `k` works).
fn req_int(flags: &Flags, name: &str, stage: &str) -> Result<usize, String> {
    let v = req_num(flags, name, stage)?;
    if v < 0.0 || !v.is_finite() {
        return Err(format!("--{name} must be a non-negative integer, got {v}"));
    }
    Ok(v.round() as usize)
}

/// Whether a valueless flag is present.
fn present(flags: &Flags, name: &str) -> bool {
    flags.iter().any(|(k, v)| *k == name && v.is_none())
}

/// A string flag value, if present.
fn opt_str<'a>(flags: &Flags<'a>, name: &str) -> Option<&'a str> {
    flags
        .iter()
        .find_map(|(k, v)| (*k == name).then_some(*v).flatten())
}

/// Parse one geometry stage's flags into a [`Stage`], returning it and the next token index.
fn parse_stage(name: &str, tokens: &[String], start: usize) -> Result<(Stage, usize), String> {
    let bools = stage_bool_flags(name);
    let (flags, next) = scan_flags(tokens, start, &bools)?;

    // Reject unknown flags up front (typos, wrong stage).
    let doc = stage_doc(name).expect("caller checked stage_doc");
    for (k, _) in &flags {
        if !doc.flags.iter().any(|f| f.flag == *k) {
            return Err(format!("stage '{name}' has no flag '--{k}'"));
        }
    }

    let stage = match name {
        "trim" => Stage::Trim {
            start: opt_num(&flags, "start")?.unwrap_or(0.0),
            len: opt_num(&flags, "len")?,
            end: opt_num(&flags, "end")?,
        },
        "pad" => Stage::Pad {
            start: opt_num(&flags, "start")?.unwrap_or(0.0),
            end: opt_num(&flags, "end")?.unwrap_or(0.0),
        },
        "rate" => {
            let fs = req_num(&flags, "fs", "rate")?;
            if fs < 1.0 || !fs.is_finite() {
                return Err(format!("rate --fs must be a positive frequency, got {fs}"));
            }
            Stage::Rate {
                fs: fs.round() as u32,
            }
        }
        "speed" => {
            let factor = req_num(&flags, "factor", "speed")?;
            // A non-positive factor would blow the frame count up by 1/1e-6 (OOM), not error.
            if factor <= 0.0 || !factor.is_finite() {
                return Err(format!("speed --factor must be positive, got {factor}"));
            }
            Stage::Speed { factor }
        }
        "repeat" => Stage::Repeat {
            count: req_int(&flags, "count", "repeat")?,
        },
        "silence" => {
            let leading = present(&flags, "leading");
            let trailing = present(&flags, "trailing");
            // Default: trim both ends when neither side is named (SoX-style).
            let (leading, trailing) = if !leading && !trailing {
                (true, true)
            } else {
                (leading, trailing)
            };
            Stage::Silence {
                threshold_db: opt_num(&flags, "threshold-db")?.unwrap_or(-60.0),
                min_s: opt_num(&flags, "min")?.unwrap_or(0.0),
                leading,
                trailing,
            }
        }
        "channels" => {
            let count = req_int(&flags, "count", "channels")?;
            // A zero-channel Signal panics the WAV writer downstream; refuse it here.
            if count == 0 {
                return Err("channels --count must be at least 1".into());
            }
            Stage::Channels { count }
        }
        "remix" => {
            let map = opt_str(&flags, "map")
                .ok_or("stage 'remix' requires --map (e.g. \"1,2\" or \"1-2,3\")")?;
            Stage::Remix {
                spec: parse_remix_map(map)?,
            }
        }
        other => return Err(format!("unknown stage '{other}'")),
    };
    Ok((stage, next))
}

/// Parse a SoX-style remix map into per-output-channel `(input, weight)` mixdowns.
///
/// Grammar: comma-separated **output channel** specs; each spec is a 1-based input channel `N`, or a
/// range `A-B` summing inputs `A..=B` (1-based, inclusive). Example: `"1-2,3"` makes two output
/// channels — the sum of inputs 1 and 2, then input 3.
pub(crate) fn parse_remix_map(spec: &str) -> Result<Vec<Vec<(usize, f32)>>, String> {
    let one_based = |s: &str| -> Result<usize, String> {
        let n = s
            .trim()
            .parse::<usize>()
            .map_err(|_| format!("bad channel '{s}' in remix map"))?;
        if n == 0 {
            return Err("remix channels are 1-based; 0 is invalid".into());
        }
        Ok(n - 1)
    };
    let mut out = Vec::new();
    for group in spec.split(',') {
        let g = group.trim();
        if g.is_empty() {
            return Err(format!("empty channel spec in remix map '{spec}'"));
        }
        let ends: Vec<&str> = g.split('-').collect();
        let mixdown = match ends.as_slice() {
            [single] => vec![(one_based(single)?, 1.0)],
            [a, b] => {
                let (lo, hi) = (one_based(a)?, one_based(b)?);
                if hi < lo {
                    return Err(format!("descending range '{g}' in remix map"));
                }
                (lo..=hi).map(|c| (c, 1.0)).collect()
            }
            _ => return Err(format!("bad range '{g}' in remix map")),
        };
        out.push(mixdown);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    #[test]
    fn parse_value_handles_si_and_floats() {
        assert_eq!(parse_value("1000").unwrap(), 1000.0);
        assert_eq!(parse_value("1k").unwrap(), 1000.0);
        assert_eq!(parse_value("1K").unwrap(), 1000.0);
        assert_eq!(parse_value("44.1k").unwrap(), 44_100.0);
        assert_eq!(parse_value("-6").unwrap(), -6.0);
        assert_eq!(parse_value("0.5").unwrap(), 0.5);
        assert!(parse_value("nope").is_err());
        assert!(parse_value("").is_err());
    }

    #[test]
    fn db_flag_converts_to_linear() {
        // gain --db -6 -> ~0.5012 linear.
        let g = parse_chain(&toks("gain --db -6")).unwrap();
        match g {
            Graph::Op(op) => {
                assert_eq!(op.kind, OpKind::Gain);
                assert!((op.params[0] - 0.501_187).abs() < 1e-4, "{:?}", op.params);
            }
            other => panic!("expected a single gain op, got {other}"),
        }
    }

    #[test]
    fn parses_a_chain() {
        let g = parse_chain(&toks("lowpass --cutoff 800 --order 4 gain --gain 0.5")).unwrap();
        assert_eq!(g.leaf_count(), 2);
    }

    #[test]
    fn empty_chain_is_identity() {
        assert_eq!(parse_chain(&[]).unwrap(), Graph::Id);
    }

    #[test]
    fn zero_param_reverse_parses() {
        // Reverse is the first zero-parameter op: params()/defaults() are empty.
        let g = parse_chain(&toks("reverse")).unwrap();
        assert_eq!(g.leaf_count(), 1);
        // And it composes with a following op without swallowing it.
        let g2 = parse_chain(&toks("reverse gain --gain 2")).unwrap();
        assert_eq!(g2.leaf_count(), 2);
    }

    #[test]
    fn fir_taps_are_variadic() {
        let g = parse_chain(&toks("fir --taps 0.5,0.3,0.2")).unwrap();
        match g {
            Graph::Op(op) => {
                assert_eq!(op.kind, OpKind::Fir);
                assert_eq!(op.params, vec![0.5, 0.3, 0.2]);
            }
            other => panic!("expected a single fir op, got {other}"),
        }
    }

    #[test]
    fn rejects_unknown_effect_and_param() {
        assert!(parse_chain(&toks("wobble")).is_err());
        assert!(parse_chain(&toks("lowpass --nope 1")).is_err());
        assert!(parse_chain(&toks("lowpass --cutoff")).is_err()); // missing value
    }

    #[test]
    fn stage_splitter_groups_ops_and_geometry() {
        // lowpass|gain fuse into one graph; trim + rate are separate geometry stages.
        let stages = parse_stages(&toks(
            "lowpass --cutoff 1k gain --gain 0.5 trim --start 1 rate --fs 8000",
        ))
        .unwrap();
        assert_eq!(stages.len(), 3);
        assert!(matches!(stages[0], Stage::Graph(_)));
        assert!(matches!(stages[1], Stage::Trim { .. }));
        assert!(matches!(stages[2], Stage::Rate { fs: 8000 }));
        // The fused graph holds both ops.
        if let Stage::Graph(g) = &stages[0] {
            assert_eq!(g.leaf_count(), 2);
        }
    }

    #[test]
    fn geometry_stage_rejected_by_parse_chain() {
        // compile/play only take a pure filter graph.
        assert!(parse_chain(&toks("trim --start 1")).is_err());
    }

    #[test]
    fn silence_defaults_to_both_ends() {
        let stages = parse_stages(&toks("silence --threshold-db -50")).unwrap();
        match &stages[0] {
            Stage::Silence {
                leading,
                trailing,
                threshold_db,
                ..
            } => {
                assert!(*leading && *trailing);
                assert_eq!(*threshold_db, -50.0);
            }
            _ => panic!("expected a silence stage"),
        }
        // Naming one side selects only that side.
        let one = parse_stages(&toks("silence --leading")).unwrap();
        match &one[0] {
            Stage::Silence {
                leading, trailing, ..
            } => assert!(*leading && !*trailing),
            _ => panic!("expected a silence stage"),
        }
    }

    #[test]
    fn remix_map_parses_channels_and_ranges() {
        // "1,2" -> swap-free identity map (out0=in0, out1=in1); 1-based -> 0-based.
        assert_eq!(
            parse_remix_map("1,2").unwrap(),
            vec![vec![(0, 1.0)], vec![(1, 1.0)]]
        );
        // "1-2,3" -> out0 = in0+in1, out1 = in2.
        assert_eq!(
            parse_remix_map("1-2,3").unwrap(),
            vec![vec![(0, 1.0), (1, 1.0)], vec![(2, 1.0)]]
        );
        assert!(parse_remix_map("0").is_err()); // 1-based
        assert!(parse_remix_map("3-1").is_err()); // descending
        assert!(parse_remix_map("1,,2").is_err()); // empty spec
    }
}

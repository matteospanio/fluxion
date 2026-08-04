//! CLI verbs: `process` (the default pipeline), `batch`, `info`, `stat`, `compile`, `effects`,
//! `synth`, plus the shared input/output plumbing.

use std::collections::HashSet;
use std::f32::consts::PI;

use fluxion::{Graph, OpKind, Signal, Unit, certify_graph, fxg, process, transform};
use fluxion_io::{
    AudioInfo, WavBlockWriter, WavEncoding, decode, decode_blocks, probe, probe_wav, read_wav,
    read_wav_blocks, read_wav_from, write_wav_encoded, write_wav_encoded_to,
};

use crate::chain::{
    STAGES, Stage, parse_chain, parse_stages, parse_value, run_stages, run_stages_with, stage_doc,
};

/// `-` (std stream) and `-n` (null sink) are not real file paths.
pub(crate) fn is_stream(path: &str) -> bool {
    path == "-" || path == "-n"
}

/// An input argument: `-` (stdin) or an existing file — but pipeline keywords win over
/// same-named files: a `.fxg` splices into the chain (the documented `fluxion in.wav chain.fxg
/// out.wav`), and an effect/stage name (e.g. a stray file called `trim`) is a chain token,
/// not audio. Only the first argument is unconditionally an input.
fn is_input_arg(arg: &str) -> bool {
    if arg == "-" {
        return true;
    }
    if arg.ends_with(".fxg") || OpKind::from_name(arg).is_some() || stage_doc(arg).is_some() {
        return false;
    }
    std::path::Path::new(arg).is_file()
}

/// True if two paths resolve to the same existing file.
fn same_file(a: &str, b: &str) -> bool {
    match (
        std::path::Path::new(a).canonicalize(),
        std::path::Path::new(b).canonicalize(),
    ) {
        (Ok(x), Ok(y)) => x == y,
        _ => false,
    }
}

/// Build an output [`WavEncoding`] from the global `--bits` / `--float` / `--no-dither` flags.
///
/// Default is 32-bit float (lossless). `--bits {16|24|32}` selects integer PCM (dithered unless
/// `--no-dither`); `--float` forces 32-bit float and is only valid with `--bits 32` (or none).
pub(crate) fn output_encoding(
    bits: Option<u16>,
    float: bool,
    no_dither: bool,
) -> Result<WavEncoding, String> {
    match (float, bits) {
        (true, None) | (true, Some(32)) => Ok(WavEncoding {
            bits: 32,
            float: true,
            dither: false,
        }),
        (true, Some(b)) => Err(format!("--float requires --bits 32 (got {b})")),
        (false, None) => Ok(WavEncoding::default()),
        (false, Some(b @ (16 | 24 | 32))) => Ok(WavEncoding {
            bits: b,
            float: false,
            dither: !no_dither,
        }),
        (false, Some(b)) => Err(format!("--bits must be 16, 24, or 32 (got {b})")),
    }
}

/// Load an input: `-` = WAV on stdin, `*.wav` via hound, anything else (FLAC/MP3/OGG/…) via Symphonia.
pub(crate) fn load_input(path: &str) -> Result<Signal, String> {
    if path == "-n" {
        return Err("'-n' is a null output sink, not an input".into());
    }
    if path == "-" {
        return read_wav_from(std::io::stdin().lock()).map_err(|e| format!("reading stdin: {e}"));
    }
    let is_wav = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("wav"));
    if is_wav {
        read_wav(path).map_err(|e| format!("reading '{path}': {e}"))
    } else {
        decode(path).map_err(|e| format!("decoding '{path}': {e}"))
    }
}

/// Write the result: `-` = WAV on stdout, `-n` = null sink (discard), otherwise a WAV file.
pub(crate) fn write_output(output: &str, signal: &Signal, enc: WavEncoding) -> Result<(), String> {
    match output {
        "-n" => Ok(()), // null sink
        "-" => write_wav_encoded_to(std::io::stdout().lock(), signal, enc)
            .map_err(|e| format!("writing stdout: {e}")),
        path => write_wav_encoded(path, signal, enc).map_err(|e| format!("writing '{path}': {e}")),
    }
}

/// `fluxion <in...> [effect|stage...] <out>` — run N inputs through the stage pipeline.
///
/// Leading args that are existing files or `-` are inputs (the first arg is always an input); the
/// last arg is the output. Multiple inputs concatenate by default, or sum with `--mix`. A sample-rate
/// mismatch across inputs is an error unless `--rate HZ` is given (each input is resampled to it).
#[allow(clippy::too_many_arguments)] // one parameter per global flag; a struct would only rename them
pub(crate) fn cmd_process(
    args: &[String],
    fs: Option<u32>,
    rate: Option<u32>,
    mix_inputs: bool,
    side_paths: &[String],
    enc: WavEncoding,
    chain_text: Option<&str>,
    dry_run: bool,
) -> Result<(), String> {
    // An effect name where an input file belongs is a discovery attempt, not a pipeline: say what
    // the op is instead of failing to open it as audio.
    if let Some(first) = args.first()
        && let Some(kind) = OpKind::from_name(first)
        && args.len() < 2
    {
        return cmd_effects(std::slice::from_ref(&kind.name().to_string()));
    }
    if args.len() < 2 {
        return Err(
            "usage: fluxion [--mix] [--rate HZ] <in.wav|->... [effect|stage ...] <out.wav|-|-n>\n\
             run `fluxion --help` for the verbs, or `fluxion effects` for the ops"
                .into(),
        );
    }
    let output = args.last().unwrap();
    let body = &args[..args.len() - 1];

    // The first arg is always an input; keep taking leading existing-file / `-` args as inputs.
    let mut n = 1;
    while n < body.len() && is_input_arg(&body[n]) {
        n += 1;
    }
    let (inputs, effects) = body.split_at(n);

    // `--chain "..."` and the argv tokens describe the same thing; accepting both would leave the
    // order between them undefined, so it is an error rather than a guess.
    let stages = match chain_text {
        Some(text) if !effects.is_empty() => {
            return Err(format!(
                "--chain and the inline effects both describe a chain; use one or the other\n\
                 (--chain {text:?}, inline: {})",
                effects.join(" ")
            ));
        }
        Some(text) => vec![Stage::Graph(parse_chain_text(text)?)],
        None => parse_stages(effects)?,
    };

    if dry_run {
        print_dry_run(inputs, &stages, output);
        return Ok(());
    }

    for inp in inputs {
        if !is_stream(inp) && !std::path::Path::new(inp).exists() {
            return Err(format!("no such file '{inp}'"));
        }
    }

    // Refuse to overwrite any input in place (file paths only; `-`/`-n` are fine).
    for inp in inputs {
        if !is_stream(inp) && !is_stream(output) && same_file(inp, output) {
            return Err(format!(
                "input and output are the same file '{output}' — refusing to overwrite"
            ));
        }
    }

    // SoX-style bounded-memory fast path: a single file input, a file (or null) output,
    // no resampling, and a pipeline that lowers whole to the realtime graph is processed
    // in fixed blocks — read → per-channel RtGraph → write — instead of loaded whole.
    // The streaming executor is block-size invariant and the streamed WAV writer is
    // byte-identical to the buffered one, so the output matches the buffered path.
    // The streaming path runs one signal through a realtime graph; a side input is a second
    // signal it has no way to carry, so a chain that uses one takes the buffered path.
    if rate.is_none() && !mix_inputs && inputs.len() == 1 && side_paths.is_empty() {
        let streamed = try_stream_process(&inputs[0], &stages, output, fs, enc)?;
        if streamed {
            return Ok(());
        }
    }

    let mut signals: Vec<Signal> = inputs
        .iter()
        .map(|p| load_input(p))
        .collect::<Result<_, _>>()?;
    if let Some(fs) = fs {
        for s in &mut signals {
            s.fs = fs; // reinterpret declared rate (no resampling)
        }
    }

    let mut signals = align_rates(signals, rate)?;

    let combined = match signals.len() {
        1 => signals.pop().unwrap(),
        _ => {
            let refs: Vec<&Signal> = signals.iter().collect();
            if mix_inputs {
                transform::mix(&refs)
            } else {
                transform::concat(&refs)
            }
        }
    };

    // Side signals are brought to the programme's rate on the way in, for the same reason every
    // other input is: two signals at different rates do not line up frame for frame.
    let sides: Vec<Signal> = side_paths
        .iter()
        .map(|p| load_input(p).map(|s| transform::ensure_fs(s, combined.fs)))
        .collect::<Result<_, _>>()?;
    let side_refs: Vec<&Signal> = sides.iter().collect();

    let out = run_stages_with(&stages, combined, &side_refs);
    write_output(output, &out, enc)
}

/// Parse a `--chain` string, rendering a syntax error with its caret.
fn parse_chain_text(text: &str) -> Result<Graph, String> {
    fluxion::parse::chain(text).map_err(|e| e.render(text))
}

/// `--dry-run`: what would run, in the canonical chain text, and nothing else.
fn print_dry_run(inputs: &[String], stages: &[Stage], output: &str) {
    println!("in : {}", inputs.join(" "));
    for stage in stages {
        match stage {
            // The graph prints in the shared syntax, so this line can be pasted back into
            // `--chain`, into Python, or into the browser.
            Stage::Graph(g) => println!("run: {g}"),
            // Geometry stages have no text form yet (docs/interfaces.md, "Not yet").
            other => println!("run: {other:?}"),
        }
    }
    if stages.is_empty() {
        println!("run: id");
    }
    println!("out: {output}");
}

/// Bring inputs to a common sample rate. With `--rate`, pin every input to it; without it,
/// differing input rates are an error (matching SoX).
///
/// `--rate` is the CLI's half of ROADMAP R2: one project rate, set once, applied on the way in.
/// Inputs already at it are not touched at all.
fn align_rates(signals: Vec<Signal>, rate: Option<u32>) -> Result<Vec<Signal>, String> {
    if let Some(target) = rate {
        return Ok(signals
            .into_iter()
            .map(|s| transform::ensure_fs(s, target))
            .collect());
    }
    let rates: HashSet<u32> = signals.iter().map(|s| s.fs).collect();
    if rates.len() > 1 {
        let mut list: Vec<u32> = rates.into_iter().collect();
        list.sort_unstable();
        let list = list
            .iter()
            .map(|r| format!("{r} Hz"))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!(
            "inputs have different sample rates ({list}); pass --rate HZ to resample them to a \
             common rate"
        ));
    }
    Ok(signals)
}

/// `fluxion batch <out-dir> <glob> [effect...]` — apply a filter chain to every file matching `glob`,
/// writing `<out-dir>/<stem>.wav`. Useful for dataset preprocessing.
pub(crate) fn cmd_batch(args: &[String], fs: Option<u32>, enc: WavEncoding) -> Result<(), String> {
    if args.len() < 2 {
        return Err("usage: fluxion batch <out-dir> <glob> [effect [--flag value]...]".into());
    }
    let (out_dir, pattern, effects) = (&args[0], &args[1], &args[2..]);
    let graph = parse_chain(effects)?;
    std::fs::create_dir_all(out_dir).map_err(|e| format!("creating '{out_dir}': {e}"))?;
    let out_dir_abs = std::path::Path::new(out_dir)
        .canonicalize()
        .map_err(|e| format!("'{out_dir}': {e}"))?;

    let mut produced = HashSet::new();
    let mut count = 0usize;
    for entry in glob::glob(pattern).map_err(|e| format!("bad glob '{pattern}': {e}"))? {
        let path = entry.map_err(|e| format!("glob: {e}"))?;
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("out");
        let out_path = out_dir_abs.join(format!("{stem}.wav"));

        if path.canonicalize().is_ok_and(|p| p == out_path) {
            return Err(format!("refusing to overwrite input '{}'", path.display()));
        }
        if !produced.insert(out_path.clone()) {
            return Err(format!(
                "output collision: two inputs map to '{}' (same stem)",
                out_path.display()
            ));
        }

        let p = path.to_str().ok_or("non-UTF-8 path")?;
        let mut signal = load_input(p)?;
        if let Some(fs) = fs {
            signal.fs = fs;
        }
        let out = process(&graph, &signal);
        write_wav_encoded(&out_path, &out, enc)
            .map_err(|e| format!("writing '{}': {e}", out_path.display()))?;
        count += 1;
    }
    if count == 0 {
        return Err(format!("no files matched glob '{pattern}'"));
    }
    eprintln!("fluxion: processed {count} file(s) → {out_dir}");
    Ok(())
}

/// `fluxion info <file>` — print header metadata. WAV goes through hound (bit-depth/encoding
/// detail); other containers (FLAC/MP3/OGG/…) go through Symphonia's [`probe`].
pub(crate) fn cmd_info(args: &[String]) -> Result<(), String> {
    let path = args.first().ok_or("usage: fluxion info <file>")?;
    let is_wav = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("wav"));

    if is_wav {
        let info = probe_wav(path).map_err(|e| format!("reading '{path}': {e}"))?;
        let fmt = if info.float { "float" } else { "int" };
        println!("{path}");
        println!("  channels    : {}", info.channels);
        println!("  sample rate : {} Hz", info.fs);
        println!("  encoding    : {}-bit {fmt}", info.bits);
        println!("  frames      : {}", info.frames);
        println!("  duration    : {:.3} s", info.seconds());
    } else {
        let info: AudioInfo = probe(path).map_err(|e| format!("reading '{path}': {e}"))?;
        let frames = info
            .frames
            .map_or_else(|| "unknown".to_string(), |n| n.to_string());
        let duration = info
            .seconds()
            .map_or_else(|| "-".to_string(), |s| format!("{s:.3} s"));
        println!("{path}");
        println!("  codec       : {}", info.codec);
        println!("  channels    : {}", info.channels);
        println!("  sample rate : {} Hz", info.fs);
        println!("  frames      : {frames}");
        println!("  duration    : {duration}");
    }
    Ok(())
}

/// `fluxion stat <file>` — signal statistics (length, extrema, peak/RMS dBFS, DC offset, crest).
pub(crate) fn cmd_stat(args: &[String]) -> Result<(), String> {
    let path = args.first().ok_or("usage: fluxion stat <file>")?;
    let sig = load_input(path)?;
    let fs = sig.fs;
    let frames = sig.frames();
    let length = frames as f64 / fs.max(1) as f64;

    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut total = 0usize;
    let mut ch_rms: Vec<f32> = Vec::with_capacity(sig.channel_count());
    for ch in &sig.channels {
        let mut ch_sq = 0.0f64;
        for &x in ch {
            min = min.min(x);
            max = max.max(x);
            sum += x as f64;
            let sq = (x as f64) * (x as f64);
            sum_sq += sq;
            ch_sq += sq;
        }
        total += ch.len();
        let rms = if ch.is_empty() {
            0.0
        } else {
            (ch_sq / ch.len() as f64).sqrt() as f32
        };
        ch_rms.push(rms);
    }
    if total == 0 {
        min = 0.0;
        max = 0.0;
    }
    let peak = min.abs().max(max.abs());
    let rms = if total == 0 {
        0.0
    } else {
        (sum_sq / total as f64).sqrt() as f32
    };
    let dc = if total == 0 {
        0.0
    } else {
        (sum / total as f64) as f32
    };
    let crest = if rms > 0.0 { peak / rms } else { f32::INFINITY };

    println!("{path}");
    println!("  frames        : {frames}");
    println!("  length        : {length:.3} s");
    println!("  sample rate   : {fs} Hz");
    println!("  channels      : {}", sig.channel_count());
    println!("  min amplitude : {min:.6}");
    println!("  max amplitude : {max:.6}");
    println!("  peak          : {} dBFS", fmt_db(peak));
    println!("  RMS           : {} dBFS", fmt_db(rms));
    println!("  DC offset     : {dc:.6}");
    println!("  crest factor  : {}", fmt_ratio(crest));

    // The mastering numbers: what the level actually is to a listener, how far it moves, and
    // whether it will clip on reconstruction. `stat` is where a terminal user looks for them.
    let lufs = fluxion_ops::integrated_loudness(&sig.channels, fs);
    println!("  loudness      : {} LUFS", fmt_lufs(lufs));
    println!(
        "  loudness range: {:.1} LU",
        fluxion_ops::loudness_range(&sig.channels, fs)
    );
    println!(
        "  true peak     : {} dBTP",
        fmt_db_value(fluxion_ops::true_peak(&sig.channels, fs))
    );

    for (i, r) in ch_rms.iter().enumerate() {
        println!("  channel {} RMS : {} dBFS", i + 1, fmt_db(*r));
    }
    Ok(())
}

/// Format a linear amplitude as dBFS, or `-inf` for silence.
fn fmt_db(x: f32) -> String {
    if x > 0.0 {
        format!("{:.2}", 20.0 * x.log10())
    } else {
        "-inf".to_string()
    }
}

/// Format an already-logarithmic value, or `-inf` where there was no signal.
fn fmt_db_value(db: f32) -> String {
    if db.is_finite() {
        format!("{db:.2}")
    } else {
        "-inf".to_string()
    }
}

/// Format a loudness, distinguishing "too quiet to gate" from "shorter than a gating block".
fn fmt_lufs(lufs: f32) -> String {
    if lufs.is_finite() {
        format!("{lufs:.2}")
    } else {
        "-inf (silent, or shorter than one 400 ms block)".to_string()
    }
}

/// Format a ratio, or `inf` when the denominator was zero.
fn fmt_ratio(x: f32) -> String {
    if x.is_finite() {
        format!("{x:.2}")
    } else {
        "inf".to_string()
    }
}

/// `fluxion compile <effect...> <out.fxg>` — serialize a filter chain to a `.fxg` graph, gated by a
/// stability certificate (at `--fs`, default 48 kHz) unless `--force`.
pub(crate) fn cmd_compile(args: &[String], fs: Option<u32>, force: bool) -> Result<(), String> {
    if args.len() < 2 {
        return Err("usage: fluxion compile <effect [--flag value]...> <out.fxg>".into());
    }
    let (effects, out) = args.split_at(args.len() - 1);
    let graph = parse_chain(effects)?;

    let cert = certify_graph(&graph, fs.unwrap_or(48_000));
    eprintln!("stability: {cert}");
    if !cert.verdict.is_shippable() && !force {
        return Err(format!(
            "refusing to write a {} graph; pass --force to override",
            cert.verdict
        ));
    }

    fxg::save(&graph, &out[0]).map_err(|e| format!("writing '{}': {e}", out[0]))?;
    eprintln!("wrote {}: {graph}", out[0]);
    Ok(())
}

/// `fluxion import <ckpt.safetensors> <out.fxg>` — import a DDSP checkpoint trained elsewhere
/// (FLAMO / torchfx state-dict) into the certified freeze pipeline: replay its param→coefficient
/// math, chain the sections as raw `biquad` ops, certify, and write a standard `.fxg` graph that
/// splices anywhere (`fluxion in.wav model.fxg out.wav`, `play`, hot-swap).
///
/// Leading `--flag value` pairs configure the conversion; the two positionals are input and
/// output. `--project-stable` clamps each section into the Jury stability triangle before
/// certification (for checkpoints trained without a stability constraint).
pub(crate) fn cmd_import(args: &[String], fs: Option<u32>, force: bool) -> Result<(), String> {
    use fluxion_io::checkpoint::{
        FlamoBiquadType, ImportOptions, Kind, SvfType, import_safetensors,
    };

    let usage = "usage: fluxion import [--kind K] [--svf-type T] [--biquad-type T] \
       [--eq-flo Hz] [--eq-fhi Hz] [--eq-max-gain dB] [--project-stable] \
       <ckpt.safetensors> <out.fxg>   (with --fs for Hz-parameterised checkpoints)";

    let mut opts = ImportOptions {
        fs,
        ..ImportOptions::default()
    };
    let mut project = false;
    let mut i = 0;
    while i < args.len() && args[i].starts_with("--") {
        let flag = &args[i][2..];
        if flag == "project-stable" {
            project = true;
            i += 1;
            continue;
        }
        let value = args
            .get(i + 1)
            .ok_or_else(|| format!("missing value for --{flag}"))?;
        match flag {
            "kind" => {
                opts.kind = Kind::from_name(value).ok_or_else(|| {
                    format!(
                        "unknown kind '{value}' \
                         (auto|flamo-sos|flamo-svf|flamo-biquad|ddsp-lowpass|ddsp-highpass)"
                    )
                })?;
            }
            "svf-type" => {
                opts.svf_type = SvfType::from_name(value).ok_or_else(|| {
                    format!(
                        "unknown SVF type '{value}' (general|lowpass|highpass|bandpass|\
                         lowshelf|highshelf|peaking|notch)"
                    )
                })?;
            }
            "biquad-type" => {
                opts.biquad_type = FlamoBiquadType::from_name(value)
                    .ok_or_else(|| format!("unknown biquad type '{value}' (lowpass|highpass)"))?;
            }
            "eq-flo" => opts.eq.f_lo = f64::from(parse_value(value)?),
            "eq-fhi" => opts.eq.f_hi = f64::from(parse_value(value)?),
            "eq-max-gain" => opts.eq.max_gain_db = f64::from(parse_value(value)?),
            _ => return Err(format!("unknown flag --{flag}\n{usage}")),
        }
        i += 2;
    }
    let rest = &args[i..];
    if rest.len() != 2 {
        return Err(usage.into());
    }
    let (src, out) = (&rest[0], &rest[1]);

    let imported = import_safetensors(src, &opts).map_err(|e| format!("importing '{src}': {e}"))?;
    for key in &imported.skipped {
        eprintln!("note: skipped non-filter parameter '{key}'");
    }

    // Flatten -> optional Jury projection -> raw biquad graph.
    let mut coeffs: Vec<f32> = imported.sections.iter().flatten().copied().collect();
    if project {
        fluxion::project_stable_flat(&mut coeffs, 1e-3);
    }
    let mut nodes = coeffs
        .chunks_exact(5)
        .map(|c| Graph::op(OpKind::Biquad, [c[0], c[1], c[2], c[3], c[4]]));
    let first = nodes.next().ok_or("checkpoint produced no sections")?;
    let graph = nodes.fold(first, |acc, n| acc | n);

    // Same stability gate as `compile`. The certificate is pole-based, so the fs
    // here only labels the report.
    let design_fs = imported.fs.or(fs).unwrap_or(48_000);
    let cert = certify_graph(&graph, design_fs);
    eprintln!("stability: {cert}");
    if !cert.verdict.is_shippable() && !force {
        return Err(format!(
            "refusing to write a {} graph; retry with --project-stable (Jury clamp) or --force",
            cert.verdict
        ));
    }

    fxg::save(&graph, out).map_err(|e| format!("writing '{out}': {e}"))?;
    eprintln!(
        "wrote {out}: {} raw biquad section(s), designed at {design_fs} Hz — process at the same \
         rate (raw coefficients do not retune)",
        imported.sections.len()
    );
    Ok(())
}

/// `fluxion effects [name]` — list every effect and geometry stage with params/units/defaults, or
/// describe just one. This is the discoverability fix (`trailing_var_arg` swallows `--help`).
pub(crate) fn cmd_effects(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("--json") => {
            println!("{}", registry_json());
            Ok(())
        }
        None => {
            println!("effects (graph ops — compose with the geometry stages below):");
            for &kind in OpKind::all() {
                print_op(kind);
            }
            println!();
            println!("geometry stages (change frames / rate / channels):");
            for doc in STAGES {
                print_stage(doc);
            }
            Ok(())
        }
        Some(name) => {
            if let Some(kind) = OpKind::from_name(name) {
                print_op(kind);
                Ok(())
            } else if let Some(doc) = stage_doc(name) {
                print_stage(doc);
                Ok(())
            } else {
                Err(format!("no effect or stage named '{name}'"))
            }
        }
    }
}

/// The whole catalog as JSON — the machine-readable form of `fluxion effects`.
///
/// This is the one place the op registry leaves the Rust world, and `scripts/gen_interfaces.py`
/// reads it to write `docs/ops.md`, the Python classes and their stubs, and the TypeScript types.
/// Generating those from here rather than from a hand-kept list is what makes "one name
/// everywhere" a check instead of a promise.
///
/// Infinite bounds become `null`, since JSON has no infinity. The geometry stages are included
/// too, marked as what they are: CLI-only, because they change frame count, rate or channel layout
/// and so cannot be `OpKind`s.
fn registry_json() -> String {
    use serde_json::{Value, json};

    // Go through `f32`'s shortest round-tripping decimal before widening, or JSON inherits the
    // f32→f64 tail: `0.707` would come out as `0.7070000171661377`.
    let num = |v: f32| -> Value {
        json!(
            v.to_string()
                .parse::<f64>()
                .expect("a finite f32 always reparses")
        )
    };
    let bound = |v: f32| -> Value { if v.is_finite() { num(v) } else { Value::Null } };

    let ops: Vec<Value> = OpKind::all()
        .iter()
        .map(|&kind| {
            json!({
                "name": kind.name(),
                "class": kind.variant(),
                "group": kind.group().as_str(),
                "variadic": kind.is_variadic(),
                "doc": kind.doc().iter().map(|l| l.trim()).collect::<Vec<_>>().join(" "),
                "params": kind.params().iter().map(|p| json!({
                    "name": p.name,
                    "unit": p.unit.as_str(),
                    "default": num(p.default),
                    "min": bound(p.min),
                    "max": bound(p.max),
                })).collect::<Vec<_>>(),
            })
        })
        .collect();

    let stages: Vec<Value> = STAGES
        .iter()
        .map(|doc| {
            json!({
                "name": doc.name,
                "summary": doc.summary,
                "flags": doc.flags.iter().map(|f| json!({
                    "name": f.flag,
                    "kind": f.kind,
                    "note": f.note,
                })).collect::<Vec<_>>(),
            })
        })
        .collect();

    serde_json::to_string_pretty(&json!({ "version": 1, "ops": ops, "stages": stages }))
        .expect("the registry is always serializable")
}

/// Short unit label for the `effects` listing.
fn unit_label(unit: Unit) -> &'static str {
    match unit {
        Unit::Hz => "Hz",
        Unit::Db => "dB",
        Unit::Seconds => "s",
        Unit::Q => "Q",
        Unit::Linear => "lin",
        // `Unit` is #[non_exhaustive]; a new unit reads as a bare linear value until it earns a label.
        _ => "lin",
    }
}

/// Print one graph op and its parameter schema.
fn print_op(kind: OpKind) {
    println!("  {}", kind.name());
    if kind == OpKind::Fir {
        println!("      --taps <lin,lin,...>   [1]   (variadic tap vector)");
        return;
    }
    let params = kind.params();
    if params.is_empty() {
        println!("      (no parameters)");
    }
    for p in params {
        println!(
            "      --{:<12} <{}>   [{}]",
            p.name,
            unit_label(p.unit),
            trim_float(p.default)
        );
    }
    if matches!(kind, OpKind::Gain | OpKind::Normalize) {
        println!("      --db <dB>            (dB alias for the linear param)");
    }
}

/// Print one geometry stage and its flags.
fn print_stage(doc: &crate::chain::StageDoc) {
    println!("  {} — {}", doc.name, doc.summary);
    for f in doc.flags {
        if f.kind == "flag" {
            println!("      --{:<12} (flag)   {}", f.flag, f.note);
        } else {
            println!("      --{:<12} <{}>   {}", f.flag, f.kind, f.note);
        }
    }
}

/// Render a float without a trailing `.0` (so defaults read `1`, `-20`, `0.707`).
fn trim_float(v: f32) -> String {
    if v == v.trunc() && v.is_finite() {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

// --- synth -----------------------------------------------------------------------------------

/// A generator waveform for the `synth` verb.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Wave {
    Sine,
    Square,
    Saw,
    Triangle,
    White,
}

impl Wave {
    fn from_name(name: &str) -> Option<Wave> {
        match name {
            "sine" => Some(Wave::Sine),
            "square" => Some(Wave::Square),
            "saw" => Some(Wave::Saw),
            "triangle" => Some(Wave::Triangle),
            "white" => Some(Wave::White),
            _ => None,
        }
    }
}

/// Deterministic xorshift32 PRNG for white noise (no `rand` dependency, reproducible output).
struct XorShift(u32);

impl XorShift {
    fn next_unit(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        (x >> 8) as f32 / (1u32 << 24) as f32 // [0, 1)
    }
}

/// Generate one channel of `frames` samples of `wave` at `freq` Hz and `gain` linear amplitude.
fn synth_channel(wave: Wave, freq: f32, fs: u32, frames: usize, gain: f32) -> Vec<f32> {
    let fs = fs.max(1) as f32;
    let mut rng = XorShift(0x_C0FF_EE11);
    (0..frames)
        .map(|k| {
            let t = k as f32 / fs;
            let phase = (freq * t).fract(); // [0, 1) within one period
            let s = match wave {
                Wave::Sine => (2.0 * PI * freq * t).sin(),
                Wave::Square => {
                    if phase < 0.5 {
                        1.0
                    } else {
                        -1.0
                    }
                }
                Wave::Saw => 2.0 * phase - 1.0,
                Wave::Triangle => 1.0 - 4.0 * (phase - 0.5).abs(),
                Wave::White => 2.0 * rng.next_unit() - 1.0,
            };
            gain * s
        })
        .collect()
}

/// `fluxion synth --wave W --freq HZ --secs S [--fs HZ] [--gain LIN] [effect...] <out.wav>` —
/// generate a signal (no input file), optionally run it through a chain, and write it.
pub(crate) fn cmd_synth(
    args: &[String],
    default_fs: Option<u32>,
    enc: WavEncoding,
) -> Result<(), String> {
    let mut wave = Wave::Sine;
    let mut freq = 440.0f32;
    let mut secs = 1.0f32;
    let mut fs = default_fs.unwrap_or(48_000);
    let mut gain = 1.0f32;

    // Leading `--flag value` pairs configure the generator; the first non-flag token starts the
    // (optional) effect chain, and the last arg is the output.
    let mut i = 0;
    while i < args.len() && args[i].starts_with("--") {
        let flag = &args[i][2..];
        let value = args
            .get(i + 1)
            .ok_or_else(|| format!("missing value for --{flag}"))?;
        match flag {
            "wave" => {
                wave = Wave::from_name(value).ok_or_else(|| {
                    format!("unknown waveform '{value}' (sine|square|saw|triangle|white)")
                })?;
            }
            "freq" => freq = parse_value(value)?,
            "secs" => secs = parse_value(value)?,
            "gain" => gain = parse_value(value)?,
            "fs" => {
                let v = parse_value(value)?;
                if v < 1.0 || !v.is_finite() {
                    return Err(format!("synth --fs must be a positive frequency, got {v}"));
                }
                fs = v.round() as u32;
            }
            other => return Err(format!("synth: unknown flag '--{other}'")),
        }
        i += 2;
    }

    let rest = &args[i..];
    let output = rest
        .last()
        .ok_or("usage: fluxion synth --wave W --freq HZ --secs S [effect...] <out.wav>")?;
    let effects = &rest[..rest.len() - 1];

    let frames = (secs.max(0.0) * fs as f32).round() as usize;
    let sig = Signal::new(fs, vec![synth_channel(wave, freq, fs, frames, gain)]);
    let stages = parse_stages(effects)?;
    let out = run_stages(&stages, sig);
    write_output(output, &out, enc)
}

/// Frames per streamed block: big enough to amortize per-block overhead, small enough
/// that peak memory stays a few MB regardless of file length.
const STREAM_BLOCK: usize = 65_536;

/// Bounded-memory fast path for [`cmd_process`]. Returns `Ok(false)` when the input,
/// output, or pipeline shape does not qualify — the caller falls back to the buffered
/// path — and `Ok(true)` after the file has been fully processed and written.
fn try_stream_process(
    input: &str,
    stages: &[Stage],
    output: &str,
    fs_override: Option<u32>,
    enc: WavEncoding,
) -> Result<bool, String> {
    if is_stream(input) || output == "-" {
        return Ok(false); // stdin/stdout need the buffered path (seekable-sink WAV header)
    }
    // Only effect stages stream; geometry stages (trim/pad/rate/…) need the whole signal.
    let mut graph: Option<fluxion::Graph> = None;
    for st in stages {
        match st {
            Stage::Graph(g) => {
                graph = Some(match graph {
                    Some(acc) => acc | g.clone(),
                    None => g.clone(),
                });
            }
            _ => return Ok(false),
        }
    }
    let graph = graph.unwrap_or(fluxion::Graph::Id);

    let is_wav = std::path::Path::new(input)
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("wav"));
    let probed_fs = if is_wav {
        probe_wav(input)
            .map_err(|e| format!("probing '{input}': {e}"))?
            .fs
    } else {
        probe(input)
            .map_err(|e| format!("probing '{input}': {e}"))?
            .fs
    };
    let fs = fs_override.unwrap_or(probed_fs);
    if fs == 0 || fluxion::to_rt_graph(&graph, fs).is_none() {
        return Ok(false); // unknown rate, or a stage that is not realtime-lowerable
    }

    let total_frames = if is_wav {
        probe_wav(input).map(|i| i.frames as usize).unwrap_or(0)
    } else {
        probe(input)
            .map(|i| i.frames.unwrap_or(0) as usize)
            .unwrap_or(0)
    };
    let mut progress = Progress::new(total_frames, fs);

    if is_wav {
        let blocks = read_wav_blocks(input, STREAM_BLOCK)
            .map_err(|e| format!("reading '{input}': {e}"))?
            .map(|r| r.map_err(|e| format!("reading '{input}': {e}")));
        stream_blocks(blocks, &graph, fs, output, enc, &mut progress)
    } else {
        let blocks = decode_blocks(input, STREAM_BLOCK)
            .map_err(|e| format!("decoding '{input}': {e}"))?
            .map(|r| r.map_err(|e| format!("decoding '{input}': {e}")));
        stream_blocks(blocks, &graph, fs, output, enc, &mut progress)
    }
}

/// A one-line progress readout for long files.
///
/// Off unless stderr is a terminal, so a redirect or a CI log never collects carriage returns —
/// and the snapshot tests stay clean. Off for short files too: a bar that finishes instantly is
/// noise. Writes to stderr so `fluxion in.wav ... -` stays a usable pipe.
struct Progress {
    total_frames: usize,
    fs: u32,
    done: usize,
    last_percent: usize,
    show: bool,
}

impl Progress {
    /// Anything under this many seconds of audio finishes before a reader could read the line.
    const MIN_SECONDS: f32 = 10.0;

    fn new(total_frames: usize, fs: u32) -> Progress {
        use std::io::IsTerminal;
        let seconds = if fs > 0 {
            total_frames as f32 / fs as f32
        } else {
            0.0
        };
        Progress {
            total_frames,
            fs,
            done: 0,
            last_percent: usize::MAX,
            show: std::io::stderr().is_terminal() && seconds >= Self::MIN_SECONDS,
        }
    }

    fn advance(&mut self, frames: usize) {
        self.done += frames;
        if !self.show || self.total_frames == 0 {
            return;
        }
        let percent = (self.done * 100 / self.total_frames).min(100);
        if percent == self.last_percent {
            return;
        }
        self.last_percent = percent;
        eprint!(
            "\r  {percent:>3}%  {:.1}s / {:.1}s",
            self.done as f32 / self.fs as f32,
            self.total_frames as f32 / self.fs as f32,
        );
    }

    fn finish(&self) {
        if self.show {
            eprintln!();
        }
    }
}

/// Drive the block loop: per-channel [`fluxion::RtGraph`] state persists across blocks
/// (sample-identical to whole-signal processing), output written incrementally.
fn stream_blocks(
    blocks: impl Iterator<Item = Result<Signal, String>>,
    graph: &fluxion::Graph,
    fs: u32,
    output: &str,
    enc: WavEncoding,
    progress: &mut Progress,
) -> Result<bool, String> {
    let mut graphs: Vec<fluxion::RtGraph> = Vec::new();
    let mut writer: Option<WavBlockWriter> = None;
    let mut outs: Vec<Vec<f32>> = Vec::new();

    for block in blocks {
        let sig = block?;
        let ch = sig.channel_count();
        if graphs.is_empty() {
            graphs = (0..ch)
                .map(|_| {
                    let mut g = fluxion::to_rt_graph(graph, fs).expect("lowerability checked");
                    g.prepare(STREAM_BLOCK);
                    g
                })
                .collect();
            outs = vec![vec![0.0f32; STREAM_BLOCK]; ch];
            if output != "-n" {
                writer = Some(
                    WavBlockWriter::create(output, fs, ch as u16, enc)
                        .map_err(|e| format!("writing '{output}': {e}"))?,
                );
            }
        } else if ch != graphs.len() {
            return Err(format!(
                "channel count changed mid-stream ({} -> {ch})",
                graphs.len()
            ));
        }
        for (c, in_ch) in sig.channels.iter().enumerate() {
            outs[c].resize(in_ch.len(), 0.0);
            graphs[c].process(in_ch, &mut outs[c]);
        }
        if let Some(w) = writer.as_mut() {
            w.write_block(&outs)
                .map_err(|e| format!("writing '{output}': {e}"))?;
        }
        progress.advance(sig.frames());
    }
    progress.finish();
    if let Some(w) = writer.take() {
        w.finalize()
            .map_err(|e| format!("writing '{output}': {e}"))?;
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::{Wave, output_encoding, synth_channel};
    use fluxion_io::WavEncoding;

    #[test]
    fn sine_hits_known_samples() {
        // fs = 8000, f = 1000 -> period 8 samples: sin(pi*k/4). k=0 -> 0, k=2 -> +1, k=6 -> -1.
        let ch = synth_channel(Wave::Sine, 1_000.0, 8_000, 8, 1.0);
        assert!(ch[0].abs() < 1e-6, "sine must start at 0");
        assert!((ch[2] - 1.0).abs() < 1e-5, "quarter period is the +peak");
        assert!(
            (ch[6] + 1.0).abs() < 1e-5,
            "three-quarter period is the -peak"
        );
    }

    #[test]
    fn white_noise_stays_in_range() {
        let gain = 0.5f32;
        let ch = synth_channel(Wave::White, 0.0, 48_000, 4_096, gain);
        assert!(
            ch.iter().all(|&x| x.abs() <= gain),
            "white noise must stay within ±gain"
        );
        // And it is actually noisy (not a constant).
        let distinct = ch.iter().any(|&x| (x - ch[0]).abs() > 1e-6);
        assert!(distinct, "white noise must vary");
    }

    #[test]
    fn encoding_flags_map_to_wav_encoding() {
        assert_eq!(
            output_encoding(None, false, false).unwrap(),
            WavEncoding::default()
        );
        assert_eq!(
            output_encoding(Some(16), false, false).unwrap(),
            WavEncoding {
                bits: 16,
                float: false,
                dither: true
            }
        );
        assert_eq!(
            output_encoding(Some(24), false, true).unwrap(),
            WavEncoding {
                bits: 24,
                float: false,
                dither: false
            }
        );
        assert_eq!(
            output_encoding(None, true, false).unwrap(),
            WavEncoding {
                bits: 32,
                float: true,
                dither: false
            }
        );
        assert!(output_encoding(Some(20), false, false).is_err());
        assert!(output_encoding(Some(16), true, false).is_err()); // float needs 32-bit
    }
}

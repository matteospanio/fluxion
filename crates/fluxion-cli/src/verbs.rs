//! CLI verbs: `process` (the default pipeline), `batch`, `info`, `stat`, `compile`, `effects`,
//! `synth`, plus the shared input/output plumbing.

use std::collections::HashSet;
use std::f32::consts::PI;

use fluxion::{OpKind, Signal, Unit, certify_graph, fxg, process, transform};
use fluxion_io::{
    AudioInfo, WavEncoding, decode, probe, probe_wav, read_wav, read_wav_from, write_wav_encoded,
    write_wav_encoded_to,
};

use crate::chain::{STAGES, parse_chain, parse_stages, parse_value, run_stages, stage_doc};

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
pub(crate) fn cmd_process(
    args: &[String],
    fs: Option<u32>,
    rate: Option<u32>,
    mix_inputs: bool,
    enc: WavEncoding,
) -> Result<(), String> {
    if args.len() < 2 {
        return Err(
            "usage: fluxion [--mix] [--rate HZ] <in.wav|->... [effect|stage ...] <out.wav|-|-n>"
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

    // Refuse to overwrite any input in place (file paths only; `-`/`-n` are fine).
    for inp in inputs {
        if !is_stream(inp) && !is_stream(output) && same_file(inp, output) {
            return Err(format!(
                "input and output are the same file '{output}' — refusing to overwrite"
            ));
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

    align_rates(&mut signals, rate)?;

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

    let stages = parse_stages(effects)?;
    let out = run_stages(&stages, combined);
    write_output(output, &out, enc)
}

/// Bring inputs to a common sample rate. With `--rate`, resample every input to it; without it,
/// differing input rates are an error (matching SoX).
fn align_rates(signals: &mut [Signal], rate: Option<u32>) -> Result<(), String> {
    if let Some(target) = rate {
        for s in signals.iter_mut() {
            if s.fs != target {
                *s = transform::resample(s, target);
            }
        }
        return Ok(());
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
    Ok(())
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

/// `fluxion effects [name]` — list every effect and geometry stage with params/units/defaults, or
/// describe just one. This is the discoverability fix (`trailing_var_arg` swallows `--help`).
pub(crate) fn cmd_effects(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
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

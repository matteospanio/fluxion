//! `fluxion` — a modern, SoX-style audio DSP command-line interface.
//!
//! Process a WAV through a parsed effect chain on the CPU. The pipeline mirrors SoX —
//! `fluxion in.wav <effect [--flag value]...> out.wav` — but with named effects, long `--flags`,
//! and explicit units. Two verbs sit alongside it:
//!
//! - `fluxion info <file.wav>` — print metadata (soxi-style).
//! - `fluxion compile <effect...> <out.fxg>` — save an effect chain as a `.fxg` graph.
//!
//! A `.fxg` file can be dropped into a pipeline as if it were an effect:
//! `fluxion in.wav chain.fxg out.wav`.

use std::process::ExitCode;

use clap::Parser;
use fluxion::{Graph, Op, OpKind, fxg, process};
use fluxion_io::{decode, probe_wav, read_wav, read_wav_from, write_wav, write_wav_to};

#[derive(Parser)]
#[command(name = "fluxion", version, about = "Modern, SoX-style audio DSP CLI")]
struct Cli {
    /// Sample-rate override in Hz (default: read from the input file). Must precede the pipeline.
    #[arg(long)]
    fs: Option<u32>,

    /// `compile`: write the graph even if its stability certificate is not shippable.
    #[arg(long)]
    force: bool,

    /// `info <file>` | `compile <effect...> <out.fxg>` | `<in.wav> [effect...] <out.wav>`
    #[arg(trailing_var_arg = true, allow_hyphen_values = true, num_args = 0..)]
    args: Vec<String>,
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("fluxion: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<(), String> {
    match cli.args.first().map(String::as_str) {
        // `soxi` is a SoX-compatible alias for `info`.
        Some("info") | Some("soxi") => cmd_info(&cli.args[1..]),
        Some("compile") => cmd_compile(&cli.args[1..], cli.fs, cli.force),
        Some("batch") => cmd_batch(&cli.args[1..], cli.fs),
        _ => cmd_process(&cli.args, cli.fs),
    }
}

/// `fluxion in.wav [effect...] out.wav` — filter a file through the chain.
///
/// `-` reads/writes WAV on stdin/stdout (`fluxion - lowpass --cutoff 800 - < in.wav > out.wav`);
/// `-n` is a null output sink (process for analysis, write nothing).
fn cmd_process(args: &[String], fs: Option<u32>) -> Result<(), String> {
    if args.len() < 2 {
        return Err("usage: fluxion <in.wav|-> [effect [--flag value]...] <out.wav|-|-n>".into());
    }
    let input = &args[0];
    let output = args.last().unwrap();
    let effects = &args[1..args.len() - 1];

    let mut signal = load_input(input)?;
    if let Some(fs) = fs {
        signal.fs = fs;
    }
    let graph = parse_chain(effects)?;
    let out = process(&graph, &signal);
    write_output(output, &out)
}

/// Load an input: `-` = WAV on stdin, `*.wav` via hound, anything else (FLAC/MP3/OGG/…) via Symphonia.
fn load_input(path: &str) -> Result<fluxion::Signal, String> {
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
fn write_output(output: &str, signal: &fluxion::Signal) -> Result<(), String> {
    match output {
        "-n" => Ok(()), // null sink
        "-" => write_wav_to(std::io::stdout().lock(), signal)
            .map_err(|e| format!("writing stdout: {e}")),
        path => write_wav(path, signal).map_err(|e| format!("writing '{path}': {e}")),
    }
}

/// `fluxion batch <out-dir> <glob> [effect...]` — apply the chain to every file matching `glob`,
/// writing `<out-dir>/<stem>.wav`. Useful for dataset preprocessing.
fn cmd_batch(args: &[String], fs: Option<u32>) -> Result<(), String> {
    if args.len() < 2 {
        return Err("usage: fluxion batch <out-dir> <glob> [effect [--flag value]...]".into());
    }
    let (out_dir, pattern, effects) = (&args[0], &args[1], &args[2..]);
    let graph = parse_chain(effects)?;
    std::fs::create_dir_all(out_dir).map_err(|e| format!("creating '{out_dir}': {e}"))?;

    let mut count = 0usize;
    for entry in glob::glob(pattern).map_err(|e| format!("bad glob '{pattern}': {e}"))? {
        let path = entry.map_err(|e| format!("glob: {e}"))?;
        let p = path.to_str().ok_or("non-UTF-8 path")?;
        let mut signal = load_input(p)?;
        if let Some(fs) = fs {
            signal.fs = fs;
        }
        let out = process(&graph, &signal);
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("out");
        let out_path = std::path::Path::new(out_dir).join(format!("{stem}.wav"));
        write_wav(&out_path, &out).map_err(|e| format!("writing '{}': {e}", out_path.display()))?;
        count += 1;
    }
    eprintln!("fluxion: processed {count} file(s) → {out_dir}");
    Ok(())
}

/// `fluxion info <file.wav>` — print header metadata.
fn cmd_info(args: &[String]) -> Result<(), String> {
    let path = args.first().ok_or("usage: fluxion info <file.wav>")?;
    let info = probe_wav(path).map_err(|e| format!("reading '{path}': {e}"))?;
    let fmt = if info.float { "float" } else { "int" };
    println!("{path}");
    println!("  channels    : {}", info.channels);
    println!("  sample rate : {} Hz", info.fs);
    println!("  encoding    : {}-bit {fmt}", info.bits);
    println!("  frames      : {}", info.frames);
    println!("  duration    : {:.3} s", info.seconds());
    Ok(())
}

/// `fluxion compile <effect...> <out.fxg>` — serialize an effect chain to a `.fxg` graph.
///
/// Runs a stability certificate (at `--fs`, default 48 kHz) on the frozen coefficients and refuses
/// to write a graph that is not shippable unless `--force` is given.
fn cmd_compile(args: &[String], fs: Option<u32>, force: bool) -> Result<(), String> {
    if args.len() < 2 {
        return Err("usage: fluxion compile <effect [--flag value]...> <out.fxg>".into());
    }
    let (effects, out) = args.split_at(args.len() - 1);
    let graph = parse_chain(effects)?;

    let cert = fluxion::certify_graph(&graph, fs.unwrap_or(48_000));
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

/// Parse a SoX-style effect chain into a series [`Graph`].
///
/// Grammar: `effect [--param value]... effect ...`. Each token is either a known [`OpKind`] name
/// (with its `--param` flags) or a `.fxg` file to load and splice in. Unspecified params use
/// defaults; an empty chain yields [`Graph::Id`] (a straight copy / transcode).
fn parse_chain(tokens: &[String]) -> Result<Graph, String> {
    let mut graph = Graph::Id;
    let mut i = 0;
    while i < tokens.len() {
        let name = &tokens[i];
        i += 1;

        let node = if let Some(kind) = OpKind::from_name(name) {
            let specs = kind.params();
            let mut params = kind.defaults();
            while i < tokens.len() && tokens[i].starts_with("--") {
                let flag = &tokens[i][2..];
                let value = tokens
                    .get(i + 1)
                    .ok_or_else(|| format!("missing value for --{flag}"))?;
                let idx = specs
                    .iter()
                    .position(|s| s.name == flag)
                    .ok_or_else(|| format!("effect '{name}' has no parameter '--{flag}'"))?;
                params[idx] = value
                    .parse::<f32>()
                    .map_err(|_| format!("invalid number for --{flag}: '{value}'"))?;
                i += 2;
            }
            Graph::from(Op::new(kind, params).map_err(|e| e.to_string())?)
        } else if name.ends_with(".fxg") {
            fxg::load(name).map_err(|e| format!("loading '{name}': {e}"))?
        } else {
            return Err(format!("unknown effect '{name}'"));
        };

        graph = if graph.is_empty() { node } else { graph | node };
    }
    Ok(graph)
}

#[cfg(test)]
mod tests {
    use super::parse_chain;
    use fluxion::Graph;

    fn toks(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
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
    fn rejects_unknown_effect_and_param() {
        assert!(parse_chain(&toks("wobble")).is_err());
        assert!(parse_chain(&toks("lowpass --nope 1")).is_err());
        assert!(parse_chain(&toks("lowpass --cutoff")).is_err()); // missing value
    }
}

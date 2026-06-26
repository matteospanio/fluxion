//! `fluxion` — a modern, SoX-style audio DSP command-line interface.
//!
//! M1 scope: process a WAV file through a parsed effect chain on the CPU. The pipeline mirrors
//! SoX — `fluxion in.wav <effect [--flag value]...> out.wav` — but with named effects, long
//! `--flags`, and explicit units. `info`/`play`/`record`/`compile` arrive in later milestones.

use std::process::ExitCode;

use clap::Parser;
use fluxion::{Graph, Op, OpKind, process};
use fluxion_io::{read_wav, write_wav};

#[derive(Parser)]
#[command(name = "fluxion", version, about = "Modern, SoX-style audio DSP CLI")]
struct Cli {
    /// Sample-rate override in Hz (default: read from the input file). Must precede the pipeline.
    #[arg(long)]
    fs: Option<u32>,

    /// in.wav [effect [--flag value]...] out.wav
    #[arg(trailing_var_arg = true, allow_hyphen_values = true, num_args = 0..)]
    pipeline: Vec<String>,
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
    if cli.pipeline.len() < 2 {
        return Err("usage: fluxion <in.wav> [effect [--flag value]...] <out.wav>".into());
    }
    let input = &cli.pipeline[0];
    let output = cli.pipeline.last().unwrap();
    let effects = &cli.pipeline[1..cli.pipeline.len() - 1];

    let mut signal = read_wav(input).map_err(|e| format!("reading '{input}': {e}"))?;
    if let Some(fs) = cli.fs {
        signal.fs = fs;
    }

    let graph = parse_chain(effects)?;
    let out = process(&graph, &signal);
    write_wav(output, &out).map_err(|e| format!("writing '{output}': {e}"))?;
    Ok(())
}

/// Parse a SoX-style effect chain into a series [`Graph`].
///
/// Grammar: `effect [--param value]... effect ...`. Each effect name must be a known [`OpKind`];
/// `--param` names come from that op's [`OpKind::params`] schema; unspecified params use defaults.
/// An empty chain yields [`Graph::Id`] (a straight copy / transcode).
fn parse_chain(tokens: &[String]) -> Result<Graph, String> {
    let mut graph = Graph::Id;
    let mut i = 0;
    while i < tokens.len() {
        let name = &tokens[i];
        i += 1;
        let kind = OpKind::from_name(name).ok_or_else(|| format!("unknown effect '{name}'"))?;
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

        let op = Op::new(kind, params).map_err(|e| e.to_string())?;
        graph = if graph.is_empty() {
            Graph::from(op)
        } else {
            graph | Graph::from(op)
        };
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

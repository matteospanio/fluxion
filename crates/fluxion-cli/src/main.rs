//! `fluxion` — a modern, SoX-style audio DSP command-line interface.
//!
//! The default pipeline runs one or more inputs through an ordered chain of **stages** and writes an
//! output:
//!
//! ```text
//! fluxion [globals] <in.wav|->... [effect|stage ...] <out.wav|-|-n>
//! ```
//!
//! Adjacent DSP effects fuse into one filter pass; **geometry stages** between them change the frame
//! count, sample rate, or channel layout. It mirrors SoX's philosophy — not its interface: effects
//! are named, flags are long, and units are explicit (seconds / Hz / dB).
//!
//! - **Effects** (per-channel, length/rate/channel-preserving, composable): `gain`, `lowpass`,
//!   `highpass`, `peaking`, `lowshelf`, `highshelf`, `notch`, `bandpass`, `allpass`, `delay`, `echo`,
//!   `cheby1low/high`, `cheby2low/high`, `reverb`, `fir`, `fade`, `tremolo`, `overdrive`, `compand`,
//!   `reverse`, `biquad`, `chorus`, `flanger`, `phaser`. Run `fluxion effects` for the full schema.
//! - **Geometry stages**: `trim`, `pad`, `rate`, `speed`, `repeat`, `silence`, `channels`, `remix`.
//! - **Numbers** accept a `k`/`K` suffix (`--cutoff 1k`); `gain --db` / `normalize --db` take dB.
//!
//! Global flags (must precede the pipeline): `--fs HZ` (reinterpret the input rate), `--rate HZ`
//! (resample inputs to a common rate), `--mix` (sum inputs instead of concatenating), `--bits
//! {16|24|32}` / `--float` / `--no-dither` (output encoding), `--secs N` (record duration), `--force`
//! (compile past a bad stability certificate).
//!
//! Other verbs: `info`/`soxi` (metadata), `stat` (signal statistics), `effects [name]` (discover the
//! grammar), `synth` (generate a tone/noise), `compile` (freeze a chain to a `.fxg` graph),
//! `import` (convert a FLAMO / torchfx DDSP checkpoint to a certified `.fxg`), `batch`
//! (glob → directory), and `play`/`record` (feature `realtime`). A `.fxg` file drops into a pipeline
//! as if it were an effect: `fluxion in.wav chain.fxg out.wav`.

use std::process::ExitCode;

use clap::Parser;

mod chain;
mod realtime;
mod verbs;

use verbs::{
    cmd_batch, cmd_compile, cmd_effects, cmd_import, cmd_info, cmd_process, cmd_stat, cmd_synth,
    output_encoding,
};

#[derive(Parser)]
#[command(name = "fluxion", version, about = "Modern, SoX-style audio DSP CLI")]
struct Cli {
    /// Reinterpret the input sample rate in Hz (no resampling). Must precede the pipeline.
    #[arg(long)]
    fs: Option<u32>,

    /// Resample every input to this rate (Hz) before combining — required to mix inputs of
    /// differing rates.
    #[arg(long)]
    rate: Option<u32>,

    /// Sum multiple inputs (zero-padded to the longest) instead of concatenating them.
    #[arg(long)]
    mix: bool,

    /// Output bit depth: 16, 24, or 32. Default: 32-bit float.
    #[arg(long)]
    bits: Option<u16>,

    /// Write 32-bit float output (the default); only valid with `--bits 32` or no `--bits`.
    #[arg(long)]
    float: bool,

    /// Disable TPDF dither on integer-PCM output.
    #[arg(long = "no-dither")]
    no_dither: bool,

    /// `compile`/`import`: write the graph even if its stability certificate is not shippable.
    #[arg(long)]
    force: bool,

    /// `record`: capture duration in seconds.
    #[arg(long, default_value_t = 5.0)]
    secs: f32,

    /// The verb or pipeline. See the crate docs / `fluxion effects` for the full grammar.
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
    let enc = output_encoding(cli.bits, cli.float, cli.no_dither)?;
    match cli.args.first().map(String::as_str) {
        // `soxi` is a SoX-compatible alias for `info`.
        Some("info") | Some("soxi") => cmd_info(&cli.args[1..]),
        Some("stat") => cmd_stat(&cli.args[1..]),
        Some("effects") => cmd_effects(&cli.args[1..]),
        Some("compile") => cmd_compile(&cli.args[1..], cli.fs, cli.force),
        Some("import") => cmd_import(&cli.args[1..], cli.fs, cli.force),
        Some("batch") => cmd_batch(&cli.args[1..], cli.fs, enc),
        Some("synth") => cmd_synth(&cli.args[1..], cli.fs, enc),
        Some("play") => realtime::play(&cli.args[1..], cli.fs),
        Some("record") => realtime::record(&cli.args[1..], cli.secs, enc),
        _ => cmd_process(&cli.args, cli.fs, cli.rate, cli.mix, enc),
    }
}

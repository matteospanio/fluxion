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
//!   `cheby1_lowpass/highpass`, `cheby2_lowpass/highpass`, `reverb`, `fir`, `fade`, `tremolo`,
//!   `overdrive`, `compand`,
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

/// What `fluxion --help` shows below the flag list.
///
/// Hand-written and deliberately short: a screen that lists the verbs, points at the two
/// self-describing commands, and shows the chain syntax beats an exhaustive dump nobody reads. The
/// budget (40 lines, 80 columns) is asserted in `tests/cli_snapshots.rs`.
const HELP: &str = "\
Pipeline:
  fluxion [globals] <in.wav|->... [effect|stage ...] <out.wav|-|-n>
  fluxion --chain \"highpass(80, 4) | gain(-3dB)\" in.wav out.wav

Verbs:
  effects [name]   list every effect and stage, or describe one
                   (--json emits the whole registry)
  stat <in>        peak/RMS dBFS, DC offset, crest factor
  info <in>        format and metadata (alias: soxi)
  synth            generate a tone or noise
  compile          freeze a chain to a portable .fxg graph
  import           convert a FLAMO / torchfx checkpoint to a certified .fxg
  batch <dir> <glob>   run one pipeline over many files
  play / record    live audio (build with --features realtime)

Chain syntax (the same text Python, C and the browser accept):
  '|' series, '+' parallel, 'name(a, b)' or 'name=a,b', suffixes 'k' and 'dB'
  See docs/chain-syntax.md; `fluxion --dry-run` prints the chain it would run.";

#[derive(Parser)]
#[command(
    name = "fluxion",
    version,
    about = "Modern, SoX-style audio DSP CLI",
    after_help = HELP
)]
struct Cli {
    /// Reinterpret the input rate in Hz (no resampling).
    #[arg(long)]
    fs: Option<u32>,

    /// Resample every input to this rate (Hz) first.
    #[arg(long)]
    rate: Option<u32>,

    /// Sum multiple inputs instead of concatenating them.
    #[arg(long)]
    mix: bool,

    /// Signal read by `side(0)`, `side(1)`, …; repeat for more.
    #[arg(long = "side", value_name = "FILE")]
    sides: Vec<String>,

    /// Output bit depth 16/24/32 (default: 32-bit float).
    #[arg(long)]
    bits: Option<u16>,

    /// Write 32-bit float output (the default).
    #[arg(long)]
    float: bool,

    /// Disable TPDF dither on integer-PCM output.
    #[arg(long = "no-dither")]
    no_dither: bool,

    /// compile/import: write past a bad stability certificate.
    #[arg(long)]
    force: bool,

    /// `record`: capture duration in seconds.
    #[arg(long, default_value_t = 5.0)]
    secs: f32,

    /// Build the chain from the shared text syntax.
    #[arg(long, value_name = "TEXT")]
    chain: Option<String>,

    /// Print the chain that would run, then exit.
    #[arg(long = "dry-run")]
    dry_run: bool,

    /// The verb or pipeline; run `fluxion effects` for the ops.
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

/// Print the same screen `--help` shows. Running `fluxion` with nothing to do is a request for
/// help, not an error — it exits 0.
fn print_help() {
    use clap::CommandFactory;
    Cli::command().print_help().expect("stdout is writable");
}

fn run(cli: Cli) -> Result<(), String> {
    let enc = output_encoding(cli.bits, cli.float, cli.no_dither)?;
    match cli.args.first().map(String::as_str) {
        // Nothing to do, or an explicit request: show the help screen and exit cleanly.
        None if cli.chain.is_none() => {
            print_help();
            Ok(())
        }
        Some("help") => match cli.args.get(1) {
            // `fluxion help lowpass` is `fluxion effects lowpass` — one less thing to know.
            Some(_) => cmd_effects(&cli.args[1..]),
            None => {
                print_help();
                Ok(())
            }
        },
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
        _ => cmd_process(
            &cli.args,
            cli.fs,
            cli.rate,
            cli.mix,
            &cli.sides,
            enc,
            cli.chain.as_deref(),
            cli.dry_run,
        ),
    }
}

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

    /// `record`: capture duration in seconds.
    #[arg(long, default_value_t = 5.0)]
    secs: f32,

    /// `info <file>` | `compile <effect...> <out.fxg>` | `play <in.wav> [effect...]` |
    /// `record [effect...] <out.wav>` | `<in.wav> [effect...] <out.wav>`
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
        Some("play") => realtime::play(&cli.args[1..], cli.fs),
        Some("record") => realtime::record(&cli.args[1..], cli.secs),
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

    // Refuse to clobber the input by writing the result over it (file paths only; `-`/`-n` are fine).
    if !is_stream(input) && !is_stream(output) && same_file(input, output) {
        return Err(format!(
            "input and output are the same file '{output}' — refusing to overwrite"
        ));
    }

    let mut signal = load_input(input)?;
    if let Some(fs) = fs {
        signal.fs = fs;
    }
    let graph = parse_chain(effects)?;
    let out = process(&graph, &signal);
    write_output(output, &out)
}

/// `-` (std stream) and `-n` (null sink) are not real file paths.
fn is_stream(path: &str) -> bool {
    path == "-" || path == "-n"
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

/// Load an input: `-` = WAV on stdin, `*.wav` via hound, anything else (FLAC/MP3/OGG/…) via Symphonia.
fn load_input(path: &str) -> Result<fluxion::Signal, String> {
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
    // Absolute, so we can detect output collisions and refuse to overwrite an input in place.
    let out_dir_abs = std::path::Path::new(out_dir)
        .canonicalize()
        .map_err(|e| format!("'{out_dir}': {e}"))?;

    let mut produced = std::collections::HashSet::new();
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
        write_wav(&out_path, &out).map_err(|e| format!("writing '{}': {e}", out_path.display()))?;
        count += 1;
    }
    if count == 0 {
        return Err(format!("no files matched glob '{pattern}'"));
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

// --- realtime `play` / `record` (feature `realtime`, CPAL) ------------------------------------

#[cfg(not(feature = "realtime"))]
mod realtime {
    fn unavailable() -> Result<(), String> {
        Err("realtime `play`/`record` need the `realtime` feature — \
             build/install with `--features realtime`"
            .into())
    }
    pub fn play(_: &[String], _: Option<u32>) -> Result<(), String> {
        unavailable()
    }
    pub fn record(_: &[String], _: f32) -> Result<(), String> {
        unavailable()
    }
}

#[cfg(feature = "realtime")]
mod realtime {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};
    use std::time::{Duration, Instant};

    use fluxion::{Signal, process};
    use fluxion_rt::channel;
    use fluxion_rt::cpal_backend::{
        default_input_config, default_output_sample_rate, run_input, run_output,
    };

    use super::{load_input, parse_chain, write_output};

    /// `fluxion play <in.wav> [effect...]` — process a file through the chain and play it live.
    pub fn play(args: &[String], fs: Option<u32>) -> Result<(), String> {
        let input = args
            .first()
            .ok_or("usage: fluxion play <in.wav> [effect...]")?;
        let mut signal = load_input(input)?;
        if let Some(fs) = fs {
            signal.fs = fs;
        }
        let graph = parse_chain(&args[1..])?;
        play_signal(&process(&graph, &signal))
    }

    fn play_signal(signal: &Signal) -> Result<(), String> {
        if signal.channels.iter().all(|c| c.is_empty()) {
            return Ok(());
        }
        let device_fs = default_output_sample_rate().map_err(|e| e.to_string())?;
        // Resample each channel to the device rate so playback is at the correct speed.
        let src: Arc<Vec<Vec<f32>>> = Arc::new(
            signal
                .channels
                .iter()
                .map(|c| resample_linear(c, signal.fs, device_fs))
                .collect(),
        );
        let (nframes, nsrc) = (src.iter().map(Vec::len).max().unwrap_or(0), src.len());

        let cursor = Arc::new(AtomicUsize::new(0));
        let done = Arc::new(AtomicBool::new(false));
        let (s_cb, cur_cb, done_cb) = (src.clone(), cursor.clone(), done.clone());

        let stream = run_output(move |buf, ch| {
            let mut pos = cur_cb.load(Relaxed);
            for frame in buf.chunks_mut(ch) {
                if pos >= nframes {
                    frame.iter_mut().for_each(|s| *s = 0.0);
                    done_cb.store(true, Relaxed);
                    continue;
                }
                for (c, s) in frame.iter_mut().enumerate() {
                    // Map device channel → source channel: mono replicates; extra device channels
                    // reuse the last source channel.
                    *s = s_cb[c.min(nsrc - 1)].get(pos).copied().unwrap_or(0.0);
                }
                pos += 1;
            }
            cur_cb.store(pos, Relaxed);
        })
        .map_err(|e| e.to_string())?;

        eprintln!(
            "fluxion: playing {:.1}s @ {device_fs} Hz",
            nframes as f32 / device_fs as f32
        );
        while !done.load(Relaxed) {
            std::thread::sleep(Duration::from_millis(50));
        }
        std::thread::sleep(Duration::from_millis(120)); // let the final block flush
        drop(stream);
        Ok(())
    }

    /// `fluxion record [effect...] <out.wav>` — capture `--secs` from the default input, process it,
    /// and write it. Capture crosses the audio thread through the lock-free ring.
    pub fn record(args: &[String], secs: f32) -> Result<(), String> {
        let out = args
            .last()
            .ok_or("usage: fluxion record [effect...] <out.wav> [--secs N]")?;
        let graph = parse_chain(&args[..args.len() - 1])?;
        let secs = secs.max(0.1);

        let (dev_fs, channels) = default_input_config().map_err(|e| e.to_string())?;
        let want = (secs * dev_fs as f32) as usize * channels;
        // Ring holds ~1 s of slack beyond the target so a slow drain never overflows.
        let (mut tx, mut rx) = channel::<f32>(want + dev_fs as usize * channels);
        let stream = run_input(move |data, _| {
            for &s in data {
                let _ = tx.push(s); // drop on overflow (shouldn't happen with the slack above)
            }
        })
        .map_err(|e| e.to_string())?;

        eprintln!("fluxion: recording {secs:.1}s @ {dev_fs} Hz ({channels} ch)…");
        let mut captured = Vec::with_capacity(want);
        let start = Instant::now();
        while captured.len() < want && start.elapsed().as_secs_f32() < secs + 1.0 {
            while let Some(s) = rx.pop() {
                captured.push(s);
                if captured.len() >= want {
                    break;
                }
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        drop(stream);
        captured.truncate(want);

        // De-interleave into channels, then process + write at the capture rate.
        let mut chans = vec![Vec::with_capacity(want / channels); channels];
        for frame in captured.chunks_exact(channels) {
            for (c, &s) in frame.iter().enumerate() {
                chans[c].push(s);
            }
        }
        write_output(out, &process(&graph, &Signal::new(dev_fs, chans)))
    }

    /// Linear-interpolation resample of one channel from `from_fs` to `to_fs`.
    ///
    /// ponytail: linear interpolation — fine for CLI playback; upgrade to polyphase/sinc SRC only if
    /// high-fidelity conversion is needed.
    fn resample_linear(x: &[f32], from_fs: u32, to_fs: u32) -> Vec<f32> {
        if from_fs == to_fs || x.len() < 2 {
            return x.to_vec();
        }
        let ratio = from_fs as f64 / to_fs as f64; // input samples per output sample
        let n_out = ((x.len() as f64) / ratio).round() as usize;
        (0..n_out)
            .map(|i| {
                let pos = i as f64 * ratio;
                let j = pos.floor() as usize;
                let frac = (pos - j as f64) as f32;
                let a = x.get(j).copied().unwrap_or(0.0);
                let b = x.get(j + 1).copied().unwrap_or(a);
                a + (b - a) * frac
            })
            .collect()
    }

    #[cfg(test)]
    mod tests {
        use super::resample_linear;

        #[test]
        fn resample_identity_when_rates_match() {
            let x = vec![0.1, -0.2, 0.3, 0.4];
            assert_eq!(resample_linear(&x, 48_000, 48_000), x);
        }

        #[test]
        fn resample_upsamples_a_ramp_linearly() {
            // x = 0,1,2,3 at 1 Hz → 2 Hz reads at positions 0,0.5,1,…: 0,0.5,1,1.5,2,2.5,3,3.
            let up = resample_linear(&[0.0, 1.0, 2.0, 3.0], 1, 2);
            assert_eq!(up.len(), 8);
            for (i, &v) in up.iter().enumerate() {
                let want = (i as f32 * 0.5).min(3.0);
                assert!((v - want).abs() < 1e-6, "at {i}: {v} vs {want}");
            }
        }

        #[test]
        fn resample_downsample_halves_length() {
            let down = resample_linear(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 2, 1);
            assert_eq!(down.len(), 3); // 6 input / 2 ≈ 3 output
            assert!((down[0] - 0.0).abs() < 1e-6 && (down[1] - 2.0).abs() < 1e-6);
        }
    }
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

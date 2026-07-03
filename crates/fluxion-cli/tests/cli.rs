//! End-to-end CLI tests (plan task I10): drive the built `fluxion` binary and check its output.

use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_fluxion")
}

fn tmp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("fxcli_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn write_wav(path: &Path, fs: u32, samples: &[f32]) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: fs,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut w = hound::WavWriter::create(path, spec).unwrap();
    for &s in samples {
        w.write_sample(s).unwrap();
    }
    w.finalize().unwrap();
}

fn read_samples(path: &Path) -> Vec<f32> {
    hound::WavReader::open(path)
        .unwrap()
        .samples::<f32>()
        .map(|s| s.unwrap())
        .collect()
}

fn wav_spec(path: &Path) -> hound::WavSpec {
    hound::WavReader::open(path).unwrap().spec()
}

/// Pull the numeric value from a `stat` line whose trimmed text starts with `label` and ends in a
/// unit (e.g. the `peak` / `RMS` dBFS lines).
fn stat_field(text: &str, label: &str) -> f32 {
    let line = text
        .lines()
        .find(|l| l.trim_start().starts_with(label))
        .unwrap_or_else(|| panic!("no '{label}' line in:\n{text}"));
    let value = line
        .split(':')
        .nth(1)
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap();
    value.parse().unwrap()
}

#[test]
fn process_applies_gain() {
    let d = tmp("gain");
    let (inp, outp) = (d.join("in.wav"), d.join("out.wav"));
    write_wav(&inp, 48_000, &[1.0, -2.0, 3.0]);

    let st = Command::new(bin())
        .args([
            inp.to_str().unwrap(),
            "gain",
            "--gain",
            "0.5",
            outp.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(st.success());

    let s = read_samples(&outp);
    assert_eq!(s.len(), 3);
    for (got, want) in s.iter().zip(&[0.5, -1.0, 1.5]) {
        assert!((got - want).abs() < 1e-6, "{got} vs {want}");
    }
    std::fs::remove_dir_all(&d).ok();
}

#[test]
fn info_prints_metadata() {
    let d = tmp("info");
    let inp = d.join("in.wav");
    write_wav(&inp, 44_100, &[0.0; 441]);

    for verb in ["info", "soxi"] {
        let out = Command::new(bin())
            .args([verb, inp.to_str().unwrap()])
            .output()
            .unwrap();
        assert!(out.status.success(), "{verb} failed");
        let text = String::from_utf8_lossy(&out.stdout);
        assert!(text.contains("44100"), "{verb}: no sample rate in:\n{text}");
        assert!(text.contains("channels"), "{verb}: no channels");
    }
    std::fs::remove_dir_all(&d).ok();
}

#[test]
fn stdin_stdout_pipe() {
    let d = tmp("pipe");
    let inp = d.join("in.wav");
    write_wav(&inp, 48_000, &[1.0, 2.0, 3.0]);

    // `fluxion - gain --gain 2 -` : WAV on stdin, WAV on stdout.
    let out = Command::new(bin())
        .args(["-", "gain", "--gain", "2", "-"])
        .stdin(Stdio::from(std::fs::File::open(&inp).unwrap()))
        .stdout(Stdio::piped())
        .output()
        .unwrap();
    assert!(out.status.success());

    let s: Vec<f32> = hound::WavReader::new(Cursor::new(out.stdout))
        .unwrap()
        .samples::<f32>()
        .map(|x| x.unwrap())
        .collect();
    for (got, want) in s.iter().zip(&[2.0, 4.0, 6.0]) {
        assert!((got - want).abs() < 1e-6, "{got} vs {want}");
    }
    std::fs::remove_dir_all(&d).ok();
}

#[test]
fn null_sink_writes_nothing() {
    let d = tmp("null");
    let inp = d.join("in.wav");
    write_wav(&inp, 48_000, &[0.5; 8]);

    let st = Command::new(bin())
        .args([inp.to_str().unwrap(), "gain", "--gain", "0.5", "-n"])
        .status()
        .unwrap();
    assert!(st.success());
    // `-n` must not create a file named "-n".
    assert!(!Path::new("-n").exists());
    std::fs::remove_dir_all(&d).ok();
}

#[test]
fn batch_processes_a_glob() {
    let d = tmp("batch");
    let (indir, outdir) = (d.join("in"), d.join("out"));
    std::fs::create_dir_all(&indir).unwrap();
    write_wav(&indir.join("a.wav"), 48_000, &[1.0, 1.0]);
    write_wav(&indir.join("b.wav"), 48_000, &[2.0, 2.0]);

    let pattern = format!("{}/*.wav", indir.to_str().unwrap());
    let st = Command::new(bin())
        .args([
            "batch",
            outdir.to_str().unwrap(),
            &pattern,
            "gain",
            "--gain",
            "0",
        ])
        .status()
        .unwrap();
    assert!(st.success());

    // Both files written; gain 0 → silence.
    for name in ["a.wav", "b.wav"] {
        let s = read_samples(&outdir.join(name));
        assert!(s.iter().all(|&v| v == 0.0), "{name} not silenced");
    }
    std::fs::remove_dir_all(&d).ok();
}

#[test]
fn process_refuses_in_equals_out() {
    let d = tmp("inout");
    let f = d.join("x.wav");
    write_wav(&f, 48_000, &[0.5; 4]);
    let st = Command::new(bin())
        .args([
            f.to_str().unwrap(),
            "gain",
            "--gain",
            "1",
            f.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(
        !st.success(),
        "writing the result over the input must be refused"
    );
    std::fs::remove_dir_all(&d).ok();
}

#[test]
fn batch_empty_glob_errors() {
    let d = tmp("emptyglob");
    let pattern = format!("{}/no_such_*.wav", d.to_str().unwrap());
    let st = Command::new(bin())
        .args([
            "batch",
            d.join("out").to_str().unwrap(),
            &pattern,
            "gain",
            "--gain",
            "1",
        ])
        .status()
        .unwrap();
    assert!(
        !st.success(),
        "a glob matching nothing must error (not silent success)"
    );
    std::fs::remove_dir_all(&d).ok();
}

#[test]
fn trim_rate_bits_roundtrip() {
    // 1 s of audio at 48 kHz -> trim to the middle 0.5 s -> resample to 24 kHz -> 16-bit PCM.
    let d = tmp("trimrate");
    let (inp, outp) = (d.join("in.wav"), d.join("out.wav"));
    let samples: Vec<f32> = (0..48_000).map(|i| 0.5 * (i as f32 * 0.01).sin()).collect();
    write_wav(&inp, 48_000, &samples);

    let st = Command::new(bin())
        .args([
            "--bits",
            "16",
            inp.to_str().unwrap(),
            "trim",
            "--start",
            "0.25",
            "--len",
            "0.5",
            "rate",
            "--fs",
            "24000",
            outp.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(st.success());

    let spec = wav_spec(&outp);
    assert_eq!(spec.sample_rate, 24_000, "rate stage must change fs");
    assert_eq!(spec.bits_per_sample, 16, "--bits 16 must select 16-bit PCM");
    assert_eq!(spec.sample_format, hound::SampleFormat::Int);
    // 0.5 s trimmed at 48 kHz (24000 frames) resampled to 24 kHz -> ~12000 frames. Read the frame
    // count via the header (the samples are 16-bit int, not f32).
    let frames = hound::WavReader::open(&outp).unwrap().duration();
    assert!(
        (frames as i64 - 12_000).abs() <= 4,
        "expected ~12000 frames, got {frames}"
    );
    std::fs::remove_dir_all(&d).ok();
}

#[test]
fn multi_input_concat_and_mix() {
    let d = tmp("multi");
    let (a, b) = (d.join("a.wav"), d.join("b.wav"));
    write_wav(&a, 48_000, &[1.0, 1.0]);
    write_wav(&b, 48_000, &[2.0, 2.0]);

    // Default: concatenate in time.
    let cat = d.join("cat.wav");
    let st = Command::new(bin())
        .args([
            a.to_str().unwrap(),
            b.to_str().unwrap(),
            cat.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(st.success());
    assert_eq!(read_samples(&cat), vec![1.0, 1.0, 2.0, 2.0]);

    // --mix: sum sample-by-sample.
    let mixed = d.join("mix.wav");
    let st = Command::new(bin())
        .args([
            "--mix",
            a.to_str().unwrap(),
            b.to_str().unwrap(),
            mixed.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(st.success());
    assert_eq!(read_samples(&mixed), vec![3.0, 3.0]);
    std::fs::remove_dir_all(&d).ok();
}

#[test]
fn multi_input_rate_mismatch_needs_rate_flag() {
    let d = tmp("ratemismatch");
    let (a, b, out) = (d.join("a.wav"), d.join("b.wav"), d.join("out.wav"));
    write_wav(&a, 48_000, &[0.1; 48]);
    write_wav(&b, 44_100, &[0.2; 44]);

    // Differing rates without --rate must error.
    let st = Command::new(bin())
        .args([
            a.to_str().unwrap(),
            b.to_str().unwrap(),
            out.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(
        !st.success(),
        "mismatched input rates must error without --rate"
    );

    // With --rate the inputs are resampled to a common rate and combine cleanly.
    let st = Command::new(bin())
        .args([
            "--rate",
            "48000",
            a.to_str().unwrap(),
            b.to_str().unwrap(),
            out.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(st.success());
    assert_eq!(wav_spec(&out).sample_rate, 48_000);
    std::fs::remove_dir_all(&d).ok();
}

#[test]
fn synth_sine_then_stat() {
    let d = tmp("synthstat");
    let out = d.join("tone.wav");
    let st = Command::new(bin())
        .args([
            "synth",
            "--wave",
            "sine",
            "--freq",
            "1000",
            "--secs",
            "1",
            "--fs",
            "48000",
            out.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(st.success());
    assert_eq!(wav_spec(&out).sample_rate, 48_000);
    assert!((read_samples(&out).len() as i64 - 48_000).abs() <= 1);

    let text = String::from_utf8_lossy(
        &Command::new(bin())
            .args(["stat", out.to_str().unwrap()])
            .output()
            .unwrap()
            .stdout,
    )
    .into_owned();

    // A full-scale sine: peak ~0 dBFS, RMS ~ -3.01 dBFS.
    let peak = stat_field(&text, "peak");
    let rms = stat_field(&text, "RMS");
    assert!(
        (-1.0..=0.1).contains(&peak),
        "peak dBFS out of range: {peak}"
    );
    assert!((-3.5..=-2.5).contains(&rms), "RMS dBFS out of range: {rms}");
    std::fs::remove_dir_all(&d).ok();
}

#[test]
fn effects_listing_is_discoverable() {
    // Full listing names a known op with its param, plus the geometry stages.
    let text = String::from_utf8_lossy(
        &Command::new(bin())
            .args(["effects"])
            .output()
            .unwrap()
            .stdout,
    )
    .into_owned();
    assert!(text.contains("lowpass"), "effects must list lowpass");
    assert!(
        text.contains("cutoff"),
        "effects must list the cutoff param"
    );
    assert!(text.contains("trim"), "effects must list the trim stage");

    // A single-name query prints just that op.
    let one = Command::new(bin())
        .args(["effects", "lowpass"])
        .output()
        .unwrap();
    assert!(one.status.success());
    assert!(String::from_utf8_lossy(&one.stdout).contains("cutoff"));
}

#[test]
fn gain_db_halves_amplitude() {
    let d = tmp("gaindb");
    let (inp, outp) = (d.join("in.wav"), d.join("out.wav"));
    write_wav(&inp, 48_000, &[1.0, 1.0, 1.0]);

    let st = Command::new(bin())
        .args([
            inp.to_str().unwrap(),
            "gain",
            "--db",
            "-6",
            outp.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(st.success());
    // 10^(-6/20) = 0.5012 — "-6 dB halves the amplitude".
    for v in read_samples(&outp) {
        assert!(
            (v - 0.5).abs() < 0.02,
            "gain --db -6 gave {v}, expected ~0.5"
        );
    }
    std::fs::remove_dir_all(&d).ok();
}

#[test]
fn zero_param_reverse_flips_time() {
    let d = tmp("reverse");
    let (inp, outp) = (d.join("in.wav"), d.join("out.wav"));
    write_wav(&inp, 48_000, &[1.0, 2.0, 3.0]);

    // `reverse` takes no params — it must parse cleanly and flip the buffer.
    let st = Command::new(bin())
        .args([inp.to_str().unwrap(), "reverse", outp.to_str().unwrap()])
        .status()
        .unwrap();
    assert!(st.success());
    assert_eq!(read_samples(&outp), vec![3.0, 2.0, 1.0]);
    std::fs::remove_dir_all(&d).ok();
}

#[test]
fn si_suffix_number_parsing() {
    // `--cutoff 1k` must equal `--cutoff 1000` — exercise SI parsing end-to-end via compile.
    let d = tmp("si");
    let (g1, g2) = (d.join("k.fxg"), d.join("plain.fxg"));
    for (cut, out) in [("1k", &g1), ("1000", &g2)] {
        let st = Command::new(bin())
            .args(["compile", "lowpass", "--cutoff", cut, out.to_str().unwrap()])
            .status()
            .unwrap();
        assert!(st.success(), "compile lowpass --cutoff {cut} failed");
    }
    assert_eq!(
        std::fs::read_to_string(&g1).unwrap(),
        std::fs::read_to_string(&g2).unwrap(),
        "--cutoff 1k must produce the same graph as --cutoff 1000"
    );
    std::fs::remove_dir_all(&d).ok();
}

#[test]
fn nonpositive_speed_and_zero_channels_are_clean_errors() {
    // Regression: `speed --factor 0` used to explode the frame count (OOM) and
    // `channels --count 0` used to panic the WAV writer (exit 101).
    let d = tmp("badgeom");
    let (inp, outp) = (d.join("in.wav"), d.join("out.wav"));
    write_wav(&inp, 48_000, &[0.1, 0.2, 0.3, 0.4]);

    for stage in [
        &["speed", "--factor", "0"][..],
        &["channels", "--count", "0"][..],
    ] {
        let out = Command::new(bin())
            .arg(inp.to_str().unwrap())
            .args(stage)
            .arg(outp.to_str().unwrap())
            .output()
            .unwrap();
        assert_eq!(
            out.status.code(),
            Some(1),
            "stage {stage:?} must fail cleanly"
        );
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(
            err.starts_with("fluxion:"),
            "expected a clean CLI error, got: {err}"
        );
    }
    std::fs::remove_dir_all(&d).ok();
}

#[test]
fn fxg_splices_into_the_default_pipeline() {
    // Regression: the documented `fluxion in.wav chain.fxg out.wav` — a pre-existing .fxg file
    // must splice into the chain, not be swallowed as a second audio input.
    let d = tmp("fxg_splice");
    let (inp, fxg, outp) = (d.join("in.wav"), d.join("chain.fxg"), d.join("out.wav"));
    write_wav(&inp, 48_000, &[1.0, -2.0, 3.0]);

    let st = Command::new(bin())
        .args(["compile", "gain", "--gain", "0.5", fxg.to_str().unwrap()])
        .status()
        .unwrap();
    assert!(st.success());

    let st = Command::new(bin())
        .args([
            inp.to_str().unwrap(),
            fxg.to_str().unwrap(),
            outp.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(
        st.success(),
        "in.wav chain.fxg out.wav must run the spliced graph"
    );
    assert_eq!(read_samples(&outp), vec![0.5, -1.0, 1.5]);
    std::fs::remove_dir_all(&d).ok();
}

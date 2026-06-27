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

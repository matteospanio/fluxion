//! Run the CLI quickstart, so it cannot rot (ROADMAP I7).
//!
//! The quickstart itself is `tests/quickstart.sh` — real shell, the way a user would type it —
//! executed here with `$FLUXION` bound to the freshly built binary and a temp directory as cwd.

use std::path::PathBuf;
use std::process::Command;

/// A private directory per test — `cargo test` runs them in parallel, so a shared name means one
/// test deleting another's files.
fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("fxquick_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Post-condition: every line of the quickstart succeeds, and the two ways of writing the same
/// chain — argv flags and `--chain` text — produce the same audio, byte for byte.
#[test]
fn the_quickstart_runs() {
    let dir = scratch("quickstart");
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/quickstart.sh");

    let out = Command::new("sh")
        .arg(&script)
        .current_dir(&dir)
        .env("FLUXION", env!("CARGO_BIN_EXE_fluxion"))
        .output()
        .expect("sh runs the quickstart");

    assert!(
        out.status.success(),
        "the quickstart failed:\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );

    let inline = std::fs::read(dir.join("out.wav")).expect("out.wav");
    let text = std::fs::read(dir.join("same.wav")).expect("same.wav");
    assert_eq!(
        inline, text,
        "`--chain \"highpass(80, 4) | gain(-3dB)\"` did not match the same chain in argv flags"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// Progress writes carriage returns, which would wreck a log or a snapshot. It has to stay off
/// whenever stderr is not a terminal — which, in a test, it never is.
#[test]
fn progress_is_silent_when_stderr_is_not_a_terminal() {
    let dir = scratch("progress");
    let long = dir.join("long.wav");

    // 30 s at 48 kHz: comfortably past the threshold at which progress would appear on a tty.
    let synth = Command::new(env!("CARGO_BIN_EXE_fluxion"))
        .args(["synth", "--wave", "sine", "--freq", "220", "--secs", "30"])
        .arg(&long)
        .output()
        .unwrap();
    assert!(synth.status.success());

    let run = Command::new(env!("CARGO_BIN_EXE_fluxion"))
        .arg(&long)
        .args(["lowpass", "--cutoff", "1k", "-n"])
        .output()
        .unwrap();
    assert!(run.status.success());
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        !stderr.contains('\r'),
        "progress leaked into a non-terminal stderr: {stderr:?}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

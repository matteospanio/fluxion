//! What the CLI says, pinned.
//!
//! Help text and error messages are the interface for anyone at a terminal, and they rot silently:
//! nothing fails when an error stops being useful. So the ten mistakes a user actually makes get
//! committed expectations, and a diff in one of them is a review decision rather than an accident.
//!
//! To accept a deliberate change:
//!
//! ```text
//! UPDATE_EXPECT=1 cargo test -p fluxion-cli --test cli_snapshots
//! ```
//!
//! No snapshot library: the whole mechanism is `expect()` below. `insta` would add a
//! dev-dependency, a companion binary and a review workflow to a repo that has none of that, and
//! CONTRIBUTING asks for minimal deps.

use std::path::{Path, PathBuf};
use std::process::Command;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_fluxion")
}

fn dir() -> PathBuf {
    let d = std::env::temp_dir().join(format!("fxsnap_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Run the binary and return `stdout + stderr`, with the machine-specific bits scrubbed.
fn run(args: &[&str]) -> String {
    let out = Command::new(bin())
        .args(args)
        .current_dir(dir())
        .output()
        .expect("the fluxion binary runs");
    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    scrub(&text)
}

/// Replace what changes between machines and releases, so a snapshot means only what it should.
fn scrub(text: &str) -> String {
    let tmp = dir();
    text.replace(tmp.to_str().unwrap(), "<TMP>")
        .replace(env!("CARGO_PKG_VERSION"), "<VERSION>")
}

/// Compare against `tests/expected/<name>.txt`, or rewrite it when `UPDATE_EXPECT` is set.
fn expect(name: &str, actual: &str) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/expected")
        .join(format!("{name}.txt"));
    if std::env::var_os("UPDATE_EXPECT").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, actual).unwrap();
        return;
    }
    let expected = std::fs::read_to_string(&path).unwrap_or_default();
    assert_eq!(
        actual.trim_end(),
        expected.trim_end(),
        "snapshot `{name}` changed.\n\
         If the new output is better, accept it with:\n  \
         UPDATE_EXPECT=1 cargo test -p fluxion-cli --test cli_snapshots\n"
    );
}

// --- help ---------------------------------------------------------------------------------

#[test]
fn help_snapshots() {
    expect("help_top", &run(&["--help"]));
    // Running `fluxion` with nothing to do is a request for help, not a usage error.
    expect("help_bare", &run(&[]));
    expect("effects_listing", &run(&["effects"]));
    expect("effects_one", &run(&["effects", "compand"]));
}

/// The help screen has to stay a screen. If a new flag pushes it over, shorten something — the
/// point of `--help` is to be read, and `fluxion effects` is where detail belongs.
#[test]
fn help_fits_one_screen() {
    let help = run(&["--help"]);
    let lines = help.lines().count();
    assert!(lines <= 40, "help is {lines} lines; the budget is 40");
    for line in help.lines() {
        let width = line.chars().count();
        assert!(
            width <= 80,
            "help line is {width} columns (budget 80): {line}"
        );
    }
}

/// `fluxion` and `fluxion help` are the same thing as `--help`, and neither is an error.
#[test]
fn no_arguments_is_not_an_error() {
    for args in [vec![], vec!["help"]] {
        let out = Command::new(bin()).args(&args).output().unwrap();
        assert!(out.status.success(), "`fluxion {args:?}` exited non-zero");
        assert_eq!(
            scrub(&String::from_utf8_lossy(&out.stdout)),
            run(&["--help"])
        );
    }
}

// --- the ten common mistakes --------------------------------------------------------------

#[test]
fn error_snapshots() {
    // A misspelled effect, and a misspelled parameter of a real one.
    expect("err_unknown_effect", &run(&["in.wav", "hipass", "out.wav"]));
    expect(
        "err_unknown_param",
        &run(&["in.wav", "lowpass", "--cutof", "800", "out.wav"]),
    );
    // A flag with nothing after it, and a value that is not a number.
    expect(
        "err_missing_value",
        &run(&["in.wav", "lowpass", "--cutoff", "out.wav"]),
    );
    expect(
        "err_bad_number",
        &run(&["in.wav", "lowpass", "--cutoff", "abc", "out.wav"]),
    );
    // A value the op refuses.
    expect(
        "err_out_of_range",
        &run(&["in.wav", "lowpass", "--cutoff", "-5", "out.wav"]),
    );
    // An input that is not there — and one that is the output.
    expect("err_missing_input", &run(&["nope.wav", "out.wav"]));
    expect("err_same_file", &run(&["same.wav", "same.wav"]));
    // The chain text: a structural error and a name error, both with a caret.
    expect(
        "err_chain_syntax",
        &run(&["--chain", "highpass(80 | gain(2)", "in.wav", "out.wav"]),
    );
    expect(
        "err_chain_unknown_op",
        &run(&["--chain", "gian(2)", "in.wav", "out.wav"]),
    );
    // Two ways of saying the same thing at once.
    expect(
        "err_chain_and_inline",
        &run(&["--chain", "gain(2)", "in.wav", "lowpass", "out.wav"]),
    );
    // A geometry stage where only a filter graph can go.
    expect(
        "err_geometry_in_compile",
        &run(&["compile", "trim", "--start", "1", "out.fxg"]),
    );
    // An op name typed on its own: show what it is rather than failing to open it as audio.
    expect("op_name_alone", &run(&["lowpass"]));
}

// --- --dry-run ----------------------------------------------------------------------------

#[test]
fn dry_run_snapshots() {
    expect(
        "dry_run_inline",
        &run(&[
            "--dry-run",
            "in.wav",
            "lowpass",
            "--cutoff",
            "1k",
            "gain",
            "--db",
            "-6",
            "out.wav",
        ]),
    );
    expect(
        "dry_run_chain",
        &run(&[
            "--dry-run",
            "--chain",
            "highpass(80, 4) | gain(-3dB)",
            "in.wav",
            "out.wav",
        ]),
    );
    expect(
        "dry_run_geometry",
        &run(&[
            "--dry-run",
            "in.wav",
            "trim",
            "--start",
            "0.5",
            "rate",
            "--fs",
            "44100",
            "lowpass",
            "out.wav",
        ]),
    );
}

/// `--dry-run` prints the chain in the shared syntax, so its output is something you can paste
/// back into `--chain`, into Python, or into the browser. That is the whole point of a canonical
/// form, and it is worth checking rather than assuming.
#[test]
fn the_dry_run_chain_can_be_pasted_back() {
    let text = run(&[
        "--dry-run",
        "in.wav",
        "lowpass",
        "--cutoff",
        "1k",
        "gain",
        "--db",
        "-6",
        "out.wav",
    ]);
    let chain = text
        .lines()
        .find_map(|l| l.strip_prefix("run: "))
        .expect("a run: line");
    let again = run(&["--dry-run", "--chain", chain, "in.wav", "out.wav"]);
    assert_eq!(text, again, "the printed chain did not reproduce itself");
}

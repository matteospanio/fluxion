//! The CLI covers the op registry, exactly.
//!
//! `fluxion effects` is what a terminal user discovers ops with, and the argv parser is what they
//! type. Both are driven from `OpKind`, so this is a tripwire rather than a second list: if an op
//! is ever added behind a feature or skipped in the listing, this goes red instead of the op simply
//! being invisible.

use std::process::Command;

use fluxion::OpKind;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_fluxion")
}

fn run(args: &[&str]) -> (bool, String) {
    let out = Command::new(bin())
        .args(args)
        .output()
        .expect("the fluxion binary runs");
    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    (out.status.success(), text)
}

/// Post-condition: every op in the registry is listed by `fluxion effects` and described by
/// `fluxion effects <name>`. An op a user cannot find is an op the CLI does not really have.
#[test]
fn the_effects_listing_covers_the_registry() {
    let (ok, listing) = run(&["effects"]);
    assert!(ok, "`fluxion effects` failed:\n{listing}");

    for &kind in OpKind::all() {
        assert!(
            listing.contains(kind.name()),
            "`fluxion effects` does not list '{}'",
            kind.name()
        );
        let (ok, described) = run(&["effects", kind.name()]);
        assert!(ok, "`fluxion effects {}` failed:\n{described}", kind.name());
        assert!(
            described.contains(kind.name()),
            "`fluxion effects {}` did not describe it:\n{described}",
            kind.name()
        );
    }
}

/// Post-condition: every parameter the registry declares is shown with the op, under the flag name
/// the parser accepts. This is what keeps `--cutoff` from drifting away from `ParamSpec::name`.
#[test]
fn the_listing_shows_every_parameter_under_its_real_flag() {
    for &kind in OpKind::all() {
        // `fir` is variadic: the parser takes `--taps a,b,c`, so it documents that instead of the
        // single `tap` prototype spec. Every other op lists its schema verbatim.
        if kind.is_variadic() {
            continue;
        }
        let (_, described) = run(&["effects", kind.name()]);
        for spec in kind.params() {
            assert!(
                described.contains(&format!("--{}", spec.name)),
                "`fluxion effects {}` omits '--{}':\n{described}",
                kind.name(),
                spec.name
            );
        }
    }
}

/// Post-condition: an unknown name is refused, so the listing above is a real catalogue and not a
/// list the parser happens to ignore.
#[test]
fn an_unknown_name_is_refused() {
    let (ok, message) = run(&["effects", "nosuchop"]);
    assert!(!ok, "`fluxion effects nosuchop` should fail");
    assert!(message.contains("nosuchop"), "{message}");
}

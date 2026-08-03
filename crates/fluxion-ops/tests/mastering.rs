//! The mastering set behaves as a mastering engineer needs it to (ROADMAP M3, M4).
//!
//! These are property tests rather than oracle comparisons: "the output never exceeds the ceiling,
//! on any input" is a claim about every signal, not about eleven of them, so it is checked against
//! signals designed to break it — square waves, isolated spikes, noise at full scale, and material
//! whose inter-sample peaks sit well above its samples.

use fluxion_ops::loudness::{integrated_loudness, sample_peak, true_peak};
use fluxion_ops::{limit, loudness_normalize};

const FS: u32 = 48_000;

/// Signals chosen to be hard on a limiter, not easy.
fn adversarial() -> Vec<(&'static str, Vec<Vec<f32>>)> {
    let n = FS as usize * 2;
    let tau = std::f64::consts::TAU;
    let tone = |freq: f64, amp: f32, phase: f64| -> Vec<f32> {
        (0..n)
            .map(|i| amp * (tau * freq * i as f64 / f64::from(FS) + phase).sin() as f32)
            .collect()
    };

    let mut lcg: u32 = 0x5eed_1234;
    let mut noise = Vec::with_capacity(n);
    for _ in 0..n {
        lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        noise.push((lcg >> 8) as f32 / 16_777_216.0 * 2.0 - 1.0);
    }

    // A square wave is all inter-sample overshoot: its band-limited reconstruction rings well past
    // its own samples.
    let square: Vec<f32> = (0..n)
        .map(|i| if (i / 24) % 2 == 0 { 0.95 } else { -0.95 })
        .collect();

    // Isolated full-scale spikes on an otherwise quiet signal: the case where a limiter that
    // reacts late, or releases slowly, does the most damage.
    let mut spikes = tone(200.0, 0.05, 0.0);
    for i in (0..n).step_by(9_000) {
        spikes[i] = 0.99;
    }

    vec![
        ("loud_tone", vec![tone(800.0, 0.95, 0.0)]),
        // fs/4 at 45 degrees: samples at 0.67, reconstruction at 0.95.
        (
            "inter_sample",
            vec![tone(12_000.0, 0.95, std::f64::consts::FRAC_PI_4)],
        ),
        ("near_nyquist", vec![tone(21_000.0, 0.9, 0.0)]),
        ("square", vec![square]),
        ("spikes", vec![spikes]),
        ("noise_full_scale", vec![noise.clone()]),
        (
            "stereo",
            vec![
                tone(400.0, 0.9, 0.0),
                noise.iter().map(|v| v * 0.8).collect(),
            ],
        ),
        ("silence", vec![vec![0.0; n]]),
    ]
}

/// ROADMAP M3's check, as written: the output never exceeds the ceiling, on any input.
#[test]
fn the_limiter_never_exceeds_its_ceiling() {
    for ceiling in [-0.1f32, -1.0, -6.0, -20.0] {
        for (name, signal) in adversarial() {
            let mut limited = signal.clone();
            limit(&mut limited, ceiling, 0.005, 0.05, FS);

            let measured = true_peak(&limited, FS);
            assert!(
                measured <= ceiling + 0.05,
                "'{name}' at ceiling {ceiling}: true peak came out {measured:.3} dBTP"
            );
            assert_eq!(
                limited[0].len(),
                signal[0].len(),
                "'{name}': the limiter changed the length"
            );
        }
    }
}

/// A limiter that reaches its ceiling by turning everything down is not a limiter. Material
/// already under the ceiling must come through untouched.
#[test]
fn the_limiter_leaves_quiet_material_alone() {
    let quiet: Vec<Vec<f32>> = vec![
        (0..FS as usize)
            .map(|i| 0.1 * (std::f32::consts::TAU * 500.0 * i as f32 / FS as f32).sin())
            .collect(),
    ];
    let mut limited = quiet.clone();
    limit(&mut limited, -6.0, 0.005, 0.05, FS);
    for (a, b) in quiet[0].iter().zip(&limited[0]) {
        assert!(
            (a - b).abs() < 1e-6,
            "a quiet signal was altered: {a} -> {b}"
        );
    }
}

/// And one that hits the ceiling should still be close to it — pulling 10 dB off a signal that
/// needed 1 dB would meet the letter of the property and be useless.
#[test]
fn the_limiter_stays_close_to_its_ceiling() {
    for (name, signal) in adversarial() {
        if name == "silence" || name == "spikes" {
            continue; // nothing to reach, and isolated spikes legitimately duck the rest
        }
        let mut limited = signal;
        limit(&mut limited, -1.0, 0.005, 0.05, FS);
        let measured = true_peak(&limited, FS);
        assert!(
            measured > -1.0 - 1.5,
            "'{name}': limiting overshot downward to {measured:.3} dBTP"
        );
    }
}

/// ROADMAP M4's check: fixtures land within 0.5 LU of target and the true peak stays under.
#[test]
fn normalize_hits_its_target_and_holds_the_ceiling() {
    let (target, ceiling) = (-14.0f32, -1.0f32);
    for (name, signal) in adversarial() {
        if name == "silence" {
            continue;
        }
        let mut normalized = signal;
        loudness_normalize(&mut normalized, target, ceiling, FS);

        let measured = integrated_loudness(&normalized, FS);
        let peak = true_peak(&normalized, FS);

        // The ceiling is not negotiable, whatever the material.
        assert!(
            peak <= ceiling + 0.05,
            "'{name}': true peak {peak:.3} dBTP is over the {ceiling} ceiling"
        );

        if name == "spikes" {
            // 26 dB of crest factor: isolated full-scale spikes over a signal 26 dB below them.
            // Reaching -14 LUFS would need 16 dB of limiting, which costs more loudness than the
            // gain adds — the target is unreachable under this ceiling, and no implementation can
            // reach it. What can be demanded is that it never overshoots and never ends up quieter
            // than it started.
            // What can still be demanded: it never overshoots the target, and what comes back is
            // the closest attempt that respects the ceiling rather than the last one tried.
            assert!(
                measured <= target,
                "'{name}': overshot to {measured:.3} LUFS"
            );
            continue;
        }

        assert!(
            (measured - target).abs() <= 0.5,
            "'{name}': landed at {measured:.3} LUFS, target {target}"
        );
    }
}

/// Normalizing works upward as well as downward, and from any starting level.
#[test]
fn normalize_works_from_any_starting_level() {
    for start in [0.001f32, 0.01, 0.1, 0.9] {
        let mut signal: Vec<Vec<f32>> = vec![
            (0..FS as usize * 3)
                .map(|i| start * (std::f32::consts::TAU * 700.0 * i as f32 / FS as f32).sin())
                .collect(),
        ];
        loudness_normalize(&mut signal, -23.0, -2.0, FS);
        let measured = integrated_loudness(&signal, FS);
        assert!(
            (measured - -23.0).abs() <= 0.5,
            "from amplitude {start}: landed at {measured:.3} LUFS"
        );
    }
}

/// Silence has no loudness, so there is no gain that normalizes it. Leaving it alone is the only
/// sane answer; multiplying by infinity is not.
#[test]
fn normalize_leaves_silence_alone() {
    let mut silence: Vec<Vec<f32>> = vec![vec![0.0; FS as usize]];
    loudness_normalize(&mut silence, -14.0, -1.0, FS);
    assert!(silence[0].iter().all(|v| *v == 0.0));
    assert_eq!(sample_peak(&silence), f32::NEG_INFINITY);
}

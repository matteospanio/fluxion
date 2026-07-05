//! Certified hot-swap demo (IS² paper: the train→certify→freeze→push→play loop, measured).
//!
//! A paced real-time render loop (one block every `buffer/fs` seconds, like an audio
//! callback) watches a directory for pushed `.fxg` graphs. Each arrival is loaded,
//! **certified on device** (the artifact-level safety gate: a graph whose verdict is not
//! `CertifiedStable` is rejected, never played), frozen to f32 sections, and crossfaded
//! into the running cascade over the lock-free command ring — while the click bound is
//! measured as the maximum sample-to-sample step of the output, fade vs steady state.
//!
//! Emits JSON lines; writes `<name>.ack` next to an applied artifact so a remote pusher
//! can measure end-to-end push→applied latency with its own clock.
//!
//! Run: `hotswap_demo <watch-dir> [buffer] [total-blocks]`

use std::f32::consts::PI;
use std::time::{Duration, Instant};

use fluxion_backend::{certify_graph, freeze};
use fluxion_core::fxg;
use fluxion_ops::{Verdict, butterworth_lowpass};
use fluxion_rt::{RtGraph, SetCoeffs, channel};

const FS: u32 = 48_000;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let watch = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "/tmp/hotswap".into());
    let buffer: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(128);
    let total: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0); // 0 = until stop file
    let fade_blocks: u32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(16);
    std::fs::create_dir_all(&watch).expect("watch dir");

    let mut g = RtGraph::filter(butterworth_lowpass(4, 1_000.0, FS));
    g.prepare(buffer);
    let (mut tx, mut rx) = channel::<SetCoeffs>(8);

    let fade_samples = (buffer as u32 * fade_blocks).max(1); // default ~43 ms at 128/48k
    let period = Duration::from_secs_f64(buffer as f64 / FS as f64);
    let mut input = vec![0.0f32; buffer];
    let mut out = vec![0.0f32; buffer];
    let mut phase = 0.0f32;
    let w = 2.0 * PI * 440.0 / FS as f32;

    let mut seen: std::collections::HashSet<String> = Default::default();
    let mut prev_last = 0.0f32;
    let (mut max_step_fade, mut max_step_steady) = (0.0f32, 0.0f32);
    let mut fade_until: u64 = 0;
    let mut next_tick = Instant::now() + period;

    let mut blk: u64 = 0;
    loop {
        // Control side (off the audio path): poll the watch dir every 8 blocks (~21 ms).
        if blk % 8 == 0 {
            if std::path::Path::new(&watch).join("stop").exists() {
                break;
            }
            for entry in std::fs::read_dir(&watch).into_iter().flatten().flatten() {
                let path = entry.path();
                let name = path.file_name().unwrap().to_string_lossy().to_string();
                if path.extension().and_then(|e| e.to_str()) != Some("fxg")
                    || !seen.insert(name.clone())
                {
                    continue;
                }
                let t_detect = Instant::now();
                let graph = match fxg::load(&path) {
                    Ok(g) => g,
                    Err(e) => {
                        println!(
                            "{{\"event\":\"load-error\",\"file\":\"{name}\",\"err\":\"{e}\"}}"
                        );
                        continue;
                    }
                };
                let cert = certify_graph(&graph, FS);
                if cert.verdict != Verdict::CertifiedStable {
                    println!(
                        "{{\"event\":\"REJECTED\",\"file\":\"{name}\",\"verdict\":\"{:?}\"}}",
                        cert.verdict
                    );
                    continue;
                }
                let Some(frozen) = freeze(&graph, FS) else {
                    println!(
                        "{{\"event\":\"REJECTED\",\"file\":\"{name}\",\"verdict\":\"not-lowerable\"}}"
                    );
                    continue;
                };
                let sos: Vec<fluxion_ops::Biquad> = frozen
                    .sections
                    .iter()
                    .map(|c| fluxion_ops::Biquad {
                        b0: c[0],
                        b1: c[1],
                        b2: c[2],
                        a1: c[3],
                        a2: c[4],
                    })
                    .collect();
                let Some(cmd) = SetCoeffs::new(0, &sos, fade_samples) else {
                    println!(
                        "{{\"event\":\"REJECTED\",\"file\":\"{name}\",\"verdict\":\"too-many-sections\"}}"
                    );
                    continue;
                };
                let _ = tx.push(cmd);
                let us = t_detect.elapsed().as_secs_f64() * 1e6;
                println!(
                    "{{\"event\":\"applied\",\"file\":\"{name}\",\"sections\":{},\"verdict\":\"CertifiedStable\",\
                     \"detect_to_push_us\":{us:.1},\"fade_samples\":{fade_samples},\"block\":{blk}}}",
                    sos.len()
                );
                fade_until = blk + 1 + (fade_samples as u64).div_ceil(buffer as u64);
                let _ = std::fs::write(path.with_extension("ack"), b"applied");
            }
        }

        // Audio side (the callback body): drain commands, process one paced block.
        for (i, x) in input.iter_mut().enumerate() {
            *x = (phase + w * i as f32).sin() * 0.5;
        }
        phase = (phase + w * buffer as f32) % (2.0 * PI);
        while let Some(cmd) = rx.pop() {
            g.apply(&cmd);
        }
        g.process(&input, &mut out);

        // Click bound: max |y[n] - y[n-1]| including across the block boundary.
        let mut m = (out[0] - prev_last).abs();
        for p in out.windows(2) {
            m = m.max((p[1] - p[0]).abs());
        }
        prev_last = out[buffer - 1];
        if blk < fade_until {
            max_step_fade = max_step_fade.max(m);
        } else if blk > 16 {
            max_step_steady = max_step_steady.max(m);
        }

        blk += 1;
        if total > 0 && blk >= total {
            break;
        }
        let now = Instant::now();
        if next_tick > now {
            std::thread::sleep(next_tick - now);
        }
        next_tick += period;
    }

    println!(
        "{{\"summary\":true,\"blocks\":{blk},\"buffer\":{buffer},\
         \"max_step_fade\":{max_step_fade:.6},\"max_step_steady\":{max_step_steady:.6}}}"
    );
}

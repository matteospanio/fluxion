//! Real-time latency CCDF harness for the paper's Fig 4 (experiment E5).
//!
//! Measures the wall time of the render-callback *body* — `RtGraph::process` on a prepared cascade
//! — per block, for 8192 blocks per (depth, buffer) config, while a control thread streams
//! `SetCoeffs` automation over the lock-free ring (the realistic worst case: crossfades active).
//! Emits one JSON line per config with p50/p90/p99/p999/max (µs) and the deadline fraction.
//!
//! Run: `cargo run -p fluxion-rt --release --example latency_ccdf`

use std::time::Instant;

use fluxion_ops::butterworth_lowpass;
use fluxion_rt::{RtGraph, SetCoeffs, channel};

const FS: u32 = 48_000;
const BLOCKS: usize = 8_192;

fn main() {
    for depth in [2usize, 5, 10] {
        for buffer in [128usize, 256, 512] {
            // Cascade of `depth` sections (order 2·depth Butterworth), as in the prior study.
            // A SetCoeffs command carries at most 8 sections, so deeper cascades split into two
            // series filter nodes; automation swaps node 0 (the crossfade path is identical).
            let (sos, alt, mut g) = if depth <= 8 {
                let sos = butterworth_lowpass(2 * depth, 4_000.0, FS);
                let alt = butterworth_lowpass(2 * depth, 2_000.0, FS);
                let g = RtGraph::filter(sos.clone());
                (sos, alt, g)
            } else {
                let half = depth / 2;
                let sos = butterworth_lowpass(2 * half, 4_000.0, FS);
                let alt = butterworth_lowpass(2 * half, 2_000.0, FS);
                let rest = butterworth_lowpass(2 * (depth - half), 6_000.0, FS);
                let g = RtGraph::series(RtGraph::filter(sos.clone()), RtGraph::filter(rest));
                (sos, alt, g)
            };
            g.prepare(buffer);

            let (mut tx, mut rx) = channel::<SetCoeffs>(64);
            let input: Vec<f32> = (0..buffer).map(|i| (0.07 * i as f32).sin()).collect();
            let mut out = vec![0.0f32; buffer];

            let mut lat_us: Vec<f64> = Vec::with_capacity(BLOCKS);
            let mut flip = false;
            for blk in 0..BLOCKS {
                // Control side: a coefficient swap every 64 blocks (crossfade constantly active).
                if blk % 64 == 0 {
                    let target = if flip { &alt } else { &sos };
                    flip = !flip;
                    let _ = tx.push(SetCoeffs::new(0, target, (buffer * 8) as u32).unwrap());
                }
                // Audio side (the callback body): drain commands, process one block.
                let t0 = Instant::now();
                while let Some(cmd) = rx.pop() {
                    g.apply(&cmd);
                }
                g.process(&input, &mut out);
                lat_us.push(t0.elapsed().as_secs_f64() * 1e6);
                std::hint::black_box(&out);
            }

            lat_us.sort_by(f64::total_cmp);
            let q = |p: f64| lat_us[((lat_us.len() as f64 - 1.0) * p) as usize];
            // CCDF curve: ~64 points geometric in the tail probability (1-p), for Fig 4.
            let mut curve = Vec::new();
            let mut tail = 0.5f64;
            while tail * BLOCKS as f64 >= 1.0 {
                curve.push(format!("[{:.6},{:.2}]", tail, q(1.0 - tail)));
                tail *= 0.8;
            }
            println!(
                "{{\"curve\":true,\"depth\":{depth},\"buffer\":{buffer},\"points\":[{}]}}",
                curve.join(",")
            );
            let deadline_us = buffer as f64 / FS as f64 * 1e6;
            println!(
                "{{\"depth\":{depth},\"buffer\":{buffer},\"deadline_us\":{deadline_us:.1},\
                  \"p50_us\":{:.2},\"p90_us\":{:.2},\"p99_us\":{:.2},\"p999_us\":{:.2},\"max_us\":{:.2},\
                  \"p99_budget_pct\":{:.3},\"blocks\":{BLOCKS}}}",
                q(0.50),
                q(0.90),
                q(0.99),
                q(0.999),
                lat_us[lat_us.len() - 1],
                q(0.99) / deadline_us * 100.0
            );
        }
    }
}

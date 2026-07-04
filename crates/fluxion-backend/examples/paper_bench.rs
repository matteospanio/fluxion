//! Paper benchmark: CPU throughput of the **shipped** paths, emitted as JSON (one object per
//! line) for the IS² paper's Fig 2a / T2 (protocol: IS22026/EXPERIMENTS.md E1).
//!
//! Run with: `cargo run -p fluxion-backend --release --example paper_bench`
//! Thread count is controlled by `RAYON_NUM_THREADS` (set 1 for the single-thread rows).

use std::hint::black_box;
use std::time::Instant;

// Same role as torch's caching allocator in the comparator benches: retain and
// reuse large buffers instead of paying mmap/munmap page faults per call.
#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

use fluxion_backend::{process, sos_filter_batch};
use fluxion_core::{Graph, OpKind, Signal};
use fluxion_ops::butterworth_lowpass;

/// Median wall-seconds of `runs` timed calls after `warmup` untimed ones.
fn median_secs(mut f: impl FnMut(), warmup: usize, runs: usize) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let mut times: Vec<f64> = (0..runs)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64()
        })
        .collect();
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

fn signal(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i + seed * 7) % 97) as f32 * 0.13).sin())
        .collect()
}

fn emit(name: &str, params: &str, msamples: f64, secs: f64) {
    let threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "all".into());
    println!(
        "{{\"bench\":\"{name}\",{params},\"threads\":\"{threads}\",\"median_s\":{secs:.6},\"msamples_per_s\":{:.1}}}",
        msamples / secs
    );
}

fn main() {
    let fs = 48_000u32;

    // E1 `batch`: 64 mono signals x 524288 samples, order-6 Butterworth (3 sections).
    {
        let (rows, frames) = (64usize, 524_288usize);
        let sos = butterworth_lowpass(6, 3_000.0, fs);
        let flat = signal(rows * frames, 0);
        let ms = (rows * frames) as f64 / 1e6;
        let s = median_secs(
            || {
                black_box(sos_filter_batch(black_box(&flat), frames, &sos));
            },
            2,
            5,
        );
        emit(
            "batch",
            &format!("\"rows\":{rows},\"frames\":{frames},\"sections\":3"),
            ms,
            s,
        );
    }

    // E1 `multichannel`: 8 ch x 60 s, order-4 lowpass, through the graph executor.
    {
        let frames = 60 * fs as usize;
        let chans: Vec<Vec<f32>> = (0..8).map(|c| signal(frames, c)).collect();
        let sig = Signal::new(fs, chans);
        let g = Graph::op(OpKind::Lowpass, [1_000.0, 4.0]);
        let ms = (8 * frames) as f64 / 1e6;
        let s = median_secs(
            || {
                black_box(process(&g, black_box(&sig)));
            },
            2,
            5,
        );
        emit(
            "multichannel",
            "\"channels\":8,\"seconds\":60,\"order\":4",
            ms,
            s,
        );
    }

    // E1 `cascade`: 1 mono x 60 s, K in {4, 8, 16} sections. The order param caps at 16
    // (8 sections), so deeper cascades chain ops — same executor path either way.
    for k in [4usize, 8, 16] {
        let frames = 60 * fs as usize;
        let sig = Signal::new(fs, vec![signal(frames, 42)]);
        let g = if k <= 8 {
            Graph::op(OpKind::Lowpass, [2_000.0, (2 * k) as f32])
        } else {
            Graph::op(OpKind::Lowpass, [2_000.0, 16.0])
                | Graph::op(OpKind::Lowpass, [2_500.0, 16.0])
        };
        let ms = frames as f64 / 1e6;
        let s = median_secs(
            || {
                black_box(process(&g, black_box(&sig)));
            },
            2,
            5,
        );
        emit(
            "cascade",
            &format!("\"channels\":1,\"seconds\":60,\"sections\":{k}"),
            ms,
            s,
        );
    }
}

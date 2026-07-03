//! Criterion micro-benchmark for the batched SOS cascade (plan tasks C3/C6, A4).
//!
//! [`sos_filter_batch`] interleaves rows into SIMD lanes and runs the cascade across the whole
//! batch (rayon over row-groups); this measures it at a couple of (rows, frames) shapes. Short
//! `sample_size` keeps it a quick tripwire, not a full study. Run: `cargo bench -p fluxion-backend`.

// `criterion_group!`/`criterion_main!` expand to public items with no docs; benches aren't API.
#![allow(missing_docs)]

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fluxion_backend::sos_filter_batch;
use fluxion_ops::butterworth_lowpass;

/// A `rows × frames` planar batch (row-major), each row the same deterministic sinusoid.
fn batch_signal(rows: usize, frames: usize) -> Vec<f32> {
    (0..rows * frames)
        .map(|i| {
            let t = (i % frames) as f32;
            0.5 * (0.05 * t).sin()
        })
        .collect()
}

fn bench_batch(c: &mut Criterion) {
    let sos = butterworth_lowpass(8, 5_000.0, 48_000u32);
    let mut g = c.benchmark_group("sos_filter_batch");
    g.sample_size(20);
    for &(rows, frames) in &[(32usize, 4_096usize), (256usize, 16_384usize)] {
        let rows_buf = batch_signal(rows, frames);
        g.bench_with_input(
            BenchmarkId::from_parameter(format!("{rows}x{frames}")),
            &rows_buf,
            |b, rows_buf| {
                b.iter(|| sos_filter_batch(black_box(rows_buf), black_box(frames), black_box(&sos)))
            },
        );
    }
    g.finish();
}

criterion_group!(benches, bench_batch);
criterion_main!(benches);

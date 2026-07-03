//! Criterion micro-benchmarks for the CPU filter kernels (plan task A4).
//!
//! Two hot inner loops the batch engine and CLI lean on: the mono SOS cascade ([`sos_filter`]) and
//! the FIR direct-vs-FFT crossover ([`fir_filter`] `O(N·M)` vs [`fft_convolve`] `O(N log N)`). Sizes
//! and `sample_size` are deliberately small — this is a quick regression tripwire, not a full
//! performance study. Run: `cargo bench -p fluxion-ops`.

// `criterion_group!`/`criterion_main!` expand to public items with no docs; benches aren't API.
#![allow(missing_docs)]

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fluxion_ops::{butterworth_lowpass, fft_convolve, fir_filter, sos_filter};

/// A deterministic pseudo-signal (two summed sinusoids); avoids a `rand` dev-dependency.
fn signal(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f32;
            0.5 * (0.05 * t).sin() + 0.3 * (0.17 * t).sin()
        })
        .collect()
}

/// Mono 8th-order (4-section) Butterworth cascade over one second at 48 kHz.
fn bench_sos(c: &mut Criterion) {
    let fs = 48_000u32;
    let x = signal(48_000);
    let sos = butterworth_lowpass(8, 5_000.0, fs);
    let mut g = c.benchmark_group("sos_filter");
    g.sample_size(20);
    g.bench_function("mono/8th-order/48k", |b| {
        b.iter(|| sos_filter(black_box(&x), black_box(&sos)))
    });
    g.finish();
}

/// Sweep tap counts across the direct-vs-FFT crossover on the same input, so the two curves can be
/// read against each other (short kernels favor `fir_filter`, long ones `fft_convolve`).
fn bench_fir_crossover(c: &mut Criterion) {
    let x = signal(16_384);
    let mut g = c.benchmark_group("fir_vs_fft");
    g.sample_size(20);
    for taps in [31usize, 127, 511, 2_047] {
        let h: Vec<f32> = (0..taps).map(|k| 1.0 / (1.0 + k as f32)).collect();
        g.bench_with_input(BenchmarkId::new("direct", taps), &h, |b, h| {
            b.iter(|| fir_filter(black_box(&x), black_box(h)))
        });
        g.bench_with_input(BenchmarkId::new("fft", taps), &h, |b, h| {
            b.iter(|| fft_convolve(black_box(&x), black_box(h)))
        });
    }
    g.finish();
}

criterion_group!(benches, bench_sos, bench_fir_crossover);
criterion_main!(benches);

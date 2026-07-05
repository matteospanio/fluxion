//! Quick probe: RtGraph::fir cost at 1024 taps (the soundlamp case-study load).
use fluxion_rt::RtGraph;
use std::time::Instant;

fn main() {
    let taps: Vec<f32> = (0..1024).map(|k| ((k as f32) * 0.01).sin() / 1024.0).collect();
    let mut g = RtGraph::fir(taps);
    let buffer = 512usize;
    g.prepare(buffer);
    let input: Vec<f32> = (0..buffer).map(|i| (0.07 * i as f32).sin()).collect();
    let mut out = vec![0.0f32; buffer];
    for _ in 0..50 { g.process(&input, &mut out); } // warmup
    let n = 2000;
    let t0 = Instant::now();
    for _ in 0..n { g.process(&input, &mut out); }
    let el = t0.elapsed().as_secs_f64() / n as f64;
    let deadline = buffer as f64 / 44_100.0;
    println!("1024-tap FIR, {buffer}-sample block: {:.3} ms/block  ({:.1}% of 44.1kHz deadline; 8ch => {:.1}%)",
        el * 1e3, el / deadline * 100.0, 8.0 * el / deadline * 100.0);
    std::hint::black_box(&out);
}

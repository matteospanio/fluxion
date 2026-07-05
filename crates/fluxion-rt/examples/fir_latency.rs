//! Per-block latency of a single 1024-tap FIR `RtGraph`, paced at real time.
//!
//! Companion to `latency_ccdf` for the anira comparison in the IS2 paper:
//! same callback-body timing, pure FIR (no SetCoeffs automation — a FIR
//! node has no live coefficient swap). Emits one JSON line per buffer size.
//!
//! Usage: fir_latency [seconds-per-config] [taps.f32] [paced|unpaced]
//!   taps.f32 = headerless little-endian f32 taps, natural order h[0..N-1];
//!   omitted or "-" -> synthetic 1024 taps.
//!   unpaced (default) = back-to-back blocks, the `latency_ccdf` (Fig. 4)
//!   protocol and the one anira's ProcessBlockFixture uses; paced = one
//!   block per buffer period, the `soundlamp_demo` protocol.

use std::time::{Duration, Instant};

use fluxion_rt::RtGraph;

const FS: u32 = 48_000;
const TAPS: usize = 1024;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seconds: f64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(60.0);
    let taps: Vec<f32> = match args.get(2).map(String::as_str) {
        Some(path) if path != "-" => std::fs::read(path)
            .expect("read taps file")
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        _ => (0..TAPS).map(|k| ((k as f32) * 0.013).sin() / 64.0).collect(),
    };
    assert_eq!(taps.len(), TAPS, "expected {TAPS} taps");
    let paced = args.get(3).map(String::as_str) == Some("paced");

    for buffer in [128usize, 256, 512] {
        let mut g = RtGraph::fir(taps.clone());
        g.prepare(buffer);

        let input: Vec<f32> = (0..buffer).map(|i| (0.07 * i as f32).sin()).collect();
        let mut out = vec![0.0f32; buffer];

        let deadline = buffer as f64 / FS as f64;
        let blocks = (seconds / deadline) as usize;
        let mut lat_us: Vec<f64> = Vec::with_capacity(blocks);

        let mut next_tick = Instant::now() + Duration::from_secs_f64(deadline);
        for _ in 0..blocks {
            let t0 = Instant::now();
            g.process(&input, &mut out);
            lat_us.push(t0.elapsed().as_secs_f64() * 1e6);
            std::hint::black_box(&out);
            if paced {
                let now = Instant::now();
                if next_tick > now {
                    std::thread::sleep(next_tick - now);
                }
                next_tick += Duration::from_secs_f64(deadline);
            }
        }

        lat_us.sort_by(f64::total_cmp);
        let q = |p: f64| lat_us[((lat_us.len() as f64 - 1.0) * p) as usize];
        let d_us = deadline * 1e6;
        let missed = lat_us.iter().filter(|&&t| t > d_us).count();
        println!(
            "{{\"case\":\"fir_latency\",\"mode\":\"{}\",\"taps\":{TAPS},\"buffer\":{buffer},\"fs\":{FS},\
             \"blocks\":{},\"deadline_us\":{d_us:.1},\"p50_us\":{:.2},\"p99_us\":{:.2},\
             \"max_us\":{:.2},\"p99_budget_pct\":{:.3},\"missed_deadlines\":{missed}}}",
            if paced { "paced" } else { "unpaced" },
            lat_us.len(),
            q(0.50),
            q(0.99),
            q(1.0),
            q(0.99) / d_us * 100.0,
        );
    }
}

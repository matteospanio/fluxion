//! Multichannel per-channel-chain case study harness (IS² paper §IV-C, the
//! SoundFood soundlamp): a mono source is upmixed through N output channels,
//! each with its **own** filter chain (EQ stages + a long directional FIR),
//! exactly the shape of a 7.1 "sound bubble" deployment. Every chain is
//! certified, then lowered to its own [`fluxion::RtGraph`].
//!
//! The chain itself is read from a spec file (the production tuning is
//! proprietary and lives outside the repo):
//! one block per output channel, blocks separated by `---`, ops one per line:
//! `highpass|lowpass <cutoff> <order>`, `highshelf|peaking <freq> <gain_db> <q>`,
//! `fir <taps.f32>` (little-endian f32 taps, relative to the spec dir),
//! `gain_db <dB>`.
//!
//! Modes:
//!   soundlamp_demo <chain.spec> <in.wav> paced <buffer> <seconds>
//!     Paced real-time loop (one block every buffer/fs, the Fig. 4 protocol):
//!     per-block processing latency percentiles, deadline fraction, peak RSS.
//!   soundlamp_demo <chain.spec> <in.wav> offline <out.wav>
//!     File-to-file at full speed: wall time, xRT factor, 8-channel output WAV
//!     (for the correctness cross-check against the production service).

use std::time::{Duration, Instant};

use fluxion_backend::{certify_graph, to_rt_graph};
use fluxion_core::{Graph, OpKind};
use fluxion_io::{WavBlockWriter, WavEncoding, read_wav_blocks};
use fluxion_rt::RtGraph;

const FS: u32 = 44_100;

fn db_to_lin(db: f32) -> f32 {
    10.0f32.powf(db / 20.0)
}

/// Parse the spec into one `Graph` per output channel.
fn load_chains(spec_path: &str) -> Vec<Graph> {
    let spec_dir = std::path::Path::new(spec_path)
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .to_path_buf();
    let text = std::fs::read_to_string(spec_path).expect("read chain spec");
    let mut chains = Vec::new();
    let mut current: Option<Graph> = None;
    let mut push = |g: &mut Option<Graph>, op: Graph| {
        *g = Some(match g.take() {
            Some(acc) => acc | op,
            None => op,
        });
    };
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line == "---" {
            chains.push(current.take().expect("empty channel block"));
            continue;
        }
        let mut it = line.split_whitespace();
        let word = it.next().unwrap();
        let args: Vec<f32> = it.clone().filter_map(|t| t.parse().ok()).collect();
        let op = match word {
            "highpass" => Graph::op(OpKind::Highpass, [args[0], args[1]]),
            "lowpass" => Graph::op(OpKind::Lowpass, [args[0], args[1]]),
            "highshelf" => Graph::op(OpKind::HighShelf, [args[0], args[1], args[2]]),
            "peaking" => Graph::op(OpKind::Peaking, [args[0], args[1], args[2]]),
            "gain_db" => Graph::op(OpKind::Gain, [db_to_lin(args[0])]),
            "fir" => {
                let taps_file = spec_dir.join(it.next().expect("fir taps file"));
                let bytes = std::fs::read(&taps_file).expect("read taps");
                let taps: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                Graph::op(OpKind::Fir, taps)
            }
            other => panic!("unknown op '{other}'"),
        };
        push(&mut current, op);
    }
    chains.push(current.take().expect("empty final channel block"));
    chains
}

/// Certify every chain and lower each to its own RtGraph.
fn lower_all(chains: &[Graph], buffer: usize) -> Vec<RtGraph> {
    chains
        .iter()
        .enumerate()
        .map(|(c, g)| {
            let cert = certify_graph(g, FS);
            eprintln!("channel {c}: verdict {:?}", cert.verdict);
            let mut rt = to_rt_graph(g, FS).expect("chain must be realtime-lowerable");
            rt.prepare(buffer);
            rt
        })
        .collect()
}

fn peak_rss_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(0)
}

/// Load the mono source (first channel) fully — the paced loop cycles over it.
fn load_mono(path: &str) -> Vec<f32> {
    let mut mono = Vec::new();
    for block in read_wav_blocks(path, 65_536).expect("open input wav") {
        let sig = block.expect("read input wav");
        mono.extend_from_slice(&sig.channels[0]);
    }
    assert!(!mono.is_empty(), "empty input");
    mono
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (spec, input, mode) = (&a[1], &a[2], a[3].as_str());
    let chains = load_chains(spec);
    let n_ch = chains.len();

    match mode {
        "paced" => {
            let buffer: usize = a[4].parse().unwrap();
            let seconds: f64 = a[5].parse().unwrap();
            let mut graphs = lower_all(&chains, buffer);
            let mono = load_mono(input);
            let mut outs = vec![vec![0.0f32; buffer]; n_ch];
            let deadline = buffer as f64 / FS as f64;
            let blocks = (seconds / deadline) as usize;
            let mut lat_us: Vec<f64> = Vec::with_capacity(blocks);

            let mut pos = 0usize;
            let mut next_tick = Instant::now() + Duration::from_secs_f64(deadline);
            for _ in 0..blocks {
                // Wrap the source (a restaurant stream loops tracks).
                let end = pos + buffer;
                let block: Vec<f32> = if end <= mono.len() {
                    mono[pos..end].to_vec()
                } else {
                    let mut b = mono[pos..].to_vec();
                    b.extend_from_slice(&mono[..end - mono.len()]);
                    b
                };
                pos = end % mono.len();

                let t0 = Instant::now();
                for (g, o) in graphs.iter_mut().zip(outs.iter_mut()) {
                    g.process(&block, o);
                }
                lat_us.push(t0.elapsed().as_secs_f64() * 1e6);
                std::hint::black_box(&outs);

                let now = Instant::now();
                if next_tick > now {
                    std::thread::sleep(next_tick - now);
                }
                next_tick += Duration::from_secs_f64(deadline);
            }

            lat_us.sort_by(f64::total_cmp);
            let q = |p: f64| lat_us[((lat_us.len() as f64 - 1.0) * p) as usize];
            let over = lat_us.iter().filter(|&&t| t > deadline * 1e6).count();
            println!(
                "{{\"case\":\"soundlamp\",\"mode\":\"paced\",\"channels\":{n_ch},\"buffer\":{buffer},\
                 \"fs\":{FS},\"blocks\":{},\"deadline_us\":{:.1},\"p50_us\":{:.1},\"p99_us\":{:.1},\
                 \"max_us\":{:.1},\"p99_budget_pct\":{:.2},\"cpu_share_pct\":{:.2},\
                 \"missed_deadlines\":{over},\"peak_rss_kb\":{}}}",
                lat_us.len(),
                deadline * 1e6,
                q(0.50),
                q(0.99),
                q(1.0),
                q(0.99) / (deadline * 1e6) * 100.0,
                q(0.50) / (deadline * 1e6) * 100.0,
                peak_rss_kb(),
            );
        }
        "offline" => {
            let out_path = &a[4];
            let block = 65_536usize;
            let mut graphs = lower_all(&chains, block);
            let mono = load_mono(input);
            let mut writer =
                WavBlockWriter::create(out_path, FS, n_ch as u16, WavEncoding::default())
                    .expect("create output wav");
            let mut outs = vec![vec![0.0f32; block]; n_ch];

            let t0 = Instant::now();
            for chunk in mono.chunks(block) {
                for (g, o) in graphs.iter_mut().zip(outs.iter_mut()) {
                    o.resize(chunk.len(), 0.0);
                    g.process(chunk, o);
                }
                writer.write_block(&outs).expect("write block");
            }
            writer.finalize().expect("finalize wav");
            let wall = t0.elapsed().as_secs_f64();
            let audio_s = mono.len() as f64 / FS as f64;
            println!(
                "{{\"case\":\"soundlamp\",\"mode\":\"offline\",\"channels\":{n_ch},\"fs\":{FS},\
                 \"audio_s\":{audio_s:.2},\"wall_s\":{wall:.3},\"xrt\":{:.1},\"peak_rss_kb\":{}}}",
                audio_s / wall,
                peak_rss_kb(),
            );
        }
        other => panic!("unknown mode '{other}' (paced|offline)"),
    }
}

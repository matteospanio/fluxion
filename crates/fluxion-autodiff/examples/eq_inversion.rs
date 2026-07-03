//! Blind coloration inversion — the paper's train→certify→freeze→play experiment
//! (IS22026/EXPERIMENTS.md E4).
//!
//! An unknown coloration filter (low-shelf | peaking | high-shelf, randomized) colours broadband
//! noise; a 3-band parametric EQ is trained from audio alone to invert it. Three modes:
//!   `design`  — train the EQ's **design parameters** (freq/gain/Q) via the design-Jacobian chain
//!               (always on the stable manifold; ours),
//!   `raw`     — train raw biquad coefficients with the in-loop Jury projection,
//!   `rawfree` — raw coefficients with NO projection (honest ablation).
//!
//! Emits JSON lines: per-iterate {loss, stability margin, params}, then a summary with the
//! magnitude-response curves (coloration, ideal inverse = -coloration, learned EQ) and the frozen
//! artifact path. Run: `cargo run -p fluxion-autodiff --release --features burn --example
//! eq_inversion -- [design|raw|rawfree] [out.fxg]`

use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;
use fluxion_autodiff::burn_backend::{sos_design, sos_trainable};
use fluxion_core::{FrozenSos, Graph, OpKind};
use fluxion_ops::{
    Biquad, certify_sos, high_shelf, low_shelf, peaking, project_stable_flat, sos_filter,
    sos_magnitude,
};

type B = Autodiff<NdArray>;

const FS: u32 = 48_000;

/// The 3-band EQ design: params = [f1, g1, q1, f2, g2, q2, f3, g3, q3] → flat coeffs (3 sections).
/// Band 1 = low shelf, band 2 = peaking, band 3 = high shelf.
fn eq_design(p: &[f32], fs: u32) -> Vec<f32> {
    let sections = [
        low_shelf(
            p[0].clamp(20.0, 2_000.0),
            p[1].clamp(-24.0, 24.0),
            p[2].clamp(0.1, 5.0),
            fs,
        ),
        peaking(
            p[3].clamp(100.0, 10_000.0),
            p[4].clamp(-24.0, 24.0),
            p[5].clamp(0.1, 5.0),
            fs,
        ),
        high_shelf(
            p[6].clamp(1_000.0, 20_000.0),
            p[7].clamp(-24.0, 24.0),
            p[8].clamp(0.1, 5.0),
            fs,
        ),
    ];
    sections
        .iter()
        .flat_map(|b| [b.b0, b.b1, b.b2, b.a1, b.a2])
        .collect()
}

fn to_sos(flat: &[f32]) -> Vec<Biquad> {
    flat.chunks_exact(5)
        .map(|c| Biquad {
            b0: c[0],
            b1: c[1],
            b2: c[2],
            a1: c[3],
            a2: c[4],
        })
        .collect()
}

/// Deterministic broadband pseudo-noise in [-1, 1) (LCG) — no RNG dependency.
fn pseudo_noise(n: usize, mut s: u32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 9) as f32 / (1u32 << 22) as f32 - 1.0
        })
        .collect()
}

/// Magnitude response (dB) of a cascade over a log-spaced frequency grid.
fn response_db(sos: &[Biquad], freqs: &[f32]) -> Vec<f32> {
    freqs
        .iter()
        .map(|&f| {
            let w = 2.0 * std::f32::consts::PI * f / FS as f32;
            20.0 * sos_magnitude(sos, w).max(1e-9).log10()
        })
        .collect()
}

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "design".into());
    let fxg_out = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "eq_inverse.fxg".into());
    let device = Default::default();

    // --- The unknown coloration (hidden from the learner; "randomized" via the LCG seed). --------
    let jit = |k: u32| (pseudo_noise(1, 0xC01_0000 + k)[0]) * 0.2; // ±20%
    let color = [
        low_shelf(200.0 * (1.0 + jit(1)), -6.0 * (1.0 + jit(2)), 0.9, FS),
        peaking(1_200.0 * (1.0 + jit(3)), 8.0 * (1.0 + jit(4)), 2.0, FS),
        high_shelf(8_000.0 * (1.0 + jit(5)), 5.0 * (1.0 + jit(6)), 0.9, FS),
    ];

    // Training signal: colored noise in, original out (waveform MSE — inversion in the time domain).
    let n = 8_192usize;
    let clean = pseudo_noise(n, 0x1234_5678);
    let colored = sos_filter(&clean, &color);
    let x = Tensor::<B, 1>::from_floats(colored.as_slice(), &device);
    let target = Tensor::<B, 1>::from_floats(clean.as_slice(), &device);

    // --- Train. -----------------------------------------------------------------------------------
    // Rprop per parameter: sign-based steps, no learning-rate tuning across heterogeneous units.
    let mut params: Vec<f32> = vec![300.0, 0.0, 0.9, 1_000.0, 0.0, 1.0, 6_000.0, 0.0, 0.9];
    let mut coeffs = eq_design(&params, FS); // raw modes train this directly
    let (mut steps, mut prev): (Vec<f32>, Vec<f32>) = match mode.as_str() {
        "design" => (
            vec![50.0, 1.0, 0.05, 100.0, 1.0, 0.05, 200.0, 1.0, 0.05],
            vec![0.0; 9],
        ),
        _ => (vec![1e-3; coeffs.len()], vec![0.0; coeffs.len()]),
    };

    let iters: usize = std::env::var("ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(400);
    for it in 0..iters {
        let (loss_v, grads) = match mode.as_str() {
            "design" => {
                let p = Tensor::<B, 1>::from_floats(params.as_slice(), &device).require_grad();
                let loss = (sos_design(x.clone(), p.clone(), eq_design, FS) - target.clone())
                    .powf_scalar(2.0)
                    .sum();
                let v = loss.clone().into_data().to_vec::<f32>().unwrap()[0];
                let g = p
                    .grad(&loss.backward())
                    .unwrap()
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap();
                (v, g)
            }
            _ => {
                let c = Tensor::<B, 1>::from_floats(coeffs.as_slice(), &device).require_grad();
                let loss = (sos_trainable(x.clone(), c.clone()) - target.clone())
                    .powf_scalar(2.0)
                    .sum();
                let v = loss.clone().into_data().to_vec::<f32>().unwrap()[0];
                let g = c
                    .grad(&loss.backward())
                    .unwrap()
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap();
                (v, g)
            }
        };

        // Rprop update on whichever vector this mode trains.
        let vars: &mut Vec<f32> = if mode == "design" {
            &mut params
        } else {
            &mut coeffs
        };
        for i in 0..vars.len() {
            let g = grads[i];
            if g != 0.0 {
                steps[i] = if prev[i] * g < 0.0 {
                    steps[i] * 0.5
                } else {
                    steps[i] * 1.2
                };
                vars[i] -= g.signum() * steps[i];
                prev[i] = g;
            }
        }
        if mode == "design" {
            coeffs = eq_design(&params, FS);
        } else if mode == "raw" {
            project_stable_flat(&mut coeffs, 1e-3); // the in-loop stability projection
        }

        let cert = certify_sos(&to_sos(&coeffs));
        println!(
            "{{\"iter\":{it},\"loss\":{loss_v:.6e},\"margin\":{:.6},\"shippable\":{}}}",
            cert.margin,
            cert.verdict.is_shippable()
        );
        if !cert.margin.is_finite() {
            println!("{{\"event\":\"diverged\",\"iter\":{it}}}"); // rawfree's honest outcome
            break;
        }
    }

    // --- Certify + freeze + response curves. ------------------------------------------------------
    let learned = to_sos(&coeffs);
    let cert = certify_sos(&learned);
    let sections: Vec<[f32; 5]> = learned
        .iter()
        .map(|b| [b.b0, b.b1, b.b2, b.a1, b.a2])
        .collect();
    if cert.verdict.is_shippable() {
        FrozenSos::new(FS, sections)
            .save(&fxg_out)
            .expect("write fxg");
    }
    // Also express the learned EQ as a *graph* artifact when training design params.
    if mode == "design" {
        let g = Graph::op(OpKind::LowShelf, [params[0], params[1], params[2]])
            | Graph::op(OpKind::Peaking, [params[3], params[4], params[5]])
            | Graph::op(OpKind::HighShelf, [params[6], params[7], params[8]]);
        let _ = fluxion_core::fxg::save(&g, format!("{fxg_out}.graph.fxg"));
    }

    let freqs: Vec<f32> = (0..120)
        .map(|i| 20.0 * (10f32).powf(i as f32 * 0.025))
        .collect();
    let fmt = |v: &[f32]| {
        v.iter()
            .map(|x| format!("{x:.3}"))
            .collect::<Vec<_>>()
            .join(",")
    };
    println!(
        "{{\"summary\":true,\"mode\":\"{mode}\",\"verdict\":\"{}\",\"margin\":{:.6},\"fxg\":\"{fxg_out}\",\
          \"freqs\":[{}],\"coloration_db\":[{}],\"learned_db\":[{}]}}",
        cert.verdict,
        cert.margin,
        fmt(&freqs),
        fmt(&response_db(&color, &freqs)),
        fmt(&response_db(&learned, &freqs)),
    );
}

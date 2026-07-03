//! Training-step throughput: analytic-adjoint SOS training, for the paper's T4
//! (IS22026/EXPERIMENTS.md E3). Times one step (forward + backward, MSE) of a trainable cascade
//! over a grid of (sections K, samples T, batch B); JSON per cell. The torch-side counterparts
//! (torchfx.ddsp analytic; torchaudio lfilter unrolled) run from a Python script on the same box.
//!
//! Run: `cargo run -p fluxion-autodiff --release --features burn --example train_step_bench`

use std::time::Instant;

use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;
use fluxion_autodiff::burn_backend::sos_trainable;
use fluxion_ops::butterworth_lowpass;

type B = Autodiff<NdArray>;

fn main() {
    let device = Default::default();
    let fs = 48_000u32;

    for k in [1usize, 4, 8] {
        for t in [1usize << 14, 1 << 16] {
            let b = 16usize;
            let sos = butterworth_lowpass(2 * k, 3_000.0, fs);
            let flat: Vec<f32> = sos
                .iter()
                .flat_map(|q| [q.b0, q.b1, q.b2, q.a1, q.a2])
                .collect();
            let rows: Vec<Vec<f32>> = (0..b)
                .map(|r| {
                    (0..t)
                        .map(|i| (((i + r * 13) % 97) as f32 * 0.13).sin())
                        .collect()
                })
                .collect();
            let targets: Vec<Tensor<B, 1>> = rows
                .iter()
                .map(|row| Tensor::<B, 1>::from_floats(row.as_slice(), &device))
                .collect();

            // One "step" = fwd+bwd over the whole batch (per-row, summed grads — protocol noted).
            let step = || {
                let c = Tensor::<B, 1>::from_floats(flat.as_slice(), &device).require_grad();
                let mut grad_acc = vec![0.0f32; flat.len()];
                for (row, tgt) in rows.iter().zip(&targets) {
                    let xr = Tensor::<B, 1>::from_floats(row.as_slice(), &device);
                    let loss = (sos_trainable(xr, c.clone()) - tgt.clone())
                        .powf_scalar(2.0)
                        .sum();
                    let g = c
                        .grad(&loss.backward())
                        .unwrap()
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap();
                    for (a, v) in grad_acc.iter_mut().zip(g) {
                        *a += v;
                    }
                }
                grad_acc[0]
            };

            // Median of 5 after 2 warmups.
            for _ in 0..2 {
                std::hint::black_box(step());
            }
            let mut times: Vec<f64> = (0..5)
                .map(|_| {
                    let t0 = Instant::now();
                    std::hint::black_box(step());
                    t0.elapsed().as_secs_f64()
                })
                .collect();
            times.sort_by(f64::total_cmp);
            println!(
                "{{\"system\":\"fluxion-analytic-cpu\",\"K\":{k},\"T\":{t},\"B\":{b},\
                  \"step_median_s\":{:.6},\"protocol\":\"per-row loop, summed grads\"}}",
                times[times.len() / 2]
            );
        }
    }
}

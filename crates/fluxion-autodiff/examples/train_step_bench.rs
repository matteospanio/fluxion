//! Training-step throughput: analytic-adjoint SOS training, for the paper's T4
//! (IS22022/EXPERIMENTS.md E3). Times one step (forward + backward, MSE) of a trainable cascade
//! over a grid of (sections K, samples T, batch B); JSON per cell. The torch-side counterparts
//! (torchfx.ddsp analytic; torchaudio lfilter unrolled) run from a Python script on the same box.
//!
//! Run (CPU): `cargo run -p fluxion-autodiff --release --features burn --example train_step_bench`
//! Run (GPU): `cargo run -p fluxion-autodiff --release --features cuda --example train_step_bench -- cuda`
//!   GPU cells cover what the resident kernels support today: the trainable single section
//!   (`biquad_train_gpu`, K=1) and the fixed-cascade adjoint step (`sos_gpu`, gradients through a
//!   frozen cascade to the input — the hybrid-model configuration) at any K.

use std::time::Instant;

use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;
use fluxion_autodiff::burn_backend::sos_trainable;
use fluxion_ops::butterworth_lowpass;

type B = Autodiff<NdArray>;

fn median5(step: &mut dyn FnMut() -> f32) -> f64 {
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
    times[times.len() / 2]
}

#[cfg(feature = "cuda")]
fn run_cuda() {
    use burn::backend::Autodiff;
    use fluxion_autodiff::cuda::{Gpu, biquad_train_gpu, sos_gpu};

    type G = Autodiff<Gpu>;
    let device = Default::default();
    let fs = 48_000u32;
    let b = 16usize;

    for t in [1usize << 14, 1 << 16] {
        // Trainable single section (K=1): input + coefficient gradients on resident kernels,
        // same per-row-loop, summed-grads protocol as the CPU row.
        let sos = butterworth_lowpass(2, 3_000.0, fs);
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
        let targets: Vec<Tensor<G, 1>> = rows
            .iter()
            .map(|row| Tensor::<G, 1>::from_floats(row.as_slice(), &device))
            .collect();
        let mut step = || {
            let c = Tensor::<G, 1>::from_floats(flat.as_slice(), &device).require_grad();
            let mut acc = 0.0f32;
            for (row, tgt) in rows.iter().zip(&targets) {
                let xr = Tensor::<G, 1>::from_floats(row.as_slice(), &device);
                let loss = (biquad_train_gpu(xr, c.clone()) - tgt.clone())
                    .powf_scalar(2.0)
                    .sum();
                let g = c
                    .grad(&loss.backward())
                    .unwrap()
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap();
                acc += g[0];
            }
            acc
        };
        println!(
            "{{\"system\":\"fluxion-analytic-cuda\",\"K\":1,\"T\":{t},\"B\":{b},\
              \"step_median_s\":{:.6},\"protocol\":\"per-row loop, summed grads, resident kernels\"}}",
            median5(&mut step)
        );

        // Fixed-cascade adjoint step at K in {1,4,8}: gradients THROUGH a frozen cascade to the
        // input (the hybrid neural-DSP training configuration), resident forward + adjoint.
        for k in [1usize, 4, 8] {
            let sos = butterworth_lowpass(2 * k, 3_000.0, fs);
            let mut step = || {
                let mut acc = 0.0f32;
                for row in &rows {
                    let xr =
                        Tensor::<G, 1>::from_floats(row.as_slice(), &device).require_grad();
                    let loss = sos_gpu(xr.clone(), &sos).powf_scalar(2.0).sum();
                    let g = xr
                        .grad(&loss.backward())
                        .unwrap()
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap();
                    acc += g[0];
                }
                acc
            };
            println!(
                "{{\"system\":\"fluxion-fixed-adjoint-cuda\",\"K\":{k},\"T\":{t},\"B\":{b},\
                  \"step_median_s\":{:.6},\"protocol\":\"per-row loop, input-grad through frozen cascade\"}}",
                median5(&mut step)
            );
        }
    }
}

fn main() {
    #[cfg(feature = "cuda")]
    if std::env::args().nth(1).as_deref() == Some("cuda") {
        run_cuda();
        return;
    }
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
            let mut step = || {
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

            println!(
                "{{\"system\":\"fluxion-analytic-cpu\",\"K\":{k},\"T\":{t},\"B\":{b},\
                  \"step_median_s\":{:.6},\"protocol\":\"per-row loop, summed grads\"}}",
                median5(&mut step)
            );
        }
    }
}

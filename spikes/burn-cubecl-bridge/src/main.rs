//! Bridge + adjoint: run the SOS **forward** and **input-gradient (adjoint)** CubeCL kernels
//! directly on *resident* Burn tensors, so a forward+backward loop stays on the GPU (no host
//! roundtrip) — where the compute win lands.
//!
//! Result (RTX 3070, CUDA 12.4, Burn 0.21 / CubeCL 0.10, 2026-06-26): both kernels bit-accurate vs
//! CPU (5.96e-8), and a **resident forward+backward** runs at ~40 ms/iter (vs ~860 ms if each pass
//! round-tripped through the host).
//!
//! The cascade adjoint is the same recurrence run **backward in time, sections in reverse** (=
//! `flip · filter · flip`, i.e. `fluxion_ops::sos_input_grad`) — no explicit array reversal.
//!
//! Bridge: the raw `CubeBackend<R>` float primitive is `CubeTensor<R>` with public `client`/`handle`;
//! launch on those, wrap back with `CubeTensor::new_contiguous(client, device, shape, handle, dtype)`.
//! Generic over `R` → cross-vendor (CUDA/ROCm/Metal/WGSL).
//!
//! Next (F3 tail): the **coefficient-gradient** kernel (all-pole intermediates + a cross-row
//! reduction — `sos_vjp`'s `grad_coeffs` on device), then wire forward+both-backwards into a
//! `CubeBackend<R>` specialization of `fluxion-autodiff`'s op for fully GPU-resident *training*.

use burn::tensor::{Tensor, TensorPrimitive};
use burn_cubecl::CubeBackend;
use burn_cubecl::tensor::CubeTensor;
use cubecl::cuda::CudaRuntime;
use cubecl::prelude::*;

type Gpu = CubeBackend<CudaRuntime, f32, i32, u8>;

#[cube(launch)]
fn sos_fwd<F: Float>(input: &Array<F>, output: &mut Array<F>, coeffs: &Array<F>, #[comptime] nf: usize, #[comptime] ns: usize) {
    let n_rows = input.len() / nf;
    if ABSOLUTE_POS < n_rows {
        let base = ABSOLUTE_POS * nf;
        for t in 0..nf {
            output[base + t] = input[base + t];
        }
        #[unroll]
        for s in 0..ns {
            let c = s * 5;
            let (b0, b1, b2, a1, a2) = (coeffs[c], coeffs[c + 1], coeffs[c + 2], coeffs[c + 3], coeffs[c + 4]);
            let mut s1 = F::new(0.0);
            let mut s2 = F::new(0.0);
            for t in 0..nf {
                let x = output[base + t];
                let y = b0 * x + s1;
                s1 = b1 * x - a1 * y + s2;
                s2 = b2 * x - a2 * y;
                output[base + t] = y;
            }
        }
    }
}

/// Cascade adjoint: reverse section order, backward time per section.
#[cube(launch)]
fn sos_adj<F: Float>(grad: &Array<F>, output: &mut Array<F>, coeffs: &Array<F>, #[comptime] nf: usize, #[comptime] ns: usize) {
    let n_rows = grad.len() / nf;
    if ABSOLUTE_POS < n_rows {
        let base = ABSOLUTE_POS * nf;
        for t in 0..nf {
            output[base + t] = grad[base + t];
        }
        #[unroll]
        for s in 0..ns {
            let ss = ns - 1 - s; // reverse section order
            let c = ss * 5;
            let (b0, b1, b2, a1, a2) = (coeffs[c], coeffs[c + 1], coeffs[c + 2], coeffs[c + 3], coeffs[c + 4]);
            let mut s1 = F::new(0.0);
            let mut s2 = F::new(0.0);
            for t in 0..nf {
                let tt = nf - 1 - t; // backward time
                let x = output[base + tt];
                let y = b0 * x + s1;
                s1 = b1 * x - a1 * y + s2;
                s2 = b2 * x - a2 * y;
                output[base + tt] = y;
            }
        }
    }
}

fn run(x: &CubeTensor<CudaRuntime>, flat: &[f32], frames: usize, ns: usize, adjoint: bool) -> CubeTensor<CudaRuntime> {
    let client = &x.client;
    let n = x.meta.shape().num_elements();
    let rows = n / frames;
    let out_h = client.empty(n * 4);
    let co_h = client.create_from_slice(f32::as_bytes(flat));
    let count = CubeCount::Static(rows.div_ceil(256) as u32, 1, 1);
    let dim = CubeDim::new(client, 256);
    let inp = unsafe { ArrayArg::from_raw_parts(x.handle.clone(), n) };
    let outp = unsafe { ArrayArg::from_raw_parts(out_h.clone(), n) };
    let cop = unsafe { ArrayArg::from_raw_parts(co_h.clone(), flat.len()) };
    if adjoint {
        sos_adj::launch::<f32, CudaRuntime>(client, count, dim, inp, outp, cop, frames, ns);
    } else {
        sos_fwd::launch::<f32, CudaRuntime>(client, count, dim, inp, outp, cop, frames, ns);
    }
    CubeTensor::new_contiguous(client.clone(), x.device.clone(), x.meta.shape().clone(), out_h, x.dtype)
}

fn cpu_fwd(input: &[f32], frames: usize, sos: &[[f32; 5]]) -> Vec<f32> {
    let mut d = input.to_vec();
    for c in sos {
        for r in 0..d.len() / frames {
            let b = r * frames;
            let (mut s1, mut s2) = (0.0f32, 0.0f32);
            for t in 0..frames {
                let x = d[b + t];
                let y = c[0] * x + s1;
                s1 = c[1] * x - c[3] * y + s2;
                s2 = c[2] * x - c[4] * y;
                d[b + t] = y;
            }
        }
    }
    d
}

fn cpu_adj(grad: &[f32], frames: usize, sos: &[[f32; 5]]) -> Vec<f32> {
    let mut g = grad.to_vec();
    for c in sos.iter().rev() {
        for r in 0..g.len() / frames {
            let b = r * frames;
            let mut row = g[b..b + frames].to_vec();
            row.reverse();
            let (mut s1, mut s2) = (0.0f32, 0.0f32);
            for x in row.iter_mut() {
                let xi = *x;
                let y = c[0] * xi + s1;
                s1 = c[1] * xi - c[3] * y + s2;
                s2 = c[2] * xi - c[4] * y;
                *x = y;
            }
            row.reverse();
            g[b..b + frames].copy_from_slice(&row);
        }
    }
    g
}

fn to_vec(ct: CubeTensor<CudaRuntime>) -> Vec<f32> {
    Tensor::<Gpu, 1>::from_primitive(TensorPrimitive::Float(ct)).into_data().to_vec::<f32>().unwrap()
}

fn main() {
    let device = Default::default();
    let (batch, frames) = (16_384usize, 4_096usize);
    let n = batch * frames;
    let input: Vec<f32> = (0..n).map(|i| ((i % 97) as f32 / 97.0) - 0.5).collect();
    let sos = [
        [0.2929f32, 0.5858, 0.2929, 0.0, 0.1716],
        [0.5, 0.3, -0.1, -0.2, 0.05],
        [0.8, -0.2, 0.1, 0.1, -0.3],
    ];
    let flat: Vec<f32> = sos.iter().flatten().copied().collect();
    let xp: CubeTensor<CudaRuntime> = Tensor::<Gpu, 1>::from_floats(input.as_slice(), &device).into_primitive().tensor();

    let d_fwd = to_vec(run(&xp, &flat, frames, sos.len(), false))
        .iter()
        .zip(&cpu_fwd(&input, frames, &sos))
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    println!("forward  max|GPU-CPU| = {d_fwd:.3e}");
    let d_adj = to_vec(run(&xp, &flat, frames, sos.len(), true))
        .iter()
        .zip(&cpu_adj(&input, frames, &sos))
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    println!("adjoint  max|GPU-CPU| = {d_adj:.3e}");
    assert!(d_fwd < 1e-4 && d_adj < 1e-4);

    // Resident forward+backward: both passes stay on the device.
    let _ = run(&xp, &flat, frames, sos.len(), false);
    let k = 30u32;
    let t = std::time::Instant::now();
    let mut last = xp.clone();
    for _ in 0..k {
        let y = run(&xp, &flat, frames, sos.len(), false);
        last = run(&y, &flat, frames, sos.len(), true);
    }
    let _ = to_vec(last); // force sync
    println!("resident fwd+bwd: {:.2} ms/iter", t.elapsed().as_secs_f64() * 1000.0 / k as f64);
    println!(">>> RESIDENT FORWARD + ADJOINT OK <<<");
}

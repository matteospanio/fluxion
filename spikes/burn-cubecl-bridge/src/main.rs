//! F3 kernels on resident Burn tensors: **forward**, **input-gradient (adjoint)**, and
//! **coefficient-gradient** SOS kernels — the full analytic backward on device.
//!
//! Result (RTX 3070, CUDA 12.4, Burn 0.21 / CubeCL 0.10, 2026-06-26): forward & adjoint bit-accurate
//! vs CPU (5.96e-8); coefficient gradient matches CPU `sos_vjp` (1.9e-4) and a finite-difference
//! check (8.9e-3, independent); a resident forward+backward loop is ~40 ms/iter vs ~860 ms if it
//! round-tripped the host.
//!
//! - **adjoint** = the recurrence run backward in time, sections reversed (= `flip·filter·flip`).
//! - **coefficient gradient** = a single pass that builds the all-pole intermediates `w = x/A` and
//!   `v = y/A` inline and accumulates the five per-coeff sums, written per row; the cross-row
//!   reduction (each coeff's gradient sums over all rows) is the tiny `[batch,5]` host sum.
//!
//! Bridge: raw `CubeBackend<R>` float primitive is `CubeTensor<R>` (public `client`/`handle`); launch
//! on those, wrap back with `CubeTensor::new_contiguous`. Generic over `R` → cross-vendor.
//!
//! Next: cascade coeff-grad (orchestrate per-section with resident intermediates) + a
//! `CubeBackend<R>` specialization of `fluxion-autodiff`'s op so `loss.backward()` runs the kernels.

use burn::tensor::{Tensor, TensorPrimitive};
use burn::tensor::backend::Backend;
use burn_cubecl::CubeBackend;
use burn_cubecl::tensor::CubeTensor;
use cubecl::cuda::{CudaDevice, CudaRuntime};
use cubecl::prelude::*;

type Gpu = CubeBackend<CudaRuntime, f32, i32, u8>;
type Ct = CubeTensor<CudaRuntime>;

#[cube(launch)]
fn sos_fwd<F: Float>(input: &Array<F>, output: &mut Array<F>, coeffs: &Array<F>, #[comptime] nf: usize, #[comptime] ns: usize) {
    let nr = input.len() / nf;
    if ABSOLUTE_POS < nr {
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

/// Cascade adjoint (input gradient): same recurrence backward in time, sections reversed.
#[cube(launch)]
fn sos_adj<F: Float>(grad: &Array<F>, output: &mut Array<F>, coeffs: &Array<F>, #[comptime] nf: usize, #[comptime] ns: usize) {
    let nr = grad.len() / nf;
    if ABSOLUTE_POS < nr {
        let base = ABSOLUTE_POS * nf;
        for t in 0..nf {
            output[base + t] = grad[base + t];
        }
        #[unroll]
        for s in 0..ns {
            let ss = ns - 1 - s;
            let c = ss * 5;
            let (b0, b1, b2, a1, a2) = (coeffs[c], coeffs[c + 1], coeffs[c + 2], coeffs[c + 3], coeffs[c + 4]);
            let mut s1 = F::new(0.0);
            let mut s2 = F::new(0.0);
            for t in 0..nf {
                let tt = nf - 1 - t;
                let x = output[base + tt];
                let y = b0 * x + s1;
                s1 = b1 * x - a1 * y + s2;
                s2 = b2 * x - a2 * y;
                output[base + tt] = y;
            }
        }
    }
}

/// Single-biquad coefficient gradient: per-row `[b0,b1,b2,a1,a2]` partials (host reduces over rows).
#[cube(launch)]
fn biquad_cgrad<F: Float>(input: &Array<F>, grad: &Array<F>, coeffs: &Array<F>, out: &mut Array<F>, #[comptime] nf: usize) {
    let nr = input.len() / nf;
    if ABSOLUTE_POS < nr {
        let base = ABSOLUTE_POS * nf;
        let (b0, b1, b2, a1, a2) = (coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4]);
        let mut w1 = F::new(0.0);
        let mut w2 = F::new(0.0); // w = x / A
        let mut v1 = F::new(0.0);
        let mut v2 = F::new(0.0); // v = y / A
        let mut s1 = F::new(0.0);
        let mut s2 = F::new(0.0); // DF2T state for y
        let mut gb0 = F::new(0.0);
        let mut gb1 = F::new(0.0);
        let mut gb2 = F::new(0.0);
        let mut ga1 = F::new(0.0);
        let mut ga2 = F::new(0.0);
        for t in 0..nf {
            let x = input[base + t];
            let g = grad[base + t];
            let wn = x - a1 * w1 - a2 * w2;
            let yn = b0 * x + s1;
            s1 = b1 * x - a1 * yn + s2;
            s2 = b2 * x - a2 * yn;
            let vn = yn - a1 * v1 - a2 * v2;
            gb0 = gb0 + g * wn;
            gb1 = gb1 + g * w1;
            gb2 = gb2 + g * w2;
            ga1 = ga1 - g * v1;
            ga2 = ga2 - g * v2;
            w2 = w1;
            w1 = wn;
            v2 = v1;
            v1 = vn;
        }
        let o = ABSOLUTE_POS * 5;
        out[o] = gb0;
        out[o + 1] = gb1;
        out[o + 2] = gb2;
        out[o + 3] = ga1;
        out[o + 4] = ga2;
    }
}

fn run(x: &Ct, flat: &[f32], nf: usize, ns: usize, adjoint: bool) -> Ct {
    let client = &x.client;
    let n = x.meta.shape().num_elements();
    let rows = n / nf;
    let out_h = client.empty(n * 4);
    let co_h = client.create_from_slice(f32::as_bytes(flat));
    let count = CubeCount::Static(rows.div_ceil(256) as u32, 1, 1);
    let dim = CubeDim::new(client, 256);
    let inp = unsafe { ArrayArg::from_raw_parts(x.handle.clone(), n) };
    let outp = unsafe { ArrayArg::from_raw_parts(out_h.clone(), n) };
    let cop = unsafe { ArrayArg::from_raw_parts(co_h.clone(), flat.len()) };
    if adjoint {
        sos_adj::launch::<f32, CudaRuntime>(client, count, dim, inp, outp, cop, nf, ns);
    } else {
        sos_fwd::launch::<f32, CudaRuntime>(client, count, dim, inp, outp, cop, nf, ns);
    }
    CubeTensor::new_contiguous(client.clone(), x.device.clone(), x.meta.shape().clone(), out_h, x.dtype)
}

fn cgrad(x: &Ct, g: &Ct, c5: &[f32], nf: usize) -> [f32; 5] {
    let client = &x.client;
    let n = x.meta.shape().num_elements();
    let rows = n / nf;
    let out_h = client.empty(rows * 5 * 4);
    let co_h = client.create_from_slice(f32::as_bytes(c5));
    biquad_cgrad::launch::<f32, CudaRuntime>(
        client,
        CubeCount::Static(rows.div_ceil(256) as u32, 1, 1),
        CubeDim::new(client, 256),
        unsafe { ArrayArg::from_raw_parts(x.handle.clone(), n) },
        unsafe { ArrayArg::from_raw_parts(g.handle.clone(), n) },
        unsafe { ArrayArg::from_raw_parts(co_h.clone(), 5) },
        unsafe { ArrayArg::from_raw_parts(out_h.clone(), rows * 5) },
        nf,
    );
    let p = f32::from_bytes(&client.read_one(out_h).unwrap()).to_vec();
    let mut acc = [0.0f32; 5];
    for r in 0..rows {
        for j in 0..5 {
            acc[j] += p[r * 5 + j];
        }
    }
    acc
}

fn cpu_fwd(input: &[f32], nf: usize, sos: &[[f32; 5]]) -> Vec<f32> {
    let mut d = input.to_vec();
    for c in sos {
        for r in 0..d.len() / nf {
            let b = r * nf;
            let (mut s1, mut s2) = (0.0f32, 0.0f32);
            for t in 0..nf {
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

fn cpu_adj(grad: &[f32], nf: usize, sos: &[[f32; 5]]) -> Vec<f32> {
    let mut g = grad.to_vec();
    for c in sos.iter().rev() {
        for r in 0..g.len() / nf {
            let b = r * nf;
            let mut row = g[b..b + nf].to_vec();
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
            g[b..b + nf].copy_from_slice(&row);
        }
    }
    g
}

fn cpu_cgrad(input: &[f32], grad: &[f32], nf: usize, c: &[f32; 5]) -> [f32; 5] {
    let (b0, b1, b2, a1, a2) = (c[0], c[1], c[2], c[3], c[4]);
    let mut acc = [0.0f32; 5];
    for r in 0..input.len() / nf {
        let base = r * nf;
        let (mut w1, mut w2, mut v1, mut v2) = (0.0f32, 0.0, 0.0, 0.0);
        let (mut s1, mut s2) = (0.0f32, 0.0);
        for t in 0..nf {
            let x = input[base + t];
            let g = grad[base + t];
            let wn = x - a1 * w1 - a2 * w2;
            let yn = b0 * x + s1;
            s1 = b1 * x - a1 * yn + s2;
            s2 = b2 * x - a2 * yn;
            let vn = yn - a1 * v1 - a2 * v2;
            acc[0] += g * wn;
            acc[1] += g * w1;
            acc[2] += g * w2;
            acc[3] -= g * v1;
            acc[4] -= g * v2;
            w2 = w1;
            w1 = wn;
            v2 = v1;
            v1 = vn;
        }
    }
    acc
}

/// Finite-difference check of the coefficient gradient (independent of the single-pass formula).
fn fd_cgrad(input: &[f32], grad: &[f32], nf: usize, c: &[f32; 5]) -> [f32; 5] {
    let dot = |cc: &[f32; 5]| -> f32 {
        cpu_fwd(input, nf, std::slice::from_ref(cc)).iter().zip(grad).map(|(a, b)| a * b).sum()
    };
    let mut out = [0.0f32; 5];
    let eps = 1e-3;
    for j in 0..5 {
        let (mut hi, mut lo) = (*c, *c);
        hi[j] += eps;
        lo[j] -= eps;
        out[j] = (dot(&hi) - dot(&lo)) / (2.0 * eps);
    }
    out
}

fn to_ct(v: &[f32], device: &CudaDevice) -> Ct {
    Tensor::<Gpu, 1>::from_floats(v, device).into_primitive().tensor()
}
fn to_vec(ct: Ct) -> Vec<f32> {
    Tensor::<Gpu, 1>::from_primitive(TensorPrimitive::Float(ct)).into_data().to_vec::<f32>().unwrap()
}

fn main() {
    let device: CudaDevice = Default::default();

    // --- forward + adjoint (cascade), bit-accuracy + resident benchmark ---
    let (batch, frames) = (16_384usize, 4_096usize);
    let n = batch * frames;
    let input: Vec<f32> = (0..n).map(|i| ((i % 97) as f32 / 97.0) - 0.5).collect();
    let sos = [[0.2929f32, 0.5858, 0.2929, 0.0, 0.1716], [0.5, 0.3, -0.1, -0.2, 0.05], [0.8, -0.2, 0.1, 0.1, -0.3]];
    let flat: Vec<f32> = sos.iter().flatten().copied().collect();
    let xp = to_ct(&input, &device);
    let d_fwd = to_vec(run(&xp, &flat, frames, sos.len(), false)).iter().zip(&cpu_fwd(&input, frames, &sos)).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    let d_adj = to_vec(run(&xp, &flat, frames, sos.len(), true)).iter().zip(&cpu_adj(&input, frames, &sos)).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    println!("forward max|GPU-CPU| = {d_fwd:.3e}   adjoint max|GPU-CPU| = {d_adj:.3e}");
    assert!(d_fwd < 1e-4 && d_adj < 1e-4);

    let _ = run(&xp, &flat, frames, sos.len(), false);
    let k = 30u32;
    let t = std::time::Instant::now();
    let mut last = xp.clone();
    for _ in 0..k {
        let y = run(&xp, &flat, frames, sos.len(), false);
        last = run(&y, &flat, frames, sos.len(), true);
    }
    let _ = to_vec(last);
    println!("resident fwd+bwd: {:.2} ms/iter", t.elapsed().as_secs_f64() * 1000.0 / k as f64);

    // --- coefficient gradient (single biquad): GPU vs CPU sos_vjp + finite difference ---
    let (cb, cf) = (256usize, 256usize);
    let cn = cb * cf;
    let cx: Vec<f32> = (0..cn).map(|i| ((i % 97) as f32 / 97.0) - 0.5).collect();
    let cg: Vec<f32> = (0..cn).map(|i| ((i % 53) as f32 / 53.0) - 0.4).collect();
    let c = [0.2929f32, 0.5858, 0.2929, 0.0, 0.1716];
    let gpu = cgrad(&to_ct(&cx, &device), &to_ct(&cg, &device), &c, cf);
    let cpu = cpu_cgrad(&cx, &cg, cf, &c);
    let fd = fd_cgrad(&cx, &cg, cf, &c);
    let d_gc = (0..5).fold(0.0f32, |m, j| m.max((gpu[j] - cpu[j]).abs()));
    let d_fd = (0..5).fold(0.0f32, |m, j| m.max((cpu[j] - fd[j]).abs() / (1.0 + cpu[j].abs())));
    println!("coeff-grad max|GPU-CPU| = {d_gc:.3e}   rel|CPU-finitediff| = {d_fd:.3e}");
    assert!(d_gc < 1e-2 && d_fd < 1e-2);
    println!(">>> RESIDENT FORWARD + ADJOINT + COEFF-GRAD OK <<<");
}

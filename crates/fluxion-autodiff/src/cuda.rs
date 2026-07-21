//! GPU-resident differentiable SOS ops (feature `cuda`).
//!
//! The analytic VJP kernels (forward, input-gradient adjoint, coefficient gradient) launch directly
//! on a resident Burn CubeCL tensor, so `loss.backward()` runs them without a host roundtrip for the
//! signal — where the GPU compute win lands (a resident forward+backward is ~40 ms/iter vs ~860 ms
//! round-tripping). Mirrors [`crate::burn_backend`]'s ops, which
//! stay the backend-agnostic (NdArray/CPU) fallback.
//!
//! - [`sos_gpu`] — fixed cascade, differentiates the input (adjoint = recurrence backward in time,
//!   sections reversed).
//! - [`biquad_train_gpu`] — a single trainable biquad: input gradient via the adjoint, coefficient
//!   gradient via the all-pole single-pass kernel (cross-row reduction = a tiny host sum).
//!
//! The raw `CubeBackend<CudaRuntime>` is used (not the fusion-wrapped `burn::backend::Cuda`) so the
//! float primitive is a `CubeTensor` with public `client`/`handle`. The kernels are runtime-generic
//! (`#[cube]`), so this is CUDA-tested but lowers to ROCm/Metal/WGSL too — a cross-vendor backend is
//! a follow-up. Both ops treat a 1-D tensor as a single row; a batched (`[batch, frames]`) entry is
//! the next step (the kernels already handle `n_rows`).

use burn::backend::Autodiff;
use burn::backend::autodiff::checkpoint::base::Checkpointer;
use burn::backend::autodiff::checkpoint::strategy::NoCheckpointing;
use burn::backend::autodiff::grads::Gradients;
use burn::backend::autodiff::ops::{Backward, Ops, OpsKind, binary, unary};
use burn::tensor::ops::FloatTensor;
use burn::tensor::{Tensor, TensorPrimitive};
use burn_cubecl::CubeBackend;
use burn_cubecl::tensor::CubeTensor;
use cubecl::cuda::{CudaDevice, CudaRuntime};
use cubecl::prelude::*;

use fluxion_ops::Biquad;

/// The raw CubeCL CUDA backend; its float primitive is a [`CubeTensor`] we launch kernels on.
pub type Gpu = CubeBackend<CudaRuntime, f32, i32, u8>;
type Ct = CubeTensor<CudaRuntime>;

#[cube(launch)]
fn sos_fwd<F: Float>(
    input: &Array<F>,
    output: &mut Array<F>,
    coeffs: &Array<F>,
    #[comptime] nf: usize,
    #[comptime] ns: usize,
) {
    let nr = input.len() / nf;
    if ABSOLUTE_POS < nr {
        let base = ABSOLUTE_POS * nf;
        for t in 0..nf {
            output[base + t] = input[base + t];
        }
        #[unroll]
        for s in 0..ns {
            let c = s * 5;
            let (b0, b1, b2, a1, a2) = (
                coeffs[c],
                coeffs[c + 1],
                coeffs[c + 2],
                coeffs[c + 3],
                coeffs[c + 4],
            );
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
fn sos_adj<F: Float>(
    grad: &Array<F>,
    output: &mut Array<F>,
    coeffs: &Array<F>,
    #[comptime] nf: usize,
    #[comptime] ns: usize,
) {
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
            let (b0, b1, b2, a1, a2) = (
                coeffs[c],
                coeffs[c + 1],
                coeffs[c + 2],
                coeffs[c + 3],
                coeffs[c + 4],
            );
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
fn biquad_cgrad<F: Float>(
    input: &Array<F>,
    grad: &Array<F>,
    coeffs: &Array<F>,
    out: &mut Array<F>,
    #[comptime] nf: usize,
) {
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

fn flat(sos: &[Biquad]) -> Vec<f32> {
    sos.iter()
        .flat_map(|b| [b.b0, b.b1, b.b2, b.a1, b.a2])
        .collect()
}

/// Run the forward (`adjoint=false`) or adjoint (`true`) SOS kernel on a resident tensor.
fn run(x: &Ct, coeffs: &[f32], nf: usize, ns: usize, adjoint: bool) -> Ct {
    let client = &x.client;
    let n = x.meta.shape().num_elements();
    let rows = n / nf;
    let out_h = client.empty(n * 4);
    let co_h = client.create_from_slice(f32::as_bytes(coeffs));
    let count = CubeCount::Static(rows.div_ceil(256) as u32, 1, 1);
    let dim = CubeDim::new(client, 256);
    let inp = unsafe { ArrayArg::from_raw_parts(x.handle.clone(), n) };
    let outp = unsafe { ArrayArg::from_raw_parts(out_h.clone(), n) };
    let cop = unsafe { ArrayArg::from_raw_parts(co_h.clone(), coeffs.len()) };
    if adjoint {
        sos_adj::launch::<f32, CudaRuntime>(client, count, dim, inp, outp, cop, nf, ns);
    } else {
        sos_fwd::launch::<f32, CudaRuntime>(client, count, dim, inp, outp, cop, nf, ns);
    }
    CubeTensor::new_contiguous(
        client.clone(),
        x.device.clone(),
        x.meta.shape().clone(),
        out_h,
        x.dtype,
    )
}

/// Coefficient gradient for a single biquad, reduced over all rows to `[5]`.
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

// ---------- fixed-coefficient op (differentiates the input only) ----------

#[derive(Debug)]
struct SosGpuBackward {
    coeffs: Vec<f32>,
    ns: usize,
    frames: usize,
}

impl Backward<Gpu, 1> for SosGpuBackward {
    type State = ();
    fn backward(self, ops: Ops<(), 1>, grads: &mut Gradients, _cp: &mut Checkpointer) {
        let (cf, ns, frames) = (self.coeffs, self.ns, self.frames);
        unary::<Gpu, _>(ops.parents, ops.node, grads, move |g: FloatTensor<Gpu>| {
            run(&g, &cf, frames, ns, true)
        });
    }
}

/// Apply a fixed SOS cascade to a resident 1-D tensor as a differentiable op; backward is the
/// analytic adjoint kernel. The GPU analogue of [`crate::burn_backend::sos`].
pub fn sos_gpu(x: Tensor<Autodiff<Gpu>, 1>, sos: &[Biquad]) -> Tensor<Autodiff<Gpu>, 1> {
    let cf = flat(sos);
    let ns = sos.len();
    let x_ad = x.into_primitive().tensor();
    let frames = x_ad.primitive.meta.shape().num_elements();
    let bw = SosGpuBackward {
        coeffs: cf.clone(),
        ns,
        frames,
    };
    let out = match bw
        .prepare::<NoCheckpointing>([x_ad.node.clone()])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => prep.finish((), run(&x_ad.primitive, &cf, frames, ns, false)),
        OpsKind::UnTracked(prep) => prep.finish(run(&x_ad.primitive, &cf, frames, ns, false)),
    };
    Tensor::from_primitive(TensorPrimitive::Float(out))
}

// ---------- single trainable biquad (differentiates input AND coefficients) ----------

#[derive(Debug)]
struct BiquadTrainBackward {
    input: Ct,
    coeffs: [f32; 5],
    frames: usize,
}

impl Backward<Gpu, 2> for BiquadTrainBackward {
    type State = ();
    fn backward(self, ops: Ops<(), 2>, grads: &mut Gradients, _cp: &mut Checkpointer) {
        let cf = self.coeffs.to_vec();
        let cf_x = cf.clone();
        let (input, frames) = (self.input, self.frames);
        let device: CudaDevice = Default::default();
        binary::<Gpu, _, _>(
            ops.parents,
            ops.node,
            grads,
            move |g: FloatTensor<Gpu>| run(&g, &cf_x, frames, 1, true),
            move |g: FloatTensor<Gpu>| {
                let acc = cgrad(&input, &g, &cf, frames);
                Tensor::<Gpu, 1>::from_floats(acc.as_slice(), &device)
                    .into_primitive()
                    .tensor()
            },
        );
    }
}

/// Apply a single trainable biquad (`coeffs = [b0, b1, b2, a1, a2]`) to a resident 1-D tensor;
/// differentiable in both the input and the coefficients. The GPU analogue of
/// [`crate::burn_backend::sos_trainable`] for one section.
pub fn biquad_train_gpu(
    x: Tensor<Autodiff<Gpu>, 1>,
    coeffs: Tensor<Autodiff<Gpu>, 1>,
) -> Tensor<Autodiff<Gpu>, 1> {
    let x_ad = x.into_primitive().tensor();
    let c_ad = coeffs.into_primitive().tensor();
    let cf: Vec<f32> =
        Tensor::<Gpu, 1>::from_primitive(TensorPrimitive::Float(c_ad.primitive.clone()))
            .into_data()
            .to_vec()
            .unwrap();
    let mut c5 = [0.0f32; 5];
    c5.copy_from_slice(&cf[..5]);
    let frames = x_ad.primitive.meta.shape().num_elements();
    let backward = BiquadTrainBackward {
        input: x_ad.primitive.clone(),
        coeffs: c5,
        frames,
    };
    let out = match backward
        .prepare::<NoCheckpointing>([x_ad.node.clone(), c_ad.node.clone()])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => prep.finish((), run(&x_ad.primitive, &cf, frames, 1, false)),
        OpsKind::UnTracked(prep) => prep.finish(run(&x_ad.primitive, &cf, frames, 1, false)),
    };
    Tensor::from_primitive(TensorPrimitive::Float(out))
}

#[cfg(test)]
mod tests {
    // Requires a GPU: cargo test -p fluxion-autodiff --features cuda
    use super::{Gpu, biquad_train_gpu, sos_gpu};
    use burn::backend::Autodiff;
    use burn::tensor::Tensor;
    use fluxion_ops::{Biquad, butterworth_lowpass, sos_filter};

    type B = Autodiff<Gpu>;

    #[test]
    fn input_gradient_gradchecks_on_gpu() {
        let device = Default::default();
        let cascade = butterworth_lowpass(6, 6_000.0, 48_000); // 3 sections
        let xs: Vec<f32> = (0..24).map(|i| (0.3 * i as f32).sin()).collect();
        let seed: Vec<f32> = (0..24).map(|i| (0.17 * i as f32 + 1.0).cos()).collect();

        let x = Tensor::<B, 1>::from_floats(xs.as_slice(), &device).require_grad();
        let loss = (sos_gpu(x.clone(), &cascade)
            * Tensor::<B, 1>::from_floats(seed.as_slice(), &device))
        .sum();
        let gx = x
            .grad(&loss.backward())
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();

        let eps = 1e-3;
        let dot = |v: &[f32]| {
            sos_filter(v, &cascade)
                .iter()
                .zip(&seed)
                .map(|(a, b)| a * b)
                .sum::<f32>()
        };
        for i in 0..xs.len() {
            let (mut hi, mut lo) = (xs.clone(), xs.clone());
            hi[i] += eps;
            lo[i] -= eps;
            let fd = (dot(&hi) - dot(&lo)) / (2.0 * eps);
            assert!(
                (gx[i] - fd).abs() < 1e-2,
                "grad[{i}] = {} vs fd {fd}",
                gx[i]
            );
        }
    }

    #[test]
    fn coefficient_gradient_gradchecks_on_gpu() {
        let device = Default::default();
        let bq = butterworth_lowpass(2, 6_000.0, 48_000)[0];
        let cv = vec![bq.b0, bq.b1, bq.b2, bq.a1, bq.a2];
        let xs: Vec<f32> = (0..20).map(|i| (0.3 * i as f32).sin()).collect();
        let seed: Vec<f32> = (0..20).map(|i| (0.2 * i as f32 + 0.5).cos()).collect();

        let x = Tensor::<B, 1>::from_floats(xs.as_slice(), &device).require_grad();
        let c = Tensor::<B, 1>::from_floats(cv.as_slice(), &device).require_grad();
        let loss = (biquad_train_gpu(x.clone(), c.clone())
            * Tensor::<B, 1>::from_floats(seed.as_slice(), &device))
        .sum();
        let grads = loss.backward();
        let gc = c.grad(&grads).unwrap().into_data().to_vec::<f32>().unwrap();

        let eps = 1e-3;
        let bq_of = |c: &[f32]| Biquad {
            b0: c[0],
            b1: c[1],
            b2: c[2],
            a1: c[3],
            a2: c[4],
        };
        let dot = |c: &[f32]| {
            sos_filter(&xs, &[bq_of(c)])
                .iter()
                .zip(&seed)
                .map(|(a, b)| a * b)
                .sum::<f32>()
        };
        for j in 0..5 {
            let (mut hi, mut lo) = (cv.clone(), cv.clone());
            hi[j] += eps;
            lo[j] -= eps;
            let fd = (dot(&hi) - dot(&lo)) / (2.0 * eps);
            assert!(
                (gc[j] - fd).abs() < 1e-2 * (1.0 + gc[j].abs()),
                "gc[{j}] = {} vs fd {fd}",
                gc[j]
            );
        }
    }
}

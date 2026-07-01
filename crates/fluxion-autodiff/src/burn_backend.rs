//! Burn integration (feature `burn`): "own the backward, rent the graph".
//!
//! Registers fluxion's hand-derived analytic backward as a Burn **custom op**, so an SOS filter runs
//! inside Burn's autograd with the analytic VJP as its gradient — no recursion unrolling.
//! Backend-agnostic: differentiates on `Autodiff<NdArray>` (CPU) or `Autodiff<Cuda>` (GPU, validated
//! in `spikes/f0-burn-cuda`). Forward reuses [`fluxion_ops::sos_filter`].
//!
//! Ops:
//! - [`sos`] — fixed coefficients; differentiates the **input** (the composable gradient).
//! - [`sos_trainable`] — coefficients are a Burn tensor too, so the **filter's own parameters
//!   train** (the DDSP case): input gradient via the adjoint, coefficient gradient via
//!   [`fluxion_ops::sos_vjp`].
//! - [`sos_design`] — the cascade is **designed** from trainable params (`cutoff`/`Q`/`gain`), so
//!   training stays on the always-stable design manifold ([`fluxion_ops::design_param_grad`] ∘
//!   [`fluxion_ops::sos_vjp`]) — the DDSP `cutoff_learnable` reparameterisation.
//! - [`fir_trainable`] — a FIR whose taps are a Burn tensor ([`fluxion_ops::fir_vjp`]).
//! - [`delay`] / [`echo`] — fixed-parameter feedforward/feedback delays; differentiate the input so
//!   they compose inside a differentiable graph ([`fluxion_ops::delay_vjp`] / `echo_vjp`).
//!
//! These run the forward/backward on the host (backend-agnostic). For a GPU-resident path — the
//! kernels launched directly on a resident Burn tensor, no host roundtrip — see [`crate::cuda`]
//! (feature `cuda`).

use burn::backend::autodiff::Autodiff;
use burn::backend::autodiff::checkpoint::base::Checkpointer;
use burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy;
use burn::backend::autodiff::grads::Gradients;
use burn::backend::autodiff::ops::{Backward, Ops, OpsKind, binary, unary};
use burn::tensor::backend::Backend;
use burn::tensor::ops::FloatTensor;
use burn::tensor::{Tensor, TensorData, TensorPrimitive};

use fluxion_ops::{
    Biquad, delay_vjp, design_param_grad, echo_vjp, fir_filter, fir_vjp, sos_filter,
    sos_input_grad, sos_vjp,
};

/// Run `f` over a 1-D float primitive via a host roundtrip (correctness-first; the GPU-kernel path
/// replaces this later). The output length may differ from the input (e.g. a coefficient gradient).
fn map_1d<B: Backend>(
    prim: FloatTensor<B>,
    f: impl FnOnce(Vec<f32>) -> Vec<f32>,
) -> FloatTensor<B> {
    let t = Tensor::<B, 1>::from_primitive(TensorPrimitive::Float(prim));
    let device = t.device();
    let v = t.into_data().to_vec::<f32>().unwrap();
    let out = f(v);
    let n = out.len();
    Tensor::<B, 1>::from_data(TensorData::new(out, [n]), &device)
        .into_primitive()
        .tensor()
}

/// Read a 1-D float primitive's values onto the host.
fn read<B: Backend>(prim: FloatTensor<B>) -> Vec<f32> {
    Tensor::<B, 1>::from_primitive(TensorPrimitive::Float(prim))
        .into_data()
        .to_vec::<f32>()
        .unwrap()
}

/// Interpret a flat `[n_sections·5]` coefficient vector as an SOS cascade.
fn to_sos(cv: &[f32]) -> Vec<Biquad> {
    cv.chunks_exact(5)
        .map(|c| Biquad {
            b0: c[0],
            b1: c[1],
            b2: c[2],
            a1: c[3],
            a2: c[4],
        })
        .collect()
}

// ---------- fixed-coefficient op (differentiates the input only) ----------

#[derive(Debug)]
struct SosBackward(Vec<Biquad>);

impl<B: Backend> Backward<B, 1> for SosBackward {
    type State = ();

    fn backward(self, ops: Ops<(), 1>, grads: &mut Gradients, _cp: &mut Checkpointer) {
        let sos = self.0;
        unary::<B, _>(ops.parents, ops.node, grads, move |g| {
            map_1d::<B>(g, |v| sos_input_grad(&v, &sos))
        });
    }
}

/// Apply an SOS cascade to a 1-D Burn tensor as a **differentiable** op (fixed coefficients):
/// forward via [`fluxion_ops::sos_filter`], backward via the analytic adjoint.
pub fn sos<B: Backend, K: CheckpointStrategy>(
    x: Tensor<Autodiff<B, K>, 1>,
    sos: &[Biquad],
) -> Tensor<Autodiff<B, K>, 1> {
    let sos = sos.to_vec();
    let ad = x.into_primitive().tensor();
    let out = match SosBackward(sos.clone())
        .prepare::<K>([ad.node.clone()])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => {
            prep.finish((), map_1d::<B>(ad.primitive, |v| sos_filter(&v, &sos)))
        }
        OpsKind::UnTracked(prep) => {
            prep.finish(map_1d::<B>(ad.primitive, |v| sos_filter(&v, &sos)))
        }
    };
    Tensor::from_primitive(TensorPrimitive::Float(out))
}

// ---------- trainable-coefficient op (differentiates input AND coefficients) ----------

#[derive(Debug)]
struct SosTrainBackward {
    sos: Vec<Biquad>,
    input: Vec<f32>,
}

impl<B: Backend> Backward<B, 2> for SosTrainBackward {
    type State = ();

    fn backward(self, ops: Ops<(), 2>, grads: &mut Gradients, _cp: &mut Checkpointer) {
        let sos_lhs = self.sos.clone();
        let sos_rhs = self.sos;
        let input = self.input;
        binary::<B, _, _>(
            ops.parents,
            ops.node,
            grads,
            // d/d(input): the adjoint filter.
            move |g| map_1d::<B>(g, move |gv| sos_input_grad(&gv, &sos_lhs)),
            // d/d(coeffs): the analytic coefficient VJP, flattened to [n_sections·5].
            move |g| {
                map_1d::<B>(g, move |gv| {
                    let (_, grad_coeffs) = sos_vjp(&input, &sos_rhs, &gv);
                    grad_coeffs
                        .iter()
                        .flat_map(|b| [b.b0, b.b1, b.b2, b.a1, b.a2])
                        .collect()
                })
            },
        );
    }
}

/// Apply an SOS cascade where the coefficients are themselves a trainable Burn tensor
/// (`[n_sections·5]`, each section `[b0, b1, b2, a1, a2]`). Differentiable in **both** the input
/// and the coefficients — gradients via the analytic adjoint and [`fluxion_ops::sos_vjp`].
pub fn sos_trainable<B: Backend, K: CheckpointStrategy>(
    x: Tensor<Autodiff<B, K>, 1>,
    coeffs: Tensor<Autodiff<B, K>, 1>,
) -> Tensor<Autodiff<B, K>, 1> {
    let x_ad = x.into_primitive().tensor();
    let c_ad = coeffs.into_primitive().tensor();
    let cascade = to_sos(&read::<B>(c_ad.primitive.clone()));
    let input = read::<B>(x_ad.primitive.clone());

    let backward = SosTrainBackward {
        sos: cascade.clone(),
        input,
    };
    let out = match backward
        .prepare::<K>([x_ad.node.clone(), c_ad.node.clone()])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => prep.finish(
            (),
            map_1d::<B>(x_ad.primitive, move |v| sos_filter(&v, &cascade)),
        ),
        OpsKind::UnTracked(prep) => prep.finish(map_1d::<B>(x_ad.primitive, move |v| {
            sos_filter(&v, &cascade)
        })),
    };
    Tensor::from_primitive(TensorPrimitive::Float(out))
}

// ---------- trainable FIR (differentiates input AND taps) ----------

#[derive(Debug)]
struct FirTrainBackward {
    taps: Vec<f32>,
    input: Vec<f32>,
}

impl<B: Backend> Backward<B, 2> for FirTrainBackward {
    type State = ();

    fn backward(self, ops: Ops<(), 2>, grads: &mut Gradients, _cp: &mut Checkpointer) {
        let (taps_x, taps_t) = (self.taps.clone(), self.taps);
        let (in_x, in_t) = (self.input.clone(), self.input);
        binary::<B, _, _>(
            ops.parents,
            ops.node,
            grads,
            // d/d(input): convolution adjoint (correlate the cotangent with the taps).
            move |g| map_1d::<B>(g, move |gv| fir_vjp(&in_x, &taps_x, &gv).0),
            // d/d(taps): correlate the cotangent with the input.
            move |g| map_1d::<B>(g, move |gv| fir_vjp(&in_t, &taps_t, &gv).1),
        );
    }
}

/// Apply a FIR filter whose `taps` are a trainable Burn tensor. Differentiable in **both** the input
/// and the taps (the canonical trained-FIR / DDSP artifact) — gradients via [`fluxion_ops::fir_vjp`].
pub fn fir_trainable<B: Backend, K: CheckpointStrategy>(
    x: Tensor<Autodiff<B, K>, 1>,
    taps: Tensor<Autodiff<B, K>, 1>,
) -> Tensor<Autodiff<B, K>, 1> {
    let x_ad = x.into_primitive().tensor();
    let t_ad = taps.into_primitive().tensor();
    let taps_v = read::<B>(t_ad.primitive.clone());
    let input = read::<B>(x_ad.primitive.clone());

    let backward = FirTrainBackward {
        taps: taps_v.clone(),
        input,
    };
    let out = match backward
        .prepare::<K>([x_ad.node.clone(), t_ad.node.clone()])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => prep.finish(
            (),
            map_1d::<B>(x_ad.primitive, move |v| fir_filter(&v, &taps_v)),
        ),
        OpsKind::UnTracked(prep) => prep.finish(map_1d::<B>(x_ad.primitive, move |v| {
            fir_filter(&v, &taps_v)
        })),
    };
    Tensor::from_primitive(TensorPrimitive::Float(out))
}

// ---------- learnable filter *design* (differentiates input AND design params) ----------

#[derive(Debug)]
struct SosDesignBackward {
    design: fn(&[f32], u32) -> Vec<f32>,
    params: Vec<f32>,
    fs: u32,
    input: Vec<f32>,
}

impl<B: Backend> Backward<B, 2> for SosDesignBackward {
    type State = ();

    fn backward(self, ops: Ops<(), 2>, grads: &mut Gradients, _cp: &mut Checkpointer) {
        let SosDesignBackward {
            design,
            params,
            fs,
            input,
        } = self;
        let sos_x = to_sos(&design(&params, fs)); // for the input adjoint
        binary::<B, _, _>(
            ops.parents,
            ops.node,
            grads,
            // d/d(input): the fixed-coefficient adjoint at the current design.
            move |g| map_1d::<B>(g, move |gv| sos_input_grad(&gv, &sos_x)),
            // d/d(params): ∂L/∂coeffs (sos_vjp) chained through ∂coeffs/∂params (design_param_grad).
            move |g| {
                map_1d::<B>(g, move |gv| {
                    let sos = to_sos(&design(&params, fs));
                    let (_, grad_bq) = sos_vjp(&input, &sos, &gv);
                    let grad_coeffs: Vec<f32> = grad_bq
                        .iter()
                        .flat_map(|b| [b.b0, b.b1, b.b2, b.a1, b.a2])
                        .collect();
                    design_param_grad(&params, &grad_coeffs, |p| design(p, fs))
                })
            },
        );
    }
}

/// Filter `x` through a cascade **designed** from trainable `params`, so the design parameters
/// (`cutoff`, `Q`, `gain`, …) train directly on the always-stable design manifold — the DDSP
/// `cutoff_learnable` reparameterisation (PROJECT.md §8.2). `design(params, fs)` is any closed-form
/// design flattened to `[b0,b1,b2,a1,a2]` per section (e.g. a fn calling
/// [`fluxion_ops::butterworth_lowpass`]). Differentiable in input (adjoint) and params
/// ([`fluxion_ops::design_param_grad`] ∘ [`fluxion_ops::sos_vjp`]).
pub fn sos_design<B: Backend, K: CheckpointStrategy>(
    x: Tensor<Autodiff<B, K>, 1>,
    params: Tensor<Autodiff<B, K>, 1>,
    design: fn(&[f32], u32) -> Vec<f32>,
    fs: u32,
) -> Tensor<Autodiff<B, K>, 1> {
    let x_ad = x.into_primitive().tensor();
    let p_ad = params.into_primitive().tensor();
    let params_v = read::<B>(p_ad.primitive.clone());
    let input = read::<B>(x_ad.primitive.clone());
    let sos_fwd = to_sos(&design(&params_v, fs));

    let backward = SosDesignBackward {
        design,
        params: params_v,
        fs,
        input,
    };
    let out = match backward
        .prepare::<K>([x_ad.node.clone(), p_ad.node.clone()])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => prep.finish(
            (),
            map_1d::<B>(x_ad.primitive, move |v| sos_filter(&v, &sos_fwd)),
        ),
        OpsKind::UnTracked(prep) => prep.finish(map_1d::<B>(x_ad.primitive, move |v| {
            sos_filter(&v, &sos_fwd)
        })),
    };
    Tensor::from_primitive(TensorPrimitive::Float(out))
}

// ---------- delay / echo (fixed params; differentiate the input — composable in a graph) ----------

#[derive(Debug)]
struct DelayBackward {
    samples: usize,
    mix: f32,
}

impl<B: Backend> Backward<B, 1> for DelayBackward {
    type State = ();

    fn backward(self, ops: Ops<(), 1>, grads: &mut Gradients, _cp: &mut Checkpointer) {
        let (samples, mix) = (self.samples, self.mix);
        unary::<B, _>(ops.parents, ops.node, grads, move |g| {
            // Delay is linear in the input, so grad_input is independent of the input values.
            map_1d::<B>(g, move |gv| {
                delay_vjp(&vec![0.0; gv.len()], samples, mix, &gv).0
            })
        });
    }
}

/// Apply a `mix`-blended delay (`samples` fixed) as a differentiable op — input gradient via
/// [`fluxion_ops::delay_vjp`], so a delay composes inside a differentiable graph.
pub fn delay<B: Backend, K: CheckpointStrategy>(
    x: Tensor<Autodiff<B, K>, 1>,
    samples: usize,
    mix: f32,
) -> Tensor<Autodiff<B, K>, 1> {
    let ad = x.into_primitive().tensor();
    let bwd = DelayBackward { samples, mix };
    let out = match bwd
        .prepare::<K>([ad.node.clone()])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => prep.finish(
            (),
            map_1d::<B>(ad.primitive, move |v| fluxion_ops::delay(&v, samples, mix)),
        ),
        OpsKind::UnTracked(prep) => prep.finish(map_1d::<B>(ad.primitive, move |v| {
            fluxion_ops::delay(&v, samples, mix)
        })),
    };
    Tensor::from_primitive(TensorPrimitive::Float(out))
}

#[derive(Debug)]
struct EchoBackward {
    samples: usize,
    feedback: f32,
    wet: f32,
}

impl<B: Backend> Backward<B, 1> for EchoBackward {
    type State = ();

    fn backward(self, ops: Ops<(), 1>, grads: &mut Gradients, _cp: &mut Checkpointer) {
        let EchoBackward {
            samples,
            feedback,
            wet,
        } = self;
        unary::<B, _>(ops.parents, ops.node, grads, move |g| {
            // Echo is linear in the input; grad_input is the adjoint loop, independent of input.
            map_1d::<B>(g, move |gv| {
                echo_vjp(&vec![0.0; gv.len()], samples, feedback, wet, &gv).0
            })
        });
    }
}

/// Apply a feedback `echo` (`samples`/`feedback`/`wet` fixed) as a differentiable op — input gradient
/// via [`fluxion_ops::echo_vjp`]'s adjoint feedback loop.
pub fn echo<B: Backend, K: CheckpointStrategy>(
    x: Tensor<Autodiff<B, K>, 1>,
    samples: usize,
    feedback: f32,
    wet: f32,
) -> Tensor<Autodiff<B, K>, 1> {
    let ad = x.into_primitive().tensor();
    let bwd = EchoBackward {
        samples,
        feedback,
        wet,
    };
    let out = match bwd
        .prepare::<K>([ad.node.clone()])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => prep.finish(
            (),
            map_1d::<B>(ad.primitive, move |v| {
                fluxion_ops::echo(&v, samples, feedback, wet)
            }),
        ),
        OpsKind::UnTracked(prep) => prep.finish(map_1d::<B>(ad.primitive, move |v| {
            fluxion_ops::echo(&v, samples, feedback, wet)
        })),
    };
    Tensor::from_primitive(TensorPrimitive::Float(out))
}

#[cfg(test)]
mod tests {
    use super::{delay, echo, fir_trainable, sos, sos_design, sos_trainable, to_sos};
    use burn::backend::{Autodiff, NdArray};
    use burn::tensor::Tensor;
    use fluxion_ops::{Biquad, butterworth_lowpass, fir_filter, sos_filter};

    type B = Autodiff<NdArray>;

    #[test]
    fn input_gradient_gradchecks() {
        let device = Default::default();
        let cascade = butterworth_lowpass(6, 6_000.0, 48_000); // 3 sections
        let xs: Vec<f32> = (0..24).map(|i| (0.3 * i as f32).sin()).collect();
        let seed: Vec<f32> = (0..24).map(|i| (0.17 * i as f32 + 1.0).cos()).collect();

        let x = Tensor::<B, 1>::from_floats(xs.as_slice(), &device).require_grad();
        let loss = (sos(x.clone(), &cascade)
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

    fn bq_of(c: &[f32]) -> Biquad {
        Biquad {
            b0: c[0],
            b1: c[1],
            b2: c[2],
            a1: c[3],
            a2: c[4],
        }
    }

    #[test]
    fn coefficient_gradient_gradchecks() {
        let device = Default::default();
        let bq = butterworth_lowpass(2, 6_000.0, 48_000)[0];
        let cv = vec![bq.b0, bq.b1, bq.b2, bq.a1, bq.a2];
        let xs: Vec<f32> = (0..20).map(|i| (0.3 * i as f32).sin()).collect();
        let seed: Vec<f32> = (0..20).map(|i| (0.2 * i as f32 + 0.5).cos()).collect();

        let x = Tensor::<B, 1>::from_floats(xs.as_slice(), &device).require_grad();
        let c = Tensor::<B, 1>::from_floats(cv.as_slice(), &device).require_grad();
        let loss = (sos_trainable(x.clone(), c.clone())
            * Tensor::<B, 1>::from_floats(seed.as_slice(), &device))
        .sum();
        let grads = loss.backward();
        let gc = c.grad(&grads).unwrap().into_data().to_vec::<f32>().unwrap();

        let eps = 1e-3;
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

    #[test]
    fn fir_gradients_gradcheck() {
        let device = Default::default();
        let taps = vec![0.2f32, -0.5, 0.3, 0.1];
        let xs: Vec<f32> = (0..24).map(|i| (0.3 * i as f32).sin()).collect();
        let seed: Vec<f32> = (0..24).map(|i| (0.17 * i as f32 + 1.0).cos()).collect();
        let seed_t = Tensor::<B, 1>::from_floats(seed.as_slice(), &device);

        let x = Tensor::<B, 1>::from_floats(xs.as_slice(), &device).require_grad();
        let t = Tensor::<B, 1>::from_floats(taps.as_slice(), &device).require_grad();
        let loss = (fir_trainable(x.clone(), t.clone()) * seed_t).sum();
        let grads = loss.backward();
        let gx = x.grad(&grads).unwrap().into_data().to_vec::<f32>().unwrap();
        let gt = t.grad(&grads).unwrap().into_data().to_vec::<f32>().unwrap();

        let eps = 1e-3;
        let dot_x = |v: &[f32]| {
            fir_filter(v, &taps)
                .iter()
                .zip(&seed)
                .map(|(a, b)| a * b)
                .sum::<f32>()
        };
        for i in 0..xs.len() {
            let (mut hi, mut lo) = (xs.clone(), xs.clone());
            hi[i] += eps;
            lo[i] -= eps;
            let fd = (dot_x(&hi) - dot_x(&lo)) / (2.0 * eps);
            assert!((gx[i] - fd).abs() < 1e-2, "gx[{i}] = {} vs fd {fd}", gx[i]);
        }
        let dot_t = |tp: &[f32]| {
            fir_filter(&xs, tp)
                .iter()
                .zip(&seed)
                .map(|(a, b)| a * b)
                .sum::<f32>()
        };
        for j in 0..taps.len() {
            let (mut hi, mut lo) = (taps.clone(), taps.clone());
            hi[j] += eps;
            lo[j] -= eps;
            let fd = (dot_t(&hi) - dot_t(&lo)) / (2.0 * eps);
            assert!((gt[j] - fd).abs() < 1e-2, "gt[{j}] = {} vs fd {fd}", gt[j]);
        }
    }

    /// A 4th-order Butterworth low-pass designed from `params[0] = cutoff`, flattened to coeffs.
    fn lp4(p: &[f32], fs: u32) -> Vec<f32> {
        butterworth_lowpass(4, p[0], fs)
            .iter()
            .flat_map(|b| [b.b0, b.b1, b.b2, b.a1, b.a2])
            .collect()
    }

    #[test]
    fn design_param_gradient_gradchecks() {
        // The learnable-cutoff op: ∂L/∂cutoff flows through Burn's tape via the design Jacobian.
        let device = Default::default();
        let fs = 48_000u32;
        let xs: Vec<f32> = (0..64).map(|i| (0.3 * i as f32).sin()).collect();
        let seed: Vec<f32> = (0..64).map(|i| (0.2 * i as f32 + 0.5).cos()).collect();
        let seed_t = Tensor::<B, 1>::from_floats(seed.as_slice(), &device);
        let cutoff = 3_000.0f32;

        let x = Tensor::<B, 1>::from_floats(xs.as_slice(), &device).require_grad();
        let p = Tensor::<B, 1>::from_floats([cutoff].as_slice(), &device).require_grad();
        let loss = (sos_design(x.clone(), p.clone(), lp4, fs) * seed_t).sum();
        let gp = p
            .grad(&loss.backward())
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();

        // FD of the loss w.r.t. cutoff through the whole design → filter.
        let dot = |c: f32| {
            let sos = to_sos(&lp4(&[c], fs));
            sos_filter(&xs, &sos)
                .iter()
                .zip(&seed)
                .map(|(a, b)| a * b)
                .sum::<f32>()
        };
        let h = 1.0;
        let fd = (dot(cutoff + h) - dot(cutoff - h)) / (2.0 * h);
        assert!(
            (gp[0] - fd).abs() <= 1e-2 * fd.abs().max(1e-3),
            "cutoff grad {} vs fd {fd}",
            gp[0]
        );
    }

    #[test]
    fn delay_echo_input_gradients_gradcheck() {
        let device = Default::default();
        let xs: Vec<f32> = (0..32).map(|i| (0.3 * i as f32).sin()).collect();
        let seed: Vec<f32> = (0..32).map(|i| (0.2 * i as f32 + 0.5).cos()).collect();
        let seed_t = Tensor::<B, 1>::from_floats(seed.as_slice(), &device);
        let eps = 1e-3;

        // delay
        let x = Tensor::<B, 1>::from_floats(xs.as_slice(), &device).require_grad();
        let loss = (delay(x.clone(), 5, 0.7) * seed_t.clone()).sum();
        let gx = x
            .grad(&loss.backward())
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let dot_d = |v: &[f32]| {
            fluxion_ops::delay(v, 5, 0.7)
                .iter()
                .zip(&seed)
                .map(|(a, b)| a * b)
                .sum::<f32>()
        };
        for i in 0..xs.len() {
            let (mut hi, mut lo) = (xs.clone(), xs.clone());
            hi[i] += eps;
            lo[i] -= eps;
            let fd = (dot_d(&hi) - dot_d(&lo)) / (2.0 * eps);
            assert!(
                (gx[i] - fd).abs() < 1e-2,
                "delay gx[{i}] = {} vs fd {fd}",
                gx[i]
            );
        }

        // echo
        let x = Tensor::<B, 1>::from_floats(xs.as_slice(), &device).require_grad();
        let loss = (echo(x.clone(), 4, 0.5, 0.6) * seed_t).sum();
        let gx = x
            .grad(&loss.backward())
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let dot_e = |v: &[f32]| {
            fluxion_ops::echo(v, 4, 0.5, 0.6)
                .iter()
                .zip(&seed)
                .map(|(a, b)| a * b)
                .sum::<f32>()
        };
        for i in 0..xs.len() {
            let (mut hi, mut lo) = (xs.clone(), xs.clone());
            hi[i] += eps;
            lo[i] -= eps;
            let fd = (dot_e(&hi) - dot_e(&lo)) / (2.0 * eps);
            assert!(
                (gx[i] - fd).abs() < 1e-2,
                "echo gx[{i}] = {} vs fd {fd}",
                gx[i]
            );
        }
    }

    #[test]
    fn fits_filter_coefficients() {
        // Train the b-coefficients (a fixed) of a biquad to match a target, via the analytic
        // coefficient gradient through Burn. Convex least-squares → converges.
        let device = Default::default();
        let denom = butterworth_lowpass(2, 4_000.0, 48_000)[0];
        let mut s = 0x1234_5678u32;
        let xs: Vec<f32> = (0..128)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 9) as f32 / (1u32 << 22) as f32 - 1.0
            })
            .collect();
        let target = Biquad {
            b0: 0.5,
            b1: 0.3,
            b2: -0.2,
            a1: denom.a1,
            a2: denom.a2,
        };
        let target_y = sos_filter(&xs, &[target]);

        let xt = Tensor::<B, 1>::from_floats(xs.as_slice(), &device);
        let tt = Tensor::<B, 1>::from_floats(target_y.as_slice(), &device);
        let w = sos_filter(
            &xs,
            &[Biquad {
                b0: 1.0,
                b1: 0.0,
                b2: 0.0,
                a1: denom.a1,
                a2: denom.a2,
            }],
        );
        let lr = 1.0 / (3.0 * w.iter().map(|v| v * v).sum::<f32>());

        let mut coeffs = vec![0.0f32, 0.0, 0.0, denom.a1, denom.a2];
        let mse = |c: &[f32]| {
            sos_filter(&xs, &[bq_of(c)])
                .iter()
                .zip(&target_y)
                .map(|(y, t)| (y - t) * (y - t))
                .sum::<f32>()
        };
        let initial = mse(&coeffs);
        for _ in 0..3_000 {
            let ct = Tensor::<B, 1>::from_floats(coeffs.as_slice(), &device).require_grad();
            let pred = sos_trainable(xt.clone(), ct.clone());
            let loss = (pred - tt.clone()).powf_scalar(2.0).sum();
            let gc = ct
                .grad(&loss.backward())
                .unwrap()
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            for i in 0..3 {
                coeffs[i] -= lr * gc[i]; // train only the numerator (denominator fixed)
            }
        }
        assert!(
            mse(&coeffs) < initial * 1e-3,
            "did not converge: {initial} -> {}",
            mse(&coeffs)
        );
        for (got, want) in [(coeffs[0], 0.5), (coeffs[1], 0.3), (coeffs[2], -0.2)] {
            assert!((got - want).abs() < 1e-2, "coeff {got} vs {want}");
        }
    }
}

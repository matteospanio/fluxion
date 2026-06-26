//! Burn integration (feature `burn`): "own the backward, rent the graph".
//!
//! Registers fluxion's hand-derived analytic backward as a Burn **custom op**, so an SOS filter runs
//! inside Burn's autograd with the analytic VJP as its gradient — no recursion unrolling.
//! Backend-agnostic: differentiates on `Autodiff<NdArray>` (CPU) or `Autodiff<Cuda>` (GPU, validated
//! in `spikes/f0-burn-cuda`). Forward reuses [`fluxion_ops::sos_filter`].
//!
//! Two ops:
//! - [`sos`] — fixed coefficients; differentiates the **input** (the composable gradient).
//! - [`sos_trainable`] — coefficients are a Burn tensor too, so the **filter's own parameters
//!   train** (the DDSP case): input gradient via the adjoint, coefficient gradient via
//!   [`fluxion_ops::sos_vjp`].
//!
//! Next: the GPU-kernel forward/backward (wire `fluxion-backend::cuda` in instead of the host
//! roundtrip) so the differentiable path is GPU-accelerated end-to-end.

use burn::backend::autodiff::Autodiff;
use burn::backend::autodiff::checkpoint::base::Checkpointer;
use burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy;
use burn::backend::autodiff::grads::Gradients;
use burn::backend::autodiff::ops::{Backward, Ops, OpsKind, binary, unary};
use burn::tensor::backend::Backend;
use burn::tensor::ops::FloatTensor;
use burn::tensor::{Tensor, TensorData, TensorPrimitive};

use fluxion_ops::{Biquad, sos_filter, sos_input_grad, sos_vjp};

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

#[cfg(test)]
mod tests {
    use super::{sos, sos_trainable};
    use burn::backend::{Autodiff, NdArray};
    use burn::tensor::Tensor;
    use fluxion_ops::{Biquad, butterworth_lowpass, sos_filter};

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

//! Burn integration (feature `burn`): "own the backward, rent the graph".
//!
//! Registers fluxion's hand-derived analytic backward as a Burn **custom op**, so a biquad runs
//! inside Burn's autograd with the analytic LTI adjoint as its gradient — no recursion unrolling.
//! Backend-agnostic: the same code differentiates on `Autodiff<NdArray>` (CPU) or `Autodiff<Cuda>`
//! (GPU, validated in `spikes/f0-burn-cuda`). The forward reuses [`fluxion_ops::biquad_forward`].
//!
//! Today this covers the single-biquad input gradient (the composable one). Coefficient gradients
//! ([`fluxion_ops::biquad_vjp`] already derives them) and the GPU-kernel-accelerated forward/backward
//! (wiring `fluxion-backend::cuda` into the custom op instead of the host roundtrip) are next.

use burn::backend::autodiff::Autodiff;
use burn::backend::autodiff::checkpoint::base::Checkpointer;
use burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy;
use burn::backend::autodiff::grads::Gradients;
use burn::backend::autodiff::ops::{Backward, Ops, OpsKind, unary};
use burn::tensor::backend::Backend;
use burn::tensor::ops::FloatTensor;
use burn::tensor::{Tensor, TensorData, TensorPrimitive};

use fluxion_ops::{Biquad, biquad_forward};

/// Run `f` over a 1-D float primitive via a host roundtrip (correctness-first; the GPU-kernel path
/// replaces this later).
fn map_1d<B: Backend>(prim: FloatTensor<B>, f: impl FnOnce(Vec<f32>) -> Vec<f32>) -> FloatTensor<B> {
    let t = Tensor::<B, 1>::from_primitive(TensorPrimitive::Float(prim));
    let device = t.device();
    let n = t.dims()[0];
    let v = t.into_data().to_vec::<f32>().unwrap();
    let out = f(v);
    Tensor::<B, 1>::from_data(TensorData::new(out, [n]), &device)
        .into_primitive()
        .tensor()
}

#[derive(Debug)]
struct BiquadBackward(Biquad);

impl<B: Backend> Backward<B, 1> for BiquadBackward {
    type State = ();

    fn backward(self, ops: Ops<(), 1>, grads: &mut Gradients, _cp: &mut Checkpointer) {
        let bq = self.0;
        // Adjoint of an LTI filter = flip · filter · flip (the input-gradient VJP, no unrolling).
        unary::<B, _>(ops.parents, ops.node, grads, move |g| {
            map_1d::<B>(g, |mut v| {
                v.reverse();
                let mut y = biquad_forward(&v, &bq);
                y.reverse();
                y
            })
        });
    }
}

/// Apply a biquad to a 1-D Burn tensor as a **differentiable** op: forward via
/// [`fluxion_ops::biquad_forward`], backward via the analytic adjoint.
pub fn biquad<B: Backend, K: CheckpointStrategy>(
    x: Tensor<Autodiff<B, K>, 1>,
    bq: Biquad,
) -> Tensor<Autodiff<B, K>, 1> {
    let ad = x.into_primitive().tensor();
    let out = match BiquadBackward(bq)
        .prepare::<K>([ad.node.clone()])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => {
            prep.finish((), map_1d::<B>(ad.primitive, |v| biquad_forward(&v, &bq)))
        }
        OpsKind::UnTracked(prep) => {
            prep.finish(map_1d::<B>(ad.primitive, |v| biquad_forward(&v, &bq)))
        }
    };
    Tensor::from_primitive(TensorPrimitive::Float(out))
}

#[cfg(test)]
mod tests {
    use super::biquad;
    use burn::backend::{Autodiff, NdArray};
    use burn::tensor::Tensor;
    use fluxion_ops::{biquad_forward, butterworth_lowpass};

    #[test]
    fn custom_analytic_backward_gradchecks() {
        type B = Autodiff<NdArray>;
        let device = Default::default();
        let bq = butterworth_lowpass(2, 6_000.0, 48_000)[0];
        let xs: Vec<f32> = (0..16).map(|i| (0.3 * i as f32).sin()).collect();
        let seed: Vec<f32> = (0..16).map(|i| (0.17 * i as f32 + 1.0).cos()).collect();

        // Gradient of <seed, biquad(x)> w.r.t. x, via Burn autograd + our custom backward.
        let x = Tensor::<B, 1>::from_floats(xs.as_slice(), &device).require_grad();
        let loss = (biquad(x.clone(), bq) * Tensor::<B, 1>::from_floats(seed.as_slice(), &device))
            .sum();
        let grads = loss.backward();
        let gx = x.grad(&grads).unwrap().into_data().to_vec::<f32>().unwrap();

        // Finite-difference reference.
        let eps = 1e-3;
        let dot = |v: &[f32]| biquad_forward(v, &bq).iter().zip(&seed).map(|(a, b)| a * b).sum::<f32>();
        for i in 0..xs.len() {
            let (mut hi, mut lo) = (xs.clone(), xs.clone());
            hi[i] += eps;
            lo[i] -= eps;
            let fd = (dot(&hi) - dot(&lo)) / (2.0 * eps);
            assert!((gx[i] - fd).abs() < 1e-2, "grad[{i}] = {} vs fd {fd}", gx[i]);
        }
    }
}

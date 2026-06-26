//! Burn integration (feature `burn`): "own the backward, rent the graph".
//!
//! Registers fluxion's hand-derived analytic backward as a Burn **custom op**, so an SOS filter runs
//! inside Burn's autograd with the analytic LTI adjoint as its gradient — no recursion unrolling.
//! Backend-agnostic: the same code differentiates on `Autodiff<NdArray>` (CPU) or `Autodiff<Cuda>`
//! (GPU, validated in `spikes/f0-burn-cuda`). Forward reuses [`fluxion_ops::sos_filter`], backward
//! [`fluxion_ops::sos_input_grad`].
//!
//! Today this is the **input gradient** for an arbitrary SOS cascade (the composable one). Next:
//! coefficient gradients ([`fluxion_ops::sos_vjp`] derives them — a binary custom op over input and
//! a trainable coeff tensor) and the GPU-kernel forward/backward (wire `fluxion-backend::cuda` into
//! the op instead of the host roundtrip).

use burn::backend::autodiff::Autodiff;
use burn::backend::autodiff::checkpoint::base::Checkpointer;
use burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy;
use burn::backend::autodiff::grads::Gradients;
use burn::backend::autodiff::ops::{Backward, Ops, OpsKind, unary};
use burn::tensor::backend::Backend;
use burn::tensor::ops::FloatTensor;
use burn::tensor::{Tensor, TensorData, TensorPrimitive};

use fluxion_ops::{Biquad, sos_filter, sos_input_grad};

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
struct SosBackward(Vec<Biquad>);

impl<B: Backend> Backward<B, 1> for SosBackward {
    type State = ();

    fn backward(self, ops: Ops<(), 1>, grads: &mut Gradients, _cp: &mut Checkpointer) {
        let sos = self.0;
        // Adjoint of the cascade = the analytic input-gradient VJP (no recursion unrolling).
        unary::<B, _>(ops.parents, ops.node, grads, move |g| {
            map_1d::<B>(g, |v| sos_input_grad(&v, &sos))
        });
    }
}

/// Apply an SOS cascade to a 1-D Burn tensor as a **differentiable** op: forward via
/// [`fluxion_ops::sos_filter`], backward via the analytic adjoint [`fluxion_ops::sos_input_grad`].
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
        OpsKind::Tracked(prep) => prep.finish((), map_1d::<B>(ad.primitive, |v| sos_filter(&v, &sos))),
        OpsKind::UnTracked(prep) => prep.finish(map_1d::<B>(ad.primitive, |v| sos_filter(&v, &sos))),
    };
    Tensor::from_primitive(TensorPrimitive::Float(out))
}

#[cfg(test)]
mod tests {
    use super::sos;
    use burn::backend::{Autodiff, NdArray};
    use burn::tensor::Tensor;
    use fluxion_ops::{butterworth_lowpass, sos_filter};

    /// Gradient flows *through* the custom-backward filter to train an upstream Burn parameter:
    /// fit a scalar gain `g` so that `filter(g·x)` matches `filter(2·x)`. `g` should converge to 2.
    #[test]
    fn trains_upstream_gain_through_filter() {
        type B = Autodiff<NdArray>;
        let device = Default::default();
        let cascade = butterworth_lowpass(4, 5_000.0, 48_000);
        let xs: Vec<f32> = (0..64).map(|i| (0.2 * i as f32).sin()).collect();
        let x = Tensor::<B, 1>::from_floats(xs.as_slice(), &device);

        // Fixed target = filter(2·x), computed on the CPU so it carries no autograd graph.
        let scaled: Vec<f32> = xs.iter().map(|v| 2.0 * v).collect();
        let target = Tensor::<B, 1>::from_floats(sos_filter(&scaled, &cascade).as_slice(), &device);

        // Step size from the regressor energy E = ‖filter(x)‖² (the problem is convex in g).
        let energy: f32 = sos_filter(&xs, &cascade).iter().map(|v| v * v).sum();
        let lr = 0.2 / energy;

        let mut g = Tensor::<B, 1>::from_floats([0.0], &device).require_grad();
        for _ in 0..200 {
            let pred = sos(x.clone() * g.clone(), &cascade); // g broadcasts over x; grad flows via our adjoint
            let loss = (pred - target.clone()).powf_scalar(2.0).sum();
            let grads = loss.backward();
            let gg = g.grad(&grads).unwrap();
            g = Tensor::from_inner(g.inner() - gg * lr).require_grad();
        }
        let fitted = g.into_data().to_vec::<f32>().unwrap()[0];
        assert!((fitted - 2.0).abs() < 0.05, "g did not converge: {fitted}");
    }

    #[test]
    fn custom_analytic_backward_gradchecks() {
        type B = Autodiff<NdArray>;
        let device = Default::default();
        let cascade = butterworth_lowpass(6, 6_000.0, 48_000); // 3 sections
        let xs: Vec<f32> = (0..24).map(|i| (0.3 * i as f32).sin()).collect();
        let seed: Vec<f32> = (0..24).map(|i| (0.17 * i as f32 + 1.0).cos()).collect();

        // Gradient of <seed, sos_filter(x)> w.r.t. x, via Burn autograd + our analytic backward.
        let x = Tensor::<B, 1>::from_floats(xs.as_slice(), &device).require_grad();
        let loss =
            (sos(x.clone(), &cascade) * Tensor::<B, 1>::from_floats(seed.as_slice(), &device)).sum();
        let grads = loss.backward();
        let gx = x.grad(&grads).unwrap().into_data().to_vec::<f32>().unwrap();

        // Finite-difference reference.
        let eps = 1e-3;
        let dot =
            |v: &[f32]| sos_filter(v, &cascade).iter().zip(&seed).map(|(a, b)| a * b).sum::<f32>();
        for i in 0..xs.len() {
            let (mut hi, mut lo) = (xs.clone(), xs.clone());
            hi[i] += eps;
            lo[i] -= eps;
            let fd = (dot(&hi) - dot(&lo)) / (2.0 * eps);
            assert!((gx[i] - fd).abs() < 1e-2, "grad[{i}] = {} vs fd {fd}", gx[i]);
        }
    }
}

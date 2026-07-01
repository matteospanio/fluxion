//! Whole-graph differentiation (plan task E12): lower a fluxion [`Graph`] onto Burn's autograd by
//! implementing [`fluxion_backend::Backend`] over Burn tensors. Then [`fluxion_backend::eval`] — the
//! *same* graph walk the CPU executor uses (plan task C1) — composes every op through Burn's tape:
//! series composes adjoints, parallel sums cotangents. The input is differentiated end-to-end
//! (each filter's coefficients are fixed at their design; train them per-op with [`crate::burn_backend::sos_design`]).

use core::marker::PhantomData;

use burn::backend::autodiff::Autodiff;
use burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend as BurnBackend;

use fluxion_backend::{Backend, eval, is_differentiable};
use fluxion_core::Graph;
use fluxion_ops::Biquad;

use crate::burn_backend::{delay as delay_op, echo as echo_op, sos as sos_op};

/// A [`fluxion_backend::Backend`] over Burn autodiff tensors — one channel is one differentiable 1-D
/// tensor. Filters / gain / delay / echo lower to the analytic-VJP custom ops in
/// [`crate::burn_backend`]; the cross-channel / non-differentiable `normalize` and `reverb` are
/// rejected up front by [`fluxion_backend::is_differentiable`], so they never reach these stubs.
struct BurnAd<B, K>(PhantomData<(B, K)>);

impl<B: BurnBackend, K: CheckpointStrategy> Backend for BurnAd<B, K> {
    type Buf = Tensor<Autodiff<B, K>, 1>;

    fn filter(&self, x: Self::Buf, sos: &[Biquad]) -> Self::Buf {
        sos_op(x, sos)
    }
    fn gain(&self, x: Self::Buf, factor: f32) -> Self::Buf {
        x.mul_scalar(factor)
    }
    fn delay(&self, x: Self::Buf, samples: usize, mix: f32) -> Self::Buf {
        delay_op(x, samples, mix)
    }
    fn echo(&self, x: Self::Buf, samples: usize, feedback: f32, wet: f32) -> Self::Buf {
        echo_op(x, samples, feedback, wet)
    }
    fn fir(&self, _x: Self::Buf, _taps: &[f32]) -> Self::Buf {
        unimplemented!(
            "FIR-in-graph isn't differentiable here (use fir_trainable) — guard with is_differentiable"
        )
    }
    fn normalize(&self, _x: Self::Buf, _peak: f32) -> Self::Buf {
        unimplemented!(
            "normalize is cross-channel / non-differentiable — guard with is_differentiable"
        )
    }
    fn reverb(&self, _x: Self::Buf, _room: f32, _damping: f32, _mix: f32) -> Self::Buf {
        unimplemented!("reverb is non-differentiable — guard with is_differentiable")
    }
    fn add(&self, a: Self::Buf, b: Self::Buf) -> Self::Buf {
        a + b
    }
    fn feedback(
        &self,
        _x: Self::Buf,
        _forward: &fluxion_core::Graph,
        _feedback: &fluxion_core::Graph,
        _fs: u32,
    ) -> Self::Buf {
        unimplemented!(
            "feedback (~) is sample-recursive — not differentiable here; guard with is_differentiable"
        )
    }
}

/// Differentiably run a whole [`Graph`] on a 1-D Burn autodiff tensor at sample rate `fs`: the input
/// is differentiated end-to-end through the composed analytic VJPs, so `loss.backward()` flows a
/// gradient through an entire effect chain. Returns `None` if the graph contains a non-differentiable
/// op (`Normalize` / `Reverb`) — see [`fluxion_backend::is_differentiable`].
///
/// # Examples
/// ```
/// use burn::backend::{Autodiff, NdArray};
/// use burn::tensor::Tensor;
/// use fluxion_autodiff::graph::diff_process;
/// use fluxion_core::{Graph, OpKind};
///
/// type B = Autodiff<NdArray>;
/// let g = Graph::op(OpKind::Lowpass, [2_000.0, 4.0]) | Graph::op(OpKind::Gain, [0.5]);
/// let x = Tensor::<B, 1>::from_floats([0.1f32, -0.2, 0.3, 0.4].as_slice(), &Default::default())
///     .require_grad();
/// let y = diff_process(&g, x.clone(), 48_000).unwrap();
/// let _grad = x.grad(&y.sum().backward()); // gradient flows through the whole chain
/// ```
pub fn diff_process<B: BurnBackend, K: CheckpointStrategy>(
    graph: &Graph,
    x: Tensor<Autodiff<B, K>, 1>,
    fs: u32,
) -> Option<Tensor<Autodiff<B, K>, 1>> {
    if !is_differentiable(graph, fs) {
        return None;
    }
    Some(eval(&BurnAd(PhantomData), graph, x, fs))
}

#[cfg(test)]
mod tests {
    use super::diff_process;
    use burn::backend::{Autodiff, NdArray};
    use burn::tensor::Tensor;
    use fluxion_backend::process;
    use fluxion_core::{Graph, OpKind, Signal};
    use fluxion_ops::{butterworth_highpass, butterworth_lowpass, sos_input_grad};

    type B = Autodiff<NdArray>;

    // (lowpass | gain 0.5) in parallel with a highpass, then an echo — series + parallel + gain.
    fn graph() -> Graph {
        ((Graph::op(OpKind::Lowpass, [2_000.0, 4.0]) | Graph::op(OpKind::Gain, [0.5]))
            + Graph::op(OpKind::Highpass, [500.0, 2.0]))
            | Graph::op(OpKind::Echo, [0.01, 0.4, 0.5])
    }

    #[test]
    fn diff_process_rejects_non_differentiable() {
        let device = Default::default();
        let g =
            Graph::op(OpKind::Lowpass, [1_000.0, 2.0]) | Graph::op(OpKind::Reverb, [0.5, 0.5, 0.3]);
        let x = Tensor::<B, 1>::from_floats([0.0f32; 8].as_slice(), &device);
        assert!(diff_process(&g, x, 48_000).is_none());
    }

    #[test]
    fn diff_process_forward_matches_cpu_process() {
        // The same `eval` walk over two backends must produce the same forward output.
        let device = Default::default();
        let fs = 48_000;
        let xs: Vec<f32> = (0..256).map(|i| (0.2 * i as f32).sin()).collect();
        let cpu = process(&graph(), &Signal::new(fs, vec![xs.clone()])).channels[0].clone();
        let x = Tensor::<B, 1>::from_floats(xs.as_slice(), &device);
        let burn = diff_process(&graph(), x, fs)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        for (a, b) in cpu.iter().zip(&burn) {
            assert!((a - b).abs() < 1e-4, "cpu {a} vs burn {b}");
        }
    }

    #[test]
    fn whole_graph_gradient_matches_cpu_analytic_adjoint() {
        // The composed Burn adjoint must equal the hand-derived CPU adjoint (bit-exact — a far
        // tighter check than finite difference, which is wildly inaccurate for filter graphs).
        // g = (lowpass(4) | gain 0.5) + highpass(2). Its input adjoint is
        //   series(lowpass, ×0.5) ⇒ 0.5·lowpass_adj(seed) ; parallel(+) ⇒ + highpass_adj(seed).
        let device = Default::default();
        let fs = 48_000u32;
        let n = 96;
        let seed: Vec<f32> = (0..n).map(|i| (0.2 * i as f32 + 0.5).cos()).collect();
        let seed_t = Tensor::<B, 1>::from_floats(seed.as_slice(), &device);
        let xs: Vec<f32> = (0..n).map(|i| (0.3 * i as f32).sin()).collect();
        let g = (Graph::op(OpKind::Lowpass, [2_000.0, 4.0]) | Graph::op(OpKind::Gain, [0.5]))
            + Graph::op(OpKind::Highpass, [500.0, 2.0]);

        let x = Tensor::<B, 1>::from_floats(xs.as_slice(), &device).require_grad();
        let loss = (diff_process(&g, x.clone(), fs).unwrap() * seed_t).sum();
        let gb = x
            .grad(&loss.backward())
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();

        let ga_lp = sos_input_grad(&seed, &butterworth_lowpass(4, 2_000.0, fs));
        let ga_hp = sos_input_grad(&seed, &butterworth_highpass(2, 500.0, fs));
        for i in 0..n {
            let analytic = 0.5 * ga_lp[i] + ga_hp[i];
            assert!(
                (gb[i] - analytic).abs() < 1e-4,
                "grad[{i}] = {} vs analytic {analytic}",
                gb[i]
            );
        }
    }
}

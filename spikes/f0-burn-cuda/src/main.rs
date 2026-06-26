//! F0 spike — prove GPU forward + autodiff on CUDA via Burn (the project's intended substrate).
//!
//! Result (RTX 3070, CUDA 12.4, Burn 0.21, 2026-06-26): **GO**.
//! - `CUDA forward  x*2 = [2.0, 4.0, 6.0, 8.0]`
//! - gradient descent fits `w -> 2.00000` (loss 120 -> 0) entirely on the GPU.
//!
//! This validates that the cross-vendor differentiable substrate (Burn + CubeCL) compiles and
//! runs on NVIDIA, de-risking plan tasks C4 (Burn backend) and E6 (Autodiff integration).

use burn::backend::{Autodiff, Cuda};
use burn::tensor::Tensor;

type B = Cuda;
type AB = Autodiff<B>;

fn main() {
    let device = Default::default();

    // 1) Plain forward on the GPU.
    let x = Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0, 4.0], &device);
    let yv = (x * 2.0).into_data().to_vec::<f32>().unwrap();
    println!("CUDA forward  x*2 = {yv:?}");
    assert_eq!(yv, vec![2.0, 4.0, 6.0, 8.0]);

    // 2) Autodiff on the GPU: fit w to minimize sum((w*x - target)^2), target = 2*x.
    let x = Tensor::<AB, 1>::from_floats([1.0, 2.0, 3.0, 4.0], &device);
    let target = Tensor::<AB, 1>::from_floats([2.0, 4.0, 6.0, 8.0], &device);
    let mut w = Tensor::<AB, 1>::from_floats([0.0], &device).require_grad();
    let lr = 0.01;
    for step in 0..=200 {
        let diff = x.clone() * w.clone() - target.clone();
        let loss = diff.clone().mul(diff).sum();
        if step % 50 == 0 {
            let l = loss.clone().into_data().to_vec::<f32>().unwrap()[0];
            let wv = w.clone().into_data().to_vec::<f32>().unwrap()[0];
            println!("step {step:3}: loss={l:.5} w={wv:.5}");
        }
        let grads = loss.backward();
        let g = w.grad(&grads).expect("w has a gradient");
        w = Tensor::from_inner(w.inner() - g * lr).require_grad();
    }
    let wv = w.into_data().to_vec::<f32>().unwrap()[0];
    println!("fitted w = {wv:.5} (target 2.0)");
    assert!((wv - 2.0).abs() < 1e-2, "did not converge: {wv}");
    println!("\n>>> GPU FORWARD + AUTODIFF OK <<<");
}

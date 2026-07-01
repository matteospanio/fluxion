//! Learn a low-pass **cutoff** by gradient descent through Burn's autograd.
//!
//! Run with: `cargo run -p fluxion-autodiff --example learn_cutoff --features burn`
//!
//! This is the DDSP core end-to-end: a filter *designed* from a single trainable `cutoff` parameter
//! (kept on the always-stable design manifold) is optimised by back-propagating an audio-domain loss
//! through fluxion's analytic VJP, which is registered as a Burn custom op ([`sos_design`]). Only the
//! cutoff trains — the coefficients are always a valid, stable Butterworth design.

use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;
use fluxion_autodiff::burn_backend::sos_design;
use fluxion_ops::{Biquad, butterworth_lowpass, sos_filter};

type B = Autodiff<NdArray>;

/// A 6th-order Butterworth low-pass designed from `params[0] = cutoff`, flattened to coefficients.
fn lp6(p: &[f32], fs: u32) -> Vec<f32> {
    butterworth_lowpass(6, p[0], fs)
        .iter()
        .flat_map(|b| [b.b0, b.b1, b.b2, b.a1, b.a2])
        .collect()
}

fn to_sos(flat: &[f32]) -> Vec<Biquad> {
    flat.chunks_exact(5)
        .map(|c| Biquad {
            b0: c[0],
            b1: c[1],
            b2: c[2],
            a1: c[3],
            a2: c[4],
        })
        .collect()
}

fn main() {
    let device = Default::default();
    let fs = 48_000u32;

    // Broadband input, and a target = the same input low-passed at the cutoff we want to recover.
    let xs: Vec<f32> = (0..2048)
        .map(|k| (k as f32 * 0.1).sin() + 0.5 * (k as f32 * 0.031).cos())
        .collect();
    let target_cutoff = 3_000.0f32;
    let target = sos_filter(&xs, &to_sos(&lp6(&[target_cutoff], fs)));

    let x = Tensor::<B, 1>::from_floats(xs.as_slice(), &device);
    let target_t = Tensor::<B, 1>::from_floats(target.as_slice(), &device);

    // Rprop on the cutoff: the Hz-scale gradient magnitude is tiny, so step on its sign with an
    // adaptive step size (no learning-rate tuning). The gradient itself flows through Burn's tape.
    let mut cutoff = 1_200.0f32;
    let (mut step, mut prev) = (400.0f32, 0.0f32);
    println!("target cutoff = {target_cutoff} Hz, starting from {cutoff} Hz");
    for it in 0..60 {
        let p = Tensor::<B, 1>::from_floats([cutoff].as_slice(), &device).require_grad();
        let loss = (sos_design(x.clone(), p.clone(), lp6, fs) - target_t.clone())
            .powf_scalar(2.0)
            .sum();
        let loss_v = loss.clone().into_data().to_vec::<f32>().unwrap()[0];
        let g = p
            .grad(&loss.backward())
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap()[0];

        step = if prev * g < 0.0 {
            step * 0.5
        } else {
            step * 1.2
        }
        .clamp(0.5, 1_000.0);
        cutoff = (cutoff - g.signum() * step).clamp(200.0, 20_000.0);
        prev = g;
        if it % 10 == 0 {
            println!("  iter {it:2}: cutoff = {cutoff:7.1} Hz   loss = {loss_v:.4e}");
        }
    }

    println!("learned cutoff = {cutoff:.1} Hz (target {target_cutoff})");
    assert!(
        (cutoff - target_cutoff).abs() < 100.0,
        "did not converge: learned {cutoff}, want {target_cutoff}"
    );
    println!("✓ converged");
}

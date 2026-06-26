//! Train a biquad's numerator coefficients to match a target, using the analytic gradient.
//!
//! Run with: `cargo run -p fluxion-ops --example fit_filter`
//!
//! This is the engine-independent core of DDSP: a forward op plus its hand-derived backward pass is
//! enough to do gradient descent — no autograd framework involved. The Burn / torch integrations
//! later reuse this exact `biquad_vjp` as a custom backward.

use fluxion_ops::{Biquad, biquad_forward, biquad_vjp, butterworth_lowpass};

/// Deterministic broadband pseudo-noise in [-1, 1) (LCG).
fn pseudo_noise(n: usize) -> Vec<f32> {
    let mut s = 0x1234_5678u32;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 9) as f32 / (1u32 << 22) as f32 - 1.0
        })
        .collect()
}

fn main() {
    let x = pseudo_noise(256);
    let denom = butterworth_lowpass(2, 4_000.0, 48_000)[0];
    let target = Biquad {
        b0: 0.5,
        b1: 0.3,
        b2: -0.2,
        ..denom
    };
    let target_y = biquad_forward(&x, &target);

    // Safe, data-derived step size: lr = 1/(3·‖w‖²) where w = 1/A(z)·x is the all-pole intermediate.
    let w = biquad_forward(
        &x,
        &Biquad {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            ..denom
        },
    );
    let lr = 1.0 / (3.0 * w.iter().map(|v| v * v).sum::<f32>());

    let mut bq = Biquad {
        b0: 0.0,
        b1: 0.0,
        b2: 0.0,
        ..denom
    };
    let mse = |bq: &Biquad| -> f32 {
        biquad_forward(&x, bq)
            .iter()
            .zip(&target_y)
            .map(|(y, t)| (y - t) * (y - t))
            .sum::<f32>()
            / x.len() as f32
    };

    println!("step      mse         b0      b1      b2");
    for step in 0..=5_000 {
        if step % 500 == 0 {
            println!(
                "{step:>4}  {:.3e}   {:+.3}  {:+.3}  {:+.3}",
                mse(&bq),
                bq.b0,
                bq.b1,
                bq.b2
            );
        }
        let y = biquad_forward(&x, &bq);
        let resid: Vec<f32> = y.iter().zip(&target_y).map(|(y, t)| y - t).collect();
        let (_, g) = biquad_vjp(&x, &bq, &resid);
        bq.b0 -= lr * g.b0;
        bq.b1 -= lr * g.b1;
        bq.b2 -= lr * g.b2;
    }
    println!(
        "\ntarget : b0={:+.3} b1={:+.3} b2={:+.3}",
        target.b0, target.b1, target.b2
    );
    println!(
        "fitted : b0={:+.3} b1={:+.3} b2={:+.3}",
        bq.b0, bq.b1, bq.b2
    );
}

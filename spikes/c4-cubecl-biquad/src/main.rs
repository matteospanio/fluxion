//! C4/F2 spike — a batched biquad as a single CubeCL kernel.
//!
//! One thread per row runs the IIR recurrence over time, so a whole DL batch is filtered in a
//! single GPU launch. Result (RTX 3070, CUDA 12.4, CubeCL 0.10, 2026-06-26): bit-accurate vs the
//! CPU (max diff 5.96e-8) and **~59x faster** than a single CPU core on 67 Msamples
//! (16384 rows x 4096) — GPU 5.7 ms/iter (warm) vs CPU 336 ms.
//!
//! This is the cross-vendor IIR kernel the whole GPU lane rests on (CubeCL lowers the same `#[cube]`
//! kernel to CUDA / ROCm / Metal / Vulkan / WGSL). A full cascade is this kernel looped over
//! sections; the differentiable backward (analytic VJP) + Burn-tensor integration are E6/C4 next.

use cubecl::cuda::CudaRuntime;
use cubecl::prelude::*;

#[cube(launch)]
fn biquad_kernel<F: Float>(
    input: &Array<F>,
    output: &mut Array<F>,
    coeffs: &Array<F>, // [b0, b1, b2, a1, a2]
    #[comptime] n_frames: usize,
) {
    // `Array::len()` is usize; `ABSOLUTE_POS` (u32) coerces into usize arithmetic.
    let n_rows = input.len() / n_frames;
    if ABSOLUTE_POS < n_rows {
        let base = ABSOLUTE_POS * n_frames;
        let b0 = coeffs[0];
        let b1 = coeffs[1];
        let b2 = coeffs[2];
        let a1 = coeffs[3];
        let a2 = coeffs[4];
        let mut s1 = F::new(0.0);
        let mut s2 = F::new(0.0);
        for t in 0..n_frames {
            let x = input[base + t];
            let y = b0 * x + s1; // Direct Form II Transposed, same as the CPU kernel
            s1 = b1 * x - a1 * y + s2;
            s2 = b2 * x - a2 * y;
            output[base + t] = y;
        }
    }
}

fn cpu_biquad(input: &[f32], frames: usize, c: &[f32; 5]) -> Vec<f32> {
    let mut out = vec![0.0f32; input.len()];
    for r in 0..input.len() / frames {
        let base = r * frames;
        let (mut s1, mut s2) = (0.0f32, 0.0f32);
        for t in 0..frames {
            let x = input[base + t];
            let y = c[0] * x + s1;
            s1 = c[1] * x - c[3] * y + s2;
            s2 = c[2] * x - c[4] * y;
            out[base + t] = y;
        }
    }
    out
}

fn main() {
    type R = CudaRuntime;
    let client = R::client(&Default::default());

    let (batch, frames) = (16384usize, 4096usize);
    let n = batch * frames;
    let input: Vec<f32> = (0..n).map(|i| ((i % 97) as f32 / 97.0) - 0.5).collect();
    let coeffs = [0.2929f32, 0.5858, 0.2929, 0.0, 0.1716]; // stable 2nd-order low-pass

    let in_h = client.create_from_slice(f32::as_bytes(&input));
    let out_h = client.create_from_slice(f32::as_bytes(&vec![0.0f32; n]));
    let co_h = client.create_from_slice(f32::as_bytes(&coeffs));
    let cube_dim = CubeDim::new(&client, 256);
    let cubes = batch.div_ceil(256) as u32;
    let launch = || {
        biquad_kernel::launch::<f32, R>(
            &client,
            CubeCount::Static(cubes, 1, 1),
            cube_dim,
            unsafe { ArrayArg::from_raw_parts(in_h.clone(), n) },
            unsafe { ArrayArg::from_raw_parts(out_h.clone(), n) },
            unsafe { ArrayArg::from_raw_parts(co_h.clone(), 5) },
            frames,
        )
    };

    // Warmup: the first launch JIT-compiles the kernel via NVRTC.
    launch();
    let gpu_out = f32::from_bytes(&client.read_one(out_h.clone()).unwrap()).to_vec();

    // Timed GPU compute: K launches, a single sync at the end.
    let k = 30u32;
    let t0 = std::time::Instant::now();
    for _ in 0..k {
        launch();
    }
    let _ = client.read_one(out_h.clone()).unwrap();
    let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0 / k as f64;

    let t1 = std::time::Instant::now();
    let cpu_out = cpu_biquad(&input, frames, &coeffs);
    let cpu_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let maxdiff = gpu_out.iter().zip(&cpu_out).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    println!("batch={batch} frames={frames}  ({} Msamples)", n / 1_000_000);
    println!("max |GPU-CPU| = {maxdiff:.3e}");
    println!(
        "GPU {gpu_ms:.1} ms/iter (warm)   CPU(1 core) {cpu_ms:.1} ms   speedup ~{:.0}x",
        cpu_ms / gpu_ms
    );
    assert!(maxdiff < 1e-4, "GPU/CPU mismatch");
    println!(">>> CUDA BATCHED BIQUAD OK <<<");
}

//! Bridge spike — run the SOS CubeCL kernel directly on a **resident** Burn tensor (no host
//! roundtrip), so the GPU compute win actually lands in a training-loop-style workflow.
//!
//! Result (RTX 3070, CUDA 12.4, Burn 0.21 / CubeCL 0.10, 2026-06-26): bit-accurate (5.96e-8), and a
//! **resident** forward runs at ~20 ms/iter vs ~430 ms for the transfer-bound one-shot path
//! (`fluxion-backend::cuda::sos_filter_batch`) and ~300 ms on a CPU core.
//!
//! How: Burn's raw `CubeBackend<R>` float primitive is `CubeTensor<R>`, whose `client` and `handle`
//! fields are public — so the kernel launches straight on the resident buffer, and the result is
//! wrapped back with `CubeTensor::new_contiguous(client, device, shape, handle, dtype)`. Generic over
//! the CubeCL runtime `R`, so the same path lowers to CUDA / ROCm / Metal / Vulkan / WGSL.
//! (Use the raw `CubeBackend<CudaRuntime, …>`, not `burn::backend::Cuda`, which is fusion-wrapped.)
//!
//! Next (F1/F3): wire this into `fluxion-autodiff`'s differentiable op so the forward AND the
//! analytic backward run as kernels on resident tensors — fully GPU-resident training.

use burn::tensor::{Tensor, TensorPrimitive};
use burn_cubecl::CubeBackend;
use burn_cubecl::tensor::CubeTensor;
use cubecl::cuda::CudaRuntime;
use cubecl::prelude::*;

type Gpu = CubeBackend<CudaRuntime, f32, i32, u8>;

#[cube(launch)]
fn sos_kernel<F: Float>(
    input: &Array<F>,
    output: &mut Array<F>,
    coeffs: &Array<F>,
    #[comptime] n_frames: usize,
    #[comptime] n_sections: usize,
) {
    let n_rows = input.len() / n_frames;
    if ABSOLUTE_POS < n_rows {
        let base = ABSOLUTE_POS * n_frames;
        for t in 0..n_frames {
            output[base + t] = input[base + t];
        }
        #[unroll]
        for s in 0..n_sections {
            let c = s * 5;
            let (b0, b1, b2, a1, a2) = (coeffs[c], coeffs[c + 1], coeffs[c + 2], coeffs[c + 3], coeffs[c + 4]);
            let mut s1 = F::new(0.0);
            let mut s2 = F::new(0.0);
            for t in 0..n_frames {
                let x = output[base + t];
                let y = b0 * x + s1;
                s1 = b1 * x - a1 * y + s2;
                s2 = b2 * x - a2 * y;
                output[base + t] = y;
            }
        }
    }
}

/// Launch the kernel directly on the resident tensor's `client` + `handle` (no host roundtrip).
fn sos_gpu(x: &CubeTensor<CudaRuntime>, flat: &[f32], frames: usize, sections: usize) -> CubeTensor<CudaRuntime> {
    let client = &x.client;
    let n = x.meta.shape().num_elements();
    let rows = n / frames;
    let out_h = client.empty(n * 4);
    let co_h = client.create_from_slice(f32::as_bytes(flat));
    sos_kernel::launch::<f32, CudaRuntime>(
        client,
        CubeCount::Static(rows.div_ceil(256) as u32, 1, 1),
        CubeDim::new(client, 256),
        unsafe { ArrayArg::from_raw_parts(x.handle.clone(), n) },
        unsafe { ArrayArg::from_raw_parts(out_h.clone(), n) },
        unsafe { ArrayArg::from_raw_parts(co_h.clone(), flat.len()) },
        frames,
        sections,
    );
    CubeTensor::new_contiguous(client.clone(), x.device.clone(), x.meta.shape().clone(), out_h, x.dtype)
}

fn cpu_sos(input: &[f32], frames: usize, sos: &[[f32; 5]]) -> Vec<f32> {
    let mut d = input.to_vec();
    for c in sos {
        for r in 0..d.len() / frames {
            let b = r * frames;
            let (mut s1, mut s2) = (0.0f32, 0.0f32);
            for t in 0..frames {
                let x = d[b + t];
                let y = c[0] * x + s1;
                s1 = c[1] * x - c[3] * y + s2;
                s2 = c[2] * x - c[4] * y;
                d[b + t] = y;
            }
        }
    }
    d
}

fn main() {
    let device = Default::default();
    let (batch, frames) = (16_384usize, 4_096usize);
    let n = batch * frames;
    let input: Vec<f32> = (0..n).map(|i| ((i % 97) as f32 / 97.0) - 0.5).collect();
    let sos = [
        [0.2929f32, 0.5858, 0.2929, 0.0, 0.1716],
        [0.5, 0.3, -0.1, -0.2, 0.05],
        [0.8, -0.2, 0.1, 0.1, -0.3],
    ];
    let flat: Vec<f32> = sos.iter().flatten().copied().collect();

    // x lives on the GPU (resident).
    let xp: CubeTensor<CudaRuntime> =
        Tensor::<Gpu, 1>::from_floats(input.as_slice(), &device).into_primitive().tensor();

    // Correctness.
    let yp = sos_gpu(&xp, &flat, frames, sos.len());
    let gpu = Tensor::<Gpu, 1>::from_primitive(TensorPrimitive::Float(yp)).into_data().to_vec::<f32>().unwrap();
    let cpu = cpu_sos(&input, frames, &sos);
    let maxdiff = gpu.iter().zip(&cpu).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    println!("resident forward, max |GPU-CPU| = {maxdiff:.3e}");
    assert!(maxdiff < 1e-4);

    // Resident benchmark: x is already on the GPU, so there is no upload per call.
    let _ = sos_gpu(&xp, &flat, frames, sos.len());
    let k = 30u32;
    let t = std::time::Instant::now();
    let mut last = sos_gpu(&xp, &flat, frames, sos.len());
    for _ in 1..k {
        last = sos_gpu(&xp, &flat, frames, sos.len());
    }
    let _ = Tensor::<Gpu, 1>::from_primitive(TensorPrimitive::Float(last)).into_data(); // force sync
    println!("resident GPU forward: {:.2} ms/iter", t.elapsed().as_secs_f64() * 1000.0 / k as f64);
    println!(">>> BURN<->CUBECL BRIDGE OK <<<");
}

//! GPU execution via CubeCL (feature `cuda`).
//!
//! Requires an NVIDIA GPU + CUDA toolkit (`libnvrtc`) to build and run; this whole module is gated
//! behind the `cuda` feature so the default build stays pure-Rust and offline.
//!
//! The kernel is the cross-vendor IIR kernel proven in `spikes/c4-cubecl-biquad` (CubeCL lowers the
//! same `#[cube]` kernel to CUDA / ROCm / Metal / Vulkan / WGSL): one thread per batch row runs the
//! whole SOS cascade over time in a single launch. The output matches the CPU
//! [`fluxion_ops::sos_filter`] per row to `f32` precision.
//!
//! Not yet here: the analytic backward (E6 — register `fluxion_ops::sos_vjp` as a Burn custom
//! gradient so this is differentiable) and the Burn-tensor / batched-`Signal` integration.

use cubecl::cuda::CudaRuntime;
use cubecl::prelude::*;

use fluxion_ops::Biquad;

#[cube(launch)]
fn sos_kernel<F: Float>(
    input: &Array<F>,
    output: &mut Array<F>,
    coeffs: &Array<F>, // n_sections * [b0, b1, b2, a1, a2]
    #[comptime] n_frames: usize,
    #[comptime] n_sections: usize,
) {
    // `Array::len()` is usize; `ABSOLUTE_POS` (u32) coerces into usize arithmetic.
    let n_rows = input.len() / n_frames;
    if ABSOLUTE_POS < n_rows {
        let base = ABSOLUTE_POS * n_frames;
        // Copy the row in, then apply each section in place (Direct Form II Transposed).
        for t in 0..n_frames {
            output[base + t] = input[base + t];
        }
        #[unroll]
        for s in 0..n_sections {
            let c = s * 5;
            let b0 = coeffs[c];
            let b1 = coeffs[c + 1];
            let b2 = coeffs[c + 2];
            let a1 = coeffs[c + 3];
            let a2 = coeffs[c + 4];
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

/// Filter a flat batch of `input.len() / frames` rows (each `frames` samples, contiguous) through
/// the SOS cascade on the GPU. Equivalent to applying [`fluxion_ops::sos_filter`] to each row.
///
/// ponytail: creates a fresh CubeCL client per call. A persistent device/client is the production
/// path (avoids re-init and keeps the JIT kernel cache warm); fine for now / for correctness tests.
pub fn sos_filter_batch(input: &[f32], frames: usize, sos: &[Biquad]) -> Vec<f32> {
    assert!(frames > 0 && !sos.is_empty() && !input.is_empty() && input.len() % frames == 0);
    let n = input.len();
    let rows = n / frames;
    let flat: Vec<f32> = sos
        .iter()
        .flat_map(|b| [b.b0, b.b1, b.b2, b.a1, b.a2])
        .collect();

    let client = CudaRuntime::client(&Default::default());
    let in_h = client.create_from_slice(f32::as_bytes(input));
    let out_h = client.create_from_slice(f32::as_bytes(&vec![0.0f32; n]));
    let co_h = client.create_from_slice(f32::as_bytes(&flat));

    sos_kernel::launch::<f32, CudaRuntime>(
        &client,
        CubeCount::Static(rows.div_ceil(256) as u32, 1, 1),
        CubeDim::new(&client, 256),
        unsafe { ArrayArg::from_raw_parts(in_h.clone(), n) },
        unsafe { ArrayArg::from_raw_parts(out_h.clone(), n) },
        unsafe { ArrayArg::from_raw_parts(co_h.clone(), flat.len()) },
        frames,
        sos.len(),
    );

    f32::from_bytes(&client.read_one(out_h).unwrap()).to_vec()
}

#[cfg(test)]
mod tests {
    use super::sos_filter_batch;
    use fluxion_ops::{butterworth_lowpass, sos_filter};

    #[test]
    fn gpu_batch_matches_cpu_per_row() {
        let (rows, frames) = (64usize, 512usize);
        let input: Vec<f32> =
            (0..rows * frames).map(|i| ((i % 71) as f32 / 71.0) - 0.5).collect();
        let sos = butterworth_lowpass(6, 4_000.0, 48_000); // 3-section cascade

        let gpu = sos_filter_batch(&input, frames, &sos);

        let mut cpu = vec![0.0f32; input.len()];
        for r in 0..rows {
            let out = sos_filter(&input[r * frames..(r + 1) * frames], &sos);
            cpu[r * frames..(r + 1) * frames].copy_from_slice(&out);
        }

        let maxdiff = gpu.iter().zip(&cpu).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(maxdiff < 1e-4, "GPU/CPU max diff {maxdiff}");
    }
}

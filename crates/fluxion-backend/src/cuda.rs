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
//! The differentiable analytic backward (E6) lives in `fluxion-autodiff` (host roundtrip) and the
//! batched API in `process_batch`. Still ahead: wiring this kernel into the differentiable op's
//! forward/backward (GPU-accelerated training, where the data is resident and the ~59× compute win
//! actually lands), since one-shot batch filtering here is transfer-bound.

use std::sync::OnceLock;

use cubecl::cuda::CudaRuntime;
use cubecl::prelude::*;

use fluxion_ops::Biquad;

/// A process-wide CubeCL client, so the JIT-compiled kernel stays cached across calls (a fresh
/// client per call re-initialises and re-JITs — ~100× slower for repeated batches).
fn client() -> &'static ComputeClient<CudaRuntime> {
    static CLIENT: OnceLock<ComputeClient<CudaRuntime>> = OnceLock::new();
    CLIENT.get_or_init(|| CudaRuntime::client(&Default::default()))
}

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
/// Borrowed-slice convenience wrapper over [`sos_filter_batch_owned`]; the copy into an owned
/// buffer is the price of the borrow (the runtime needs owned data for its async upload).
pub fn sos_filter_batch(input: &[f32], frames: usize, sos: &[Biquad]) -> Vec<f32> {
    sos_filter_batch_owned(input.to_vec(), frames, sos)
}

/// [`sos_filter_batch`] taking ownership of the input: the buffer is handed to the runtime
/// zero-copy and uploaded through pipelined pinned staging (~6.4 GB/s on PCIe 3.0 with the
/// patched CubeCL — see the cubecl fork note in the workspace Cargo.toml).
///
/// One-shot calls are **transfer-bound**: the kernel is a ~23 ms sliver in a ~106 ms round trip
/// for 67 Msamples. The GPU pays off when data is resident and reused across launches; the CubeCL
/// client is cached (module-level `client`) so the JIT'd kernel is reused across calls.
pub fn sos_filter_batch_owned(input: Vec<f32>, frames: usize, sos: &[Biquad]) -> Vec<f32> {
    assert!(frames > 0 && !sos.is_empty() && !input.is_empty() && input.len() % frames == 0);
    let n = input.len();
    let rows = n / frames;
    let flat: Vec<f32> = sos
        .iter()
        .flat_map(|b| [b.b0, b.b1, b.b2, b.a1, b.a2])
        .collect();

    let client = client();
    let in_h = client.create(cubecl::bytes::Bytes::from_elems(input));
    let out_h = client.empty(n * std::mem::size_of::<f32>()); // no 268 MB zero-upload
    let co_h = client.create_from_slice(f32::as_bytes(&flat));

    sos_kernel::launch::<f32, CudaRuntime>(
        client,
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
        let input: Vec<f32> = (0..rows * frames)
            .map(|i| ((i % 71) as f32 / 71.0) - 0.5)
            .collect();
        let sos = butterworth_lowpass(6, 4_000.0, 48_000); // 3-section cascade

        let gpu = sos_filter_batch(&input, frames, &sos);

        let mut cpu = vec![0.0f32; input.len()];
        for r in 0..rows {
            let out = sos_filter(&input[r * frames..(r + 1) * frames], &sos);
            cpu[r * frames..(r + 1) * frames].copy_from_slice(&out);
        }

        let maxdiff = gpu
            .iter()
            .zip(&cpu)
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(maxdiff < 1e-4, "GPU/CPU max diff {maxdiff}");
    }

    // Run with: cargo test -p fluxion-backend --features cuda -- --ignored --nocapture
    #[test]
    #[ignore = "GPU benchmark"]
    fn bench_repeated_calls() {
        let (rows, frames) = (16_384usize, 4_096usize);
        let input: Vec<f32> = (0..rows * frames)
            .map(|i| ((i % 97) as f32 / 97.0) - 0.5)
            .collect();
        let sos = butterworth_lowpass(6, 4_000.0, 48_000);
        for k in 0..5 {
            let t = std::time::Instant::now();
            let _ = sos_filter_batch(&input, frames, &sos);
            println!("call {k}: {:.1} ms", t.elapsed().as_secs_f64() * 1000.0);
        }
    }
}

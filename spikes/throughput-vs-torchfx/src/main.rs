//! GPU SOS-cascade throughput: fluxion's CubeCL kernel, three regimes (see README.md for the
//! head-to-head vs torchfx). 16384×4096 = 67 Msamples, order-8 (4 sections).
//!
//! - **resident reused-out** — input + output stay on the GPU (pure kernel).
//! - **resident alloc-out** — output allocated per call (matches torchfx returning a new tensor).
//! - **one-shot transfer** — upload input + download output per call (the "filter a file" path).

use cubecl::cuda::CudaRuntime;
use cubecl::prelude::*;
use std::time::Instant;

#[cube(launch)]
fn sos_kernel<F: Float>(input: &Array<F>, output: &mut Array<F>, coeffs: &Array<F>, #[comptime] nf: usize, #[comptime] ns: usize) {
    let n_rows = input.len() / nf;
    if ABSOLUTE_POS < n_rows {
        let base = ABSOLUTE_POS * nf;
        for t in 0..nf {
            output[base + t] = input[base + t];
        }
        #[unroll]
        for s in 0..ns {
            let c = s * 5;
            let (b0, b1, b2, a1, a2) = (coeffs[c], coeffs[c + 1], coeffs[c + 2], coeffs[c + 3], coeffs[c + 4]);
            let mut s1 = F::new(0.0);
            let mut s2 = F::new(0.0);
            for t in 0..nf {
                let x = output[base + t];
                let y = b0 * x + s1;
                s1 = b1 * x - a1 * y + s2;
                s2 = b2 * x - a2 * y;
                output[base + t] = y;
            }
        }
    }
}

fn main() {
    type R = CudaRuntime;
    let client = R::client(&Default::default());
    let (batch, frames, ns) = (16384usize, 4096usize, 4usize);
    let n = batch * frames;
    let input: Vec<f32> = (0..n).map(|i| ((i % 97) as f32 / 97.0) - 0.5).collect();
    let sec = [0.2929f32, 0.5858, 0.2929, 0.0, 0.1716];
    let coeffs: Vec<f32> = (0..ns).flat_map(|_| sec).collect();
    let in_h = client.create_from_slice(f32::as_bytes(&input));
    let co_h = client.create_from_slice(f32::as_bytes(&coeffs));
    let out_h = client.create_from_slice(f32::as_bytes(&vec![0.0f32; n]));
    let dim = CubeDim::new(&client, 256);
    let cubes = batch.div_ceil(256) as u32;
    let nc = coeffs.len();
    let msamp = n as f64 / 1e6;
    let k = 50u32;

    macro_rules! fire {
        ($out:expr) => {
            sos_kernel::launch::<f32, R>(&client, CubeCount::Static(cubes, 1, 1), dim,
                unsafe { ArrayArg::from_raw_parts(in_h.clone(), n) },
                unsafe { ArrayArg::from_raw_parts($out, n) },
                unsafe { ArrayArg::from_raw_parts(co_h.clone(), nc) }, frames, ns);
        };
    }

    fire!(out_h.clone());
    let _ = client.read_one(out_h.clone()).unwrap(); // warmup (NVRTC JIT)

    let t0 = Instant::now();
    for _ in 0..k {
        fire!(out_h.clone());
    }
    let _ = client.read_one(out_h.clone()).unwrap();
    let el = t0.elapsed().as_secs_f64() / k as f64;
    println!("fluxion GPU resident reused-out : {:6.3} ms   {:7.0} Msamples/s", el * 1000.0, msamp / el);

    let t0 = Instant::now();
    let mut last = out_h.clone();
    for _ in 0..k {
        let o = client.empty(n * 4);
        fire!(o.clone());
        last = o;
    }
    let _ = client.read_one(last).unwrap();
    let el = t0.elapsed().as_secs_f64() / k as f64;
    println!("fluxion GPU resident alloc-out  : {:6.3} ms   {:7.0} Msamples/s", el * 1000.0, msamp / el);

    let t0 = Instant::now();
    for _ in 0..k {
        let ih = client.create_from_slice(f32::as_bytes(&input));
        let oh = client.empty(n * 4);
        sos_kernel::launch::<f32, R>(&client, CubeCount::Static(cubes, 1, 1), dim,
            unsafe { ArrayArg::from_raw_parts(ih, n) },
            unsafe { ArrayArg::from_raw_parts(oh.clone(), n) },
            unsafe { ArrayArg::from_raw_parts(co_h.clone(), nc) }, frames, ns);
        let _ = client.read_one(oh).unwrap();
    }
    let el = t0.elapsed().as_secs_f64() / k as f64;
    println!("fluxion GPU one-shot transfer   : {:6.3} ms   {:7.0} Msamples/s", el * 1000.0, msamp / el);
}

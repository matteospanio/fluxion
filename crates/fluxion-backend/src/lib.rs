//! `fluxion-backend` — lowering targets for the DSP graph.
//!
//! One kernel definition, all vendors: lowers the `fluxion-core` IR to CubeCL (CUDA / ROCm-HIP /
//! Metal / Vulkan / WGSL) and to a CPU-SIMD reference path, fusing SOS cascades into a single
//! dispatch to stay off the kernel-launch-latency cliff (see `PROJECT.md` §4.2, §5).
//!
//! Empty scaffold for now.

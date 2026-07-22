//! `fluxion-py` — Python bindings (PyO3 + maturin).
//!
//! A torchaudio-style eager API: build an effect [`Chain`] from filter/effect constructors, compose
//! with `|`, and apply it to a NumPy array. Plus the differentiable primitives ([`sos_forward`] /
//! [`sos_backward`]) that the Python-side `torch.autograd.Function` adapter wraps so gradients flow
//! through fluxion's analytic VJPs.

// PyO3 0.22's `#[pymethods]`/`#[pyfunction]` macros expand to `unsafe` calls in safe fns, which the
// edition-2024 `unsafe_op_in_unsafe_fn` lint flags in the generated code; they also expand a
// same-type `PyErr` conversion that trips `clippy::useless_conversion`. Both are macro artifacts —
// the macros are sound.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::useless_conversion,
    clippy::type_complexity
)]

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use fluxion_core::{Graph, Op, OpKind, Signal};
use fluxion_io::checkpoint::{
    self, FlamoBiquadType, ImportOptions, Kind as CkptKind, StateDict, SvfType,
    Tensor as CkptTensor,
};
use fluxion_ops::{Biquad, certify_sos, project_stable_flat, sos_filter, sos_input_grad, sos_vjp};

mod rt;
use rt::RtChain;

fn make(kind: OpKind, params: Vec<f32>) -> PyResult<Chain> {
    let op = Op::new(kind, params).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(Chain {
        graph: Graph::Op(op),
    })
}

/// Read any DLPack-capable tensor (numpy / torch / jax CPU array) or array-like as a contiguous 1-D
/// `float32` numpy array. **Zero-copy** when the input is already `float32` + C-contiguous — DLPack
/// shares the buffer and `ascontiguousarray(dtype=float32)` is then a no-op; a copy happens only to
/// satisfy the contiguous-float32 contract. Fluxion's outputs are numpy arrays, which are DLPack
/// producers, so `torch.from_dlpack(...)` / `jax.dlpack.from_dlpack(...)` consume them zero-copy too.
fn as_f32_1d<'py>(x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let py = x.py();
    let np = py.import_bound("numpy")?;
    let arr = if x.hasattr("__dlpack__")? {
        np.call_method1("from_dlpack", (x,))?
    } else {
        np.call_method1("asarray", (x,))?
    };
    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("dtype", "float32")?;
    let arr = np.call_method("ascontiguousarray", (arr,), Some(&kwargs))?;
    arr.downcast_into::<PyArray1<f32>>()
        .map_err(|e| PyValueError::new_err(format!("expected a 1-D float32-compatible array: {e}")))
}

/// Read a 1-D `(T,)` or 2-D `(C, T)` (channels-first, last axis = time) DLPack tensor / array-like as
/// `(channels, ndim)` — the input to a multichannel [`Signal`]. Same zero-copy contract as
/// [`as_f32_1d`]. Anything other than 1-D/2-D is an error.
fn as_channels(x: &Bound<'_, PyAny>) -> PyResult<(Vec<Vec<f32>>, usize)> {
    let py = x.py();
    let np = py.import_bound("numpy")?;
    let arr = if x.hasattr("__dlpack__")? {
        np.call_method1("from_dlpack", (x,))?
    } else {
        np.call_method1("asarray", (x,))?
    };
    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("dtype", "float32")?;
    let arr = np.call_method("ascontiguousarray", (arr,), Some(&kwargs))?;
    let ndim: usize = arr.getattr("ndim")?.extract()?;
    match ndim {
        1 => {
            let a = arr
                .downcast_into::<PyArray1<f32>>()
                .map_err(|e| PyValueError::new_err(format!("1-D array expected: {e}")))?;
            Ok((vec![a.readonly().as_slice()?.to_vec()], 1))
        }
        2 => {
            let a = arr
                .downcast_into::<PyArray2<f32>>()
                .map_err(|e| PyValueError::new_err(format!("2-D array expected: {e}")))?;
            let ro = a.readonly();
            let view = ro.as_array();
            Ok((view.outer_iter().map(|row| row.to_vec()).collect(), 2))
        }
        n => Err(PyValueError::new_err(format!(
            "expected a 1-D (T,) or 2-D (C, T) array, got {n}-D"
        ))),
    }
}

fn to_sos(coeffs: &[f32]) -> PyResult<Vec<Biquad>> {
    if coeffs.is_empty() || coeffs.len() % 5 != 0 {
        return Err(PyValueError::new_err(
            "coeffs length must be a positive multiple of 5 (one [b0,b1,b2,a1,a2] per section)",
        ));
    }
    Ok(coeffs
        .chunks_exact(5)
        .map(|c| Biquad {
            b0: c[0],
            b1: c[1],
            b2: c[2],
            a1: c[3],
            a2: c[4],
        })
        .collect())
}

/// A lazy effect chain — a DSP graph. Compose with `|`, apply with `.process(x, fs)`.
#[pyclass]
#[derive(Clone)]
struct Chain {
    graph: Graph,
}

#[pymethods]
impl Chain {
    /// Apply the chain at sample rate `fs`, returning a new `float32` array of the same shape.
    /// Accepts a 1-D `(T,)` signal or a 2-D `(C, T)` multichannel signal (channels-first, last axis =
    /// time) — any DLPack tensor (numpy / torch / jax CPU) or array-like. For a *batch* of independent
    /// mono signals, iterate rows or use per-row `process` (parallel/cross-channel ops treat a 2-D
    /// input as one multichannel signal).
    fn process<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        fs: u32,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (channels, ndim) = as_channels(x)?;
        let out = fluxion_backend::process(&self.graph, &Signal::new(fs, channels));
        if ndim == 1 {
            let ch = out.channels.into_iter().next().unwrap_or_default();
            Ok(ch.into_pyarray_bound(py).into_any())
        } else {
            let arr = PyArray2::from_vec2_bound(py, &out.channels)
                .map_err(|e| PyValueError::new_err(format!("output channels are ragged: {e}")))?;
            Ok(arr.into_any())
        }
    }

    /// Apply the chain to a **batch** of independent mono signals: a 2-D `(B, T)` array (each row is
    /// one signal) at sample rate `fs`, returning a new `(B, T)` `float32` array. Every row is
    /// filtered independently — the result is identical to calling [`process`](Self::process) on each
    /// row on its own, but a pure-filter chain over equal-length rows is routed through the batched
    /// SIMD kernel (the IIR recurrence vectorizes *across the batch*), so this is the fast path for
    /// many equal-length mono clips (data augmentation, training minibatches). Same zero-copy DLPack
    /// input contract as [`process`]. This is the CPU batch path; the GPU variant is
    /// [`sos_filter_batch_gpu`], available only in the CUDA-built wheel.
    fn process_batch<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        fs: u32,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let (rows, ndim) = as_channels(x)?;
        if ndim != 2 {
            return Err(PyValueError::new_err(
                "process_batch expects a 2-D (B, T) array (each row is one mono signal)",
            ));
        }
        let batch: Vec<Signal> = rows.into_iter().map(|r| Signal::new(fs, vec![r])).collect();
        let out = fluxion_backend::process_batch(&self.graph, &batch);
        // Each output signal is one row: take its (mono) first channel, matching per-row `process`.
        let out_rows: Vec<Vec<f32>> = out
            .into_iter()
            .map(|s| s.channels.into_iter().next().unwrap_or_default())
            .collect();
        PyArray2::from_vec2_bound(py, &out_rows)
            .map_err(|e| PyValueError::new_err(format!("batch output rows are ragged: {e}")))
    }

    /// The designed SOS coefficients as a flat `[b0,b1,b2,a1,a2]·n_sections` `float32` array for a
    /// **pure-filter** chain at `fs` (used to seed a trainable `fluxion.torch.SosModule`). Errors if
    /// the chain isn't a single cascade (contains gain / delay / reverb / a parallel branch / …).
    fn sos_coeffs<'py>(&self, py: Python<'py>, fs: u32) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let sos = fluxion_backend::graph_to_sos(&self.graph, fs).ok_or_else(|| {
            PyValueError::new_err(
                "chain is not a single filter cascade (has gain/delay/parallel/…)",
            )
        })?;
        let flat: Vec<f32> = sos
            .iter()
            .flat_map(|b| [b.b0, b.b1, b.b2, b.a1, b.a2])
            .collect();
        Ok(flat.into_pyarray_bound(py))
    }

    /// `self | other` — run `self`, then feed its output to `other` (series composition).
    fn __or__(&self, other: &Chain) -> Chain {
        Chain {
            graph: self.graph.clone() | other.graph.clone(),
        }
    }

    /// `self + other` — run both on the same input and sum (parallel composition).
    fn __add__(&self, other: &Chain) -> Chain {
        Chain {
            graph: self.graph.clone() + other.graph.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!("Chain({:?})", self.graph)
    }
}

// --- effect/filter constructors (torchaudio-style) -------------------------------------------

#[pyfunction]
fn lowpass(cutoff: f32, order: u32) -> PyResult<Chain> {
    make(OpKind::Lowpass, vec![cutoff, order as f32])
}
#[pyfunction]
fn highpass(cutoff: f32, order: u32) -> PyResult<Chain> {
    make(OpKind::Highpass, vec![cutoff, order as f32])
}
#[pyfunction]
fn cheby1_lowpass(cutoff: f32, order: u32, ripple_db: f32) -> PyResult<Chain> {
    make(OpKind::Cheby1Lowpass, vec![cutoff, order as f32, ripple_db])
}
#[pyfunction]
fn cheby1_highpass(cutoff: f32, order: u32, ripple_db: f32) -> PyResult<Chain> {
    make(
        OpKind::Cheby1Highpass,
        vec![cutoff, order as f32, ripple_db],
    )
}
#[pyfunction]
fn cheby2_lowpass(cutoff: f32, order: u32, atten_db: f32) -> PyResult<Chain> {
    make(OpKind::Cheby2Lowpass, vec![cutoff, order as f32, atten_db])
}
#[pyfunction]
fn cheby2_highpass(cutoff: f32, order: u32, atten_db: f32) -> PyResult<Chain> {
    make(OpKind::Cheby2Highpass, vec![cutoff, order as f32, atten_db])
}
#[pyfunction]
fn reverb(room: f32, damping: f32, mix: f32) -> PyResult<Chain> {
    make(OpKind::Reverb, vec![room, damping, mix])
}
#[pyfunction]
fn peaking(frequency: f32, gain: f32, q: f32) -> PyResult<Chain> {
    make(OpKind::Peaking, vec![frequency, gain, q])
}
#[pyfunction]
fn low_shelf(cutoff: f32, gain: f32, q: f32) -> PyResult<Chain> {
    make(OpKind::LowShelf, vec![cutoff, gain, q])
}
#[pyfunction]
fn high_shelf(cutoff: f32, gain: f32, q: f32) -> PyResult<Chain> {
    make(OpKind::HighShelf, vec![cutoff, gain, q])
}
#[pyfunction]
fn notch(frequency: f32, q: f32) -> PyResult<Chain> {
    make(OpKind::Notch, vec![frequency, q])
}
#[pyfunction]
fn bandpass(frequency: f32, q: f32) -> PyResult<Chain> {
    make(OpKind::Bandpass, vec![frequency, q])
}
#[pyfunction]
fn allpass(frequency: f32, q: f32) -> PyResult<Chain> {
    make(OpKind::Allpass, vec![frequency, q])
}
#[pyfunction]
fn gain(value: f32) -> PyResult<Chain> {
    make(OpKind::Gain, vec![value])
}
#[pyfunction]
fn normalize(peak: f32) -> PyResult<Chain> {
    make(OpKind::Normalize, vec![peak])
}
#[pyfunction]
fn delay(seconds: f32, mix: f32) -> PyResult<Chain> {
    make(OpKind::Delay, vec![seconds, mix])
}
#[pyfunction]
fn echo(seconds: f32, feedback: f32, wet: f32) -> PyResult<Chain> {
    make(OpKind::Echo, vec![seconds, feedback, wet])
}
#[pyfunction]
fn fir(taps: Vec<f32>) -> PyResult<Chain> {
    make(OpKind::Fir, taps) // a trained/frozen FIR: y[n] = Σ_k taps[k]·x[n-k]
}

// --- differentiable SOS primitives (wrapped by the Python autograd adapter) -------------------

/// Forward pass of an SOS cascade. `coeffs` is a flat `[b0,b1,b2,a1,a2]·n_sections` array.
#[pyfunction]
fn sos_forward<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    coeffs: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let (x, coeffs) = (as_f32_1d(x)?, as_f32_1d(coeffs)?);
    let (x, coeffs) = (x.readonly(), coeffs.readonly());
    let sos = to_sos(coeffs.as_slice()?)?;
    Ok(sos_filter(x.as_slice()?, &sos).into_pyarray_bound(py))
}

/// Analytic backward pass: returns `(grad_x, grad_coeffs)` for `grad_out = ∂L/∂y`. `grad_coeffs` is
/// flat `[b0,b1,b2,a1,a2]·n_sections`.
#[pyfunction]
fn sos_backward<'py>(
    py: Python<'py>,
    grad_out: &Bound<'py, PyAny>,
    x: &Bound<'py, PyAny>,
    coeffs: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>)> {
    let (grad_out, x, coeffs) = (as_f32_1d(grad_out)?, as_f32_1d(x)?, as_f32_1d(coeffs)?);
    let (grad_out, x, coeffs) = (grad_out.readonly(), x.readonly(), coeffs.readonly());
    let sos = to_sos(coeffs.as_slice()?)?;
    let g = grad_out.as_slice()?;
    let grad_x = sos_input_grad(g, &sos);
    let (_, grad_coeffs) = sos_vjp(x.as_slice()?, &sos, g);
    let gc: Vec<f32> = grad_coeffs
        .iter()
        .flat_map(|b| [b.b0, b.b1, b.b2, b.a1, b.a2])
        .collect();
    Ok((grad_x.into_pyarray_bound(py), gc.into_pyarray_bound(py)))
}

/// Filter a flat batch of `len(x) / frames` equal-length rows through an SOS cascade on the GPU
/// (CUDA). `coeffs` is flat `[b0,b1,b2,a1,a2]·n_sections`; returns the flat filtered batch. Available
/// only in the CUDA-built ("GPU") wheel — check [`__cuda__`]. The kernel is bit-accurate vs the CPU
/// path; a one-shot call is transfer-bound, so it pays off on resident/reused data.
#[cfg(feature = "cuda")]
#[pyfunction]
fn sos_filter_batch_gpu<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    frames: usize,
    coeffs: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let (x, coeffs) = (as_f32_1d(x)?, as_f32_1d(coeffs)?);
    let (x, coeffs) = (x.readonly(), coeffs.readonly());
    let sos = to_sos(coeffs.as_slice()?)?;
    if frames == 0 || x.as_slice()?.len() % frames != 0 {
        return Err(PyValueError::new_err(
            "len(x) must be a positive multiple of frames",
        ));
    }
    let out = fluxion_backend::cuda::sos_filter_batch(x.as_slice()?, frames, &sos);
    Ok(out.into_pyarray_bound(py))
}

// --- checkpoint import (FLAMO / torchfx DDSP state-dicts -> certified sections) ---------------

/// Parse checkpoint-import options shared by [`import_state_dict`].
#[allow(clippy::too_many_arguments)]
fn ckpt_options(
    kind: &str,
    fs: Option<u32>,
    svf_type: &str,
    biquad_type: &str,
    eq_f_lo: f64,
    eq_f_hi: f64,
    eq_max_gain_db: f64,
) -> PyResult<ImportOptions> {
    let mut opts = ImportOptions { fs, ..ImportOptions::default() };
    opts.kind = CkptKind::from_name(kind)
        .ok_or_else(|| PyValueError::new_err(format!("unknown kind '{kind}'")))?;
    opts.svf_type = SvfType::from_name(svf_type)
        .ok_or_else(|| PyValueError::new_err(format!("unknown svf_type '{svf_type}'")))?;
    opts.biquad_type = FlamoBiquadType::from_name(biquad_type)
        .ok_or_else(|| PyValueError::new_err(format!("unknown biquad_type '{biquad_type}'")))?;
    opts.eq.f_lo = eq_f_lo;
    opts.eq.f_hi = eq_f_hi;
    opts.eq.max_gain_db = eq_max_gain_db;
    Ok(opts)
}

/// Convert a state-dict of named arrays (FLAMO / torchfx DDSP checkpoint tensors) into SOS
/// sections and certify them. Returns `(sections, verdict, margin, fs)` where `sections` is
/// `(n_sections, 5)` `[b0,b1,b2,a1,a2]` (`a0` normalised), `verdict` is the stability ladder
/// string (`certified-stable` / `marginally-stable` / …), and `fs` is a sample rate embedded in
/// the artifact (if any). `project_stable=True` clamps each section into the Jury stability
/// triangle before certification. The conversion math is the same Rust code the `fluxion import`
/// CLI verb runs — see `fluxion-io::checkpoint`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (tensors, kind="auto", fs=None, svf_type="general", biquad_type="lowpass",
                    eq_f_lo=40.0, eq_f_hi=16_000.0, eq_max_gain_db=18.0, project_stable=false))]
fn import_state_dict<'py>(
    py: Python<'py>,
    tensors: &Bound<'py, PyDict>,
    kind: &str,
    fs: Option<u32>,
    svf_type: &str,
    biquad_type: &str,
    eq_f_lo: f64,
    eq_f_hi: f64,
    eq_max_gain_db: f64,
    project_stable: bool,
) -> PyResult<(Bound<'py, PyArray2<f32>>, String, f32, Option<u32>)> {
    let opts = ckpt_options(kind, fs, svf_type, biquad_type, eq_f_lo, eq_f_hi, eq_max_gain_db)?;

    // Any array-like / DLPack value -> contiguous float32 n-D numpy -> (shape, flat data).
    let np = py.import_bound("numpy")?;
    let mut sd = StateDict::new();
    for (k, v) in tensors.iter() {
        let key: String = k.extract()?;
        let arr = if v.hasattr("__dlpack__")? {
            np.call_method1("from_dlpack", (&v,))?
        } else {
            np.call_method1("asarray", (&v,))?
        };
        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("dtype", "float32")?;
        let arr = np.call_method("ascontiguousarray", (arr,), Some(&kwargs))?;
        let arr = arr
            .downcast_into::<numpy::PyArrayDyn<f32>>()
            .map_err(|e| PyValueError::new_err(format!("tensor '{key}': {e}")))?;
        let ro = arr.readonly();
        sd.insert(
            key,
            CkptTensor { shape: ro.shape().to_vec(), data: ro.as_slice()?.to_vec() },
        );
    }

    let imported = checkpoint::sections_from_state_dict(&sd, &opts)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut coeffs: Vec<f32> = imported.sections.iter().flatten().copied().collect();
    if project_stable {
        project_stable_flat(&mut coeffs, 1e-3);
    }
    let sos = to_sos(&coeffs)?;
    let cert = certify_sos(&sos);

    let n = coeffs.len() / 5;
    let arr = numpy::PyArray2::from_vec2_bound(
        py,
        &coeffs.chunks_exact(5).map(|c| c.to_vec()).collect::<Vec<_>>(),
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    debug_assert_eq!(arr.shape(), [n, 5]);
    Ok((arr, cert.verdict.to_string(), cert.margin, imported.fs))
}

/// Chain `(n_sections, 5)` `[b0,b1,b2,a1,a2]` sections as raw `biquad` graph ops, certify at
/// `fs`, and write a standard `.fxg` graph (the same artifact `fluxion compile`/`import` write —
/// it splices into any CLI pipeline and hot-swaps). Refuses a non-shippable certificate unless
/// `force=True`. Returns `(verdict, margin)`.
#[pyfunction]
#[pyo3(signature = (path, sections, fs=48_000, force=false))]
fn save_biquad_fxg(
    py: Python<'_>,
    path: &str,
    sections: &Bound<'_, PyAny>,
    fs: u32,
    force: bool,
) -> PyResult<(String, f32)> {
    let np = py.import_bound("numpy")?;
    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("dtype", "float32")?;
    let arr = np.call_method("ascontiguousarray", (sections,), Some(&kwargs))?;
    let arr = arr
        .downcast_into::<numpy::PyArrayDyn<f32>>()
        .map_err(|e| PyValueError::new_err(format!("sections: {e}")))?;
    let ro = arr.readonly();
    if ro.shape().len() != 2 || ro.shape()[1] != 5 || ro.shape()[0] == 0 {
        return Err(PyValueError::new_err(format!(
            "sections must be a non-empty (n_sections, 5) array, got {:?}",
            ro.shape()
        )));
    }
    let flat = ro.as_slice()?;

    let mut nodes = flat
        .chunks_exact(5)
        .map(|c| Graph::op(OpKind::Biquad, [c[0], c[1], c[2], c[3], c[4]]));
    let first = nodes.next().expect("checked non-empty");
    let graph = nodes.fold(first, |acc, n| acc | n);

    let cert = fluxion_backend::certify_graph(&graph, fs);
    if !cert.verdict.is_shippable() && !force {
        return Err(PyValueError::new_err(format!(
            "refusing to write a {} graph (margin {:.2e}); project or pass force=True",
            cert.verdict, cert.margin
        )));
    }
    fluxion_core::fxg::save(&graph, path)
        .map_err(|e| PyValueError::new_err(format!("writing '{path}': {e}")))?;
    Ok((cert.verdict.to_string(), cert.margin))
}

#[pymodule]
fn _fluxion(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Chain>()?;
    m.add_class::<RtChain>()?;
    // True in the CUDA-built ("GPU") wheel, False in the default ("CPU") wheel.
    m.add("__cuda__", cfg!(feature = "cuda"))?;
    #[cfg(feature = "cuda")]
    m.add_function(wrap_pyfunction!(sos_filter_batch_gpu, m)?)?;
    macro_rules! add {
        ($($f:ident),* $(,)?) => { $( m.add_function(wrap_pyfunction!($f, m)?)?; )* };
    }
    add!(
        lowpass,
        highpass,
        cheby1_lowpass,
        cheby1_highpass,
        cheby2_lowpass,
        cheby2_highpass,
        reverb,
        peaking,
        low_shelf,
        high_shelf,
        notch,
        bandpass,
        allpass,
        gain,
        normalize,
        delay,
        echo,
        fir,
        sos_forward,
        sos_backward,
        import_state_dict,
        save_biquad_fxg,
    );
    Ok(())
}

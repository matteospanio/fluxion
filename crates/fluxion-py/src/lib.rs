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

/// A lazy effect chain — a DSP graph. Compose with `|` and `+`, apply with `chain(x, fs)`.
///
/// `subclass` so the generated per-op classes (`fluxion.filter.Lowpass`, …) can inherit it: they
/// are one `__new__` each, forwarding to this class's registry-driven constructor.
#[pyclass(subclass)]
#[derive(Clone)]
struct Chain {
    graph: Graph,
}

#[pymethods]
impl Chain {
    /// `Chain(name, *params)` — one op, by its registry name.
    ///
    /// This is the single constructor every generated op class goes through, which is why there is
    /// no per-op entry point in this module any more: `fluxion.filter.Lowpass(800, 4)` is
    /// `Chain("lowpass", 800.0, 4.0)` with a nicer name and a docstring. Missing parameters take
    /// their registry defaults; `Chain()` is the pass-through.
    #[new]
    #[pyo3(signature = (name=None, *params))]
    fn new(name: Option<&str>, params: Vec<f32>) -> PyResult<Chain> {
        let Some(name) = name else {
            return Ok(Chain { graph: Graph::Id });
        };
        let kind = OpKind::from_name(name).ok_or_else(|| {
            let help = fluxion_core::suggest::closest(name, OpKind::all().iter().map(|k| k.name()))
                .map(|s| format!(" — did you mean '{s}'?"))
                .unwrap_or_default();
            PyValueError::new_err(format!("unknown op '{name}'{help}"))
        })?;
        // Fill trailing parameters from the registry defaults, exactly as the chain text does.
        let mut values = kind.defaults();
        if kind.is_variadic() && !params.is_empty() {
            values = params;
        } else {
            if params.len() > values.len() {
                return Err(PyValueError::new_err(format!(
                    "op '{name}' takes {} parameter(s), got {}",
                    values.len(),
                    params.len()
                )));
            }
            values[..params.len()].copy_from_slice(&params);
        }
        make(kind, values)
    }

    /// `chain(x, fs)` — apply the chain, so a chain is simply callable. Same contract as
    /// [`process`](Self::process).
    #[pyo3(signature = (x, fs, sides=None))]
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        fs: u32,
        sides: Option<Vec<Bound<'py, PyAny>>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.process(py, x, fs, sides)
    }

    /// Apply the chain and also return what its observer taps saw (ROADMAP A1).
    ///
    /// Returns `(audio, readings)`. The audio is bit-identical to [`process`](Self::process)'s — a
    /// tap reads and never writes — so an analyser can live in the chain that renders. Each reading
    /// is a dict: `label`, `kind`, and either `bin_hz` + `magnitude` for a spectrum or `peak_db`,
    /// `rms_db` and `short_term_lufs` for a meter. They come back in the order the chain reaches
    /// them.
    #[pyo3(signature = (x, fs, sides=None))]
    fn taps<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        fs: u32,
        sides: Option<Vec<Bound<'py, PyAny>>>,
    ) -> PyResult<(Bound<'py, PyAny>, Vec<Py<PyDict>>)> {
        let (channels, ndim) = as_channels(x)?;
        let sides: Vec<Signal> = sides
            .unwrap_or_default()
            .iter()
            .map(|s| as_channels(s).map(|(c, _)| Signal::new(fs, c)))
            .collect::<PyResult<_>>()?;
        let side_refs: Vec<&Signal> = sides.iter().collect();
        let (out, readings) = fluxion_backend::process_taps_with(
            &self.graph,
            &Signal::new(fs, channels),
            &side_refs,
        );

        let audio = if ndim == 1 {
            let ch = out.channels.into_iter().next().unwrap_or_default();
            ch.into_pyarray_bound(py).into_any()
        } else {
            PyArray2::from_vec2_bound(py, &out.channels)
                .map_err(|e| PyValueError::new_err(format!("output channels are ragged: {e}")))?
                .into_any()
        };

        let mut out_readings = Vec::with_capacity(readings.len());
        for reading in readings {
            let d = PyDict::new_bound(py);
            d.set_item("label", reading.label)?;
            match reading.data {
                fluxion_core::TapData::Spectrum { bin_hz, magnitude } => {
                    d.set_item("kind", "spectrum")?;
                    d.set_item("bin_hz", bin_hz)?;
                    d.set_item("magnitude", magnitude.into_pyarray_bound(py))?;
                }
                fluxion_core::TapData::Meter {
                    peak_db,
                    rms_db,
                    short_term_lufs,
                } => {
                    d.set_item("kind", "meter")?;
                    d.set_item("peak_db", peak_db)?;
                    d.set_item("rms_db", rms_db)?;
                    d.set_item("short_term_lufs", short_term_lufs)?;
                }
            }
            out_readings.push(d.unbind());
        }
        Ok((audio, out_readings))
    }

    /// The canonical chain text — the same string the CLI's `--chain`, `fluxion.chain()`, C's
    /// `fx_chain_from_text` and the browser accept, so `fluxion.chain(str(c)) == c`.
    fn __str__(&self) -> String {
        self.graph.to_string()
    }

    /// Apply the chain at sample rate `fs`, returning a new `float32` array of the same shape.
    /// Accepts a 1-D `(T,)` signal or a 2-D `(C, T)` multichannel signal (channels-first, last axis =
    /// time) — any DLPack tensor (numpy / torch / jax CPU) or array-like. For a *batch* of independent
    /// mono signals, iterate rows or use per-row `process` (parallel/cross-channel ops treat a 2-D
    /// input as one multichannel signal).
    ///
    /// `sides` supplies the extra signals a chain's `side(0)`, `side(1)`, … read (ROADMAP S1) — a
    /// key for a gate, a source to duck against. They are taken to be at `fs` like the input, and
    /// are read as silence past their end.
    #[pyo3(signature = (x, fs, sides=None))]
    fn process<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        fs: u32,
        sides: Option<Vec<Bound<'py, PyAny>>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (channels, ndim) = as_channels(x)?;
        let sides: Vec<Signal> = sides
            .unwrap_or_default()
            .iter()
            .map(|s| as_channels(s).map(|(c, _)| Signal::new(fs, c)))
            .collect::<PyResult<_>>()?;
        let side_refs: Vec<&Signal> = sides.iter().collect();
        let out = fluxion_backend::process_with(&self.graph, &Signal::new(fs, channels), &side_refs);
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

    /// Serialize this chain to a `.fxg` graph artifact at `path` — the interchange format the C
    /// ABI (`fx_graph_load_fxg`) and the CLI load. Ops are stored **un-designed** (coefficients are
    /// computed when the graph is lowered at a sample rate), so one `.fxg` is sample-rate-agnostic
    /// and carries the whole chain — biquads, FIR taps, gain, delay — not just SOS sections (unlike
    /// [`save_biquad_fxg`]). Provision one `.fxg` per output channel this way, then load and stream
    /// them from a C/C++ host via `fx_graph_load_fxg` + `fx_rt_new`.
    fn save_fxg(&self, path: &str) -> PyResult<()> {
        fluxion_core::fxg::save(&self.graph, path)
            .map_err(|e| PyValueError::new_err(format!("writing '{path}': {e}")))
    }

    /// Certify this chain's stability at sample rate `fs`, returning `(verdict, margin)`: the
    /// verdict string on the stability ladder (`certified-stable` / `marginally-stable` /
    /// `indeterminate` / `not-certified` / `unstable`) and the numerical margin (`1 − spectral
    /// radius`; `NaN` if indeterminate). This is the same certificate `fx_rt_new` and
    /// `RtChain.from_chain` gate on (they refuse `unstable`) — call it in a provisioning script to
    /// fail fast per channel before shipping the `.fxg`.
    fn certify(&self, fs: u32) -> (String, f32) {
        let cert = fluxion_backend::certify_graph(&self.graph, fs);
        (cert.verdict.to_string(), cert.margin)
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

// --- the registry, and the two ways into it ---------------------------------------------------

/// Build a chain from the shared text syntax — the same string the CLI's `--chain`, C's
/// `fx_chain_from_text` and the browser's `Chain.fromText` accept.
///
/// ```python
/// fx.chain("highpass(80, 4) | gain(-3dB)")
/// ```
///
/// A syntax or name error raises `ValueError` with the caret rendering, so the message points at
/// the character that is wrong and suggests a fix where there is one.
#[pyfunction]
fn chain(text: &str) -> PyResult<Chain> {
    match fluxion_core::parse::chain(text) {
        Ok(graph) => Ok(Chain { graph }),
        Err(e) => Err(PyValueError::new_err(e.render(text))),
    }
}

/// The op registry as a list of dicts: `name`, `class`, `group`, `variadic`, `doc`, `params`.
///
/// This is what `scripts/gen_interfaces.py` generates `fluxion.filter` / `fluxion.effect` and their
/// stubs from, and what the conformance test compares those modules against — so a new op cannot
/// reach Rust without reaching Python.
#[pyfunction]
fn ops_table(py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
    // JSON has no infinity; Python does, so bounds come through as real floats here.
    let mut out = Vec::with_capacity(OpKind::all().len());
    for &kind in OpKind::all() {
        let params: Vec<Py<PyDict>> = kind
            .params()
            .iter()
            .map(|p| -> PyResult<Py<PyDict>> {
                let d = PyDict::new_bound(py);
                d.set_item("name", p.name)?;
                d.set_item("unit", p.unit.as_str())?;
                d.set_item("default", p.default)?;
                d.set_item("min", p.min)?;
                d.set_item("max", p.max)?;
                Ok(d.unbind())
            })
            .collect::<PyResult<_>>()?;

        let d = PyDict::new_bound(py);
        d.set_item("name", kind.name())?;
        d.set_item("class", kind.variant())?;
        d.set_item("group", kind.group().as_str())?;
        d.set_item("variadic", kind.is_variadic())?;
        d.set_item(
            "doc",
            kind.doc()
                .iter()
                .map(|l| l.trim())
                .collect::<Vec<_>>()
                .join(" "),
        )?;
        d.set_item("params", params)?;
        out.push(d.unbind());
    }
    Ok(out)
}

// --- audio files ------------------------------------------------------------------------------

/// Read an audio file as `(samples, fs)`: a `(channels, frames)` `float32` array and its sample
/// rate. WAV goes through hound, everything else (FLAC / MP3 / OGG / …) through Symphonia — the
/// same readers the CLI uses, so Python and the terminal decode a file identically.
///
/// This is what backs `fluxion.Wave.from_file`, and it is why the wheel needs no `soundfile`.
#[pyfunction]
fn read_audio(py: Python<'_>, path: &str) -> PyResult<(Py<PyArray2<f32>>, u32)> {
    let is_wav = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("wav"));
    let signal = if is_wav {
        fluxion_io::read_wav(path)
            .map_err(|e| PyValueError::new_err(format!("reading '{path}': {e}")))?
    } else {
        fluxion_io::decode(path)
            .map_err(|e| PyValueError::new_err(format!("decoding '{path}': {e}")))?
    };
    let fs = signal.fs;
    let arr = PyArray2::from_vec2_bound(py, &signal.channels)
        .map_err(|e| PyValueError::new_err(format!("channels are ragged: {e}")))?;
    Ok((arr.unbind(), fs))
}

/// Resample `data` from `from_fs` to `to_fs`, returning an array of the same shape with
/// `round(frames · to_fs/from_fs)` frames (ROADMAP R2).
///
/// This is what backs `fluxion.Wave.ensure_fs`, and it is the same converter the CLI's `--rate` and
/// the browser's `ensureFs` run — a project rate means one rate *and* one conversion.
#[pyfunction]
fn ensure_fs<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    from_fs: u32,
    to_fs: u32,
) -> PyResult<Bound<'py, PyAny>> {
    if from_fs == 0 || to_fs == 0 {
        return Err(PyValueError::new_err(format!(
            "sample rates must be positive (got {from_fs} -> {to_fs})"
        )));
    }
    let (channels, ndim) = as_channels(data)?;
    let out = fluxion_ops::transform::ensure_fs(Signal::new(from_fs, channels), to_fs);
    if ndim == 1 {
        let ch = out.channels.into_iter().next().unwrap_or_default();
        Ok(ch.into_pyarray_bound(py).into_any())
    } else {
        let arr = PyArray2::from_vec2_bound(py, &out.channels)
            .map_err(|e| PyValueError::new_err(format!("output channels are ragged: {e}")))?;
        Ok(arr.into_any())
    }
}

/// Write a `(channels, frames)` (or 1-D) array to a WAV file at `fs`.
///
/// `bits` is `None` for 32-bit float (lossless, the default) or 16 / 24 / 32 for dithered integer
/// PCM — the same choices as the CLI's `--bits` / `--float`.
#[pyfunction]
#[pyo3(signature = (path, data, fs, bits=None))]
fn write_audio(path: &str, data: &Bound<'_, PyAny>, fs: u32, bits: Option<u16>) -> PyResult<()> {
    let (channels, _) = as_channels(data)?;
    let encoding = match bits {
        None => fluxion_io::WavEncoding {
            bits: 32,
            float: true,
            dither: false,
        },
        Some(b @ (16 | 24 | 32)) => fluxion_io::WavEncoding {
            bits: b,
            float: false,
            dither: true,
        },
        Some(b) => {
            return Err(PyValueError::new_err(format!(
                "bits must be 16, 24 or 32 (got {b})"
            )));
        }
    };
    fluxion_io::write_wav_encoded(path, &Signal::new(fs, channels), encoding)
        .map_err(|e| PyValueError::new_err(format!("writing '{path}': {e}")))
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
    let mut opts = ImportOptions {
        fs,
        ..ImportOptions::default()
    };
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
    let opts = ckpt_options(
        kind,
        fs,
        svf_type,
        biquad_type,
        eq_f_lo,
        eq_f_hi,
        eq_max_gain_db,
    )?;

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
            CkptTensor {
                shape: ro.shape().to_vec(),
                data: ro.as_slice()?.to_vec(),
            },
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
        &coeffs
            .chunks_exact(5)
            .map(|c| c.to_vec())
            .collect::<Vec<_>>(),
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
    // No per-op entry points: every op reaches Python as a generated class over `Chain`, named
    // from `ops_table()`. That is why this list is short and cannot drift.
    add!(
        chain,
        ops_table,
        read_audio,
        write_audio,
        ensure_fs,
        sos_forward,
        sos_backward,
        import_state_dict,
        save_biquad_fxg,
    );
    Ok(())
}

//! `fluxion-py` — Python bindings (PyO3 + maturin).
//!
//! A torchaudio-style eager API: build an effect [`Chain`] from filter/effect constructors, compose
//! with `|`, and apply it to a NumPy array. Plus the differentiable primitives ([`sos_forward`] /
//! [`sos_backward`]) that the Python-side `torch.autograd.Function` adapter wraps so gradients flow
//! through fluxion's analytic VJPs.

// PyO3 0.22's `#[pymethods]`/`#[pyfunction]` macros expand to `unsafe` calls in safe fns, which the
// edition-2024 `unsafe_op_in_unsafe_fn` lint flags in the generated code. The macros are sound.
#![allow(unsafe_op_in_unsafe_fn)]

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use fluxion_core::{Graph, Op, OpKind, Signal};
use fluxion_ops::{Biquad, sos_filter, sos_input_grad, sos_vjp};

fn make(kind: OpKind, params: Vec<f32>) -> PyResult<Chain> {
    let op = Op::new(kind, params).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(Chain { graph: Graph::Op(op) })
}

fn to_sos(coeffs: &[f32]) -> PyResult<Vec<Biquad>> {
    if coeffs.is_empty() || coeffs.len() % 5 != 0 {
        return Err(PyValueError::new_err(
            "coeffs length must be a positive multiple of 5 (one [b0,b1,b2,a1,a2] per section)",
        ));
    }
    Ok(coeffs
        .chunks_exact(5)
        .map(|c| Biquad { b0: c[0], b1: c[1], b2: c[2], a1: c[3], a2: c[4] })
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
    /// Apply the chain to a 1-D `float32` array at sample rate `fs`, returning a new array.
    fn process<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f32>,
        fs: u32,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let input = x.as_slice()?.to_vec();
        let out = fluxion_backend::process(&self.graph, &Signal::new(fs, vec![input]));
        let ch = out.channels.into_iter().next().unwrap_or_default();
        Ok(ch.into_pyarray_bound(py))
    }

    /// `self | other` — run `self`, then feed its output to `other` (series composition).
    fn __or__(&self, other: &Chain) -> Chain {
        Chain { graph: self.graph.clone() | other.graph.clone() }
    }

    /// `self + other` — run both on the same input and sum (parallel composition).
    fn __add__(&self, other: &Chain) -> Chain {
        Chain { graph: self.graph.clone() + other.graph.clone() }
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
    make(OpKind::Cheby1Highpass, vec![cutoff, order as f32, ripple_db])
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

// --- differentiable SOS primitives (wrapped by the Python autograd adapter) -------------------

/// Forward pass of an SOS cascade. `coeffs` is a flat `[b0,b1,b2,a1,a2]·n_sections` array.
#[pyfunction]
fn sos_forward<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f32>,
    coeffs: PyReadonlyArray1<'py, f32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let sos = to_sos(coeffs.as_slice()?)?;
    Ok(sos_filter(x.as_slice()?, &sos).into_pyarray_bound(py))
}

/// Analytic backward pass: returns `(grad_x, grad_coeffs)` for `grad_out = ∂L/∂y`. `grad_coeffs` is
/// flat `[b0,b1,b2,a1,a2]·n_sections`.
#[pyfunction]
fn sos_backward<'py>(
    py: Python<'py>,
    grad_out: PyReadonlyArray1<'py, f32>,
    x: PyReadonlyArray1<'py, f32>,
    coeffs: PyReadonlyArray1<'py, f32>,
) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>)> {
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

#[pymodule]
fn _fluxion(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Chain>()?;
    macro_rules! add {
        ($($f:ident),* $(,)?) => { $( m.add_function(wrap_pyfunction!($f, m)?)?; )* };
    }
    add!(
        lowpass,
        highpass,
        cheby1_lowpass,
        cheby1_highpass,
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
        sos_forward,
        sos_backward,
    );
    Ok(())
}

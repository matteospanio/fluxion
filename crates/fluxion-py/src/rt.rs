//! Realtime bindings: [`RtChain`] — the streaming, stateful executor behind `fluxion.RtChain`.
//!
//! [`crate::Chain::process`] is the batch path: stateless across calls (each call starts from zero
//! filter state) and allocating, so it cannot stream a signal block-by-block. `RtChain` wraps the
//! `fluxion-rt` engine instead: a [`Chain`] is lowered once with `fluxion_backend::to_rt_graph`
//! (designing every op's coefficients at a fixed sample rate), certified on the stability ladder,
//! and pre-sized with `RtGraph::prepare` — after which `process` filters caller-provided numpy
//! blocks **in place, allocation-free, carrying state across calls**, releasing the GIL while the
//! Rust kernel runs. Streaming a signal in chunks is bit-identical to filtering it whole (the
//! `SosStream` invariant), which is what makes it usable inside an audio callback pipeline.

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1, PyUntypedArrayMethods};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;

use fluxion_backend::{Verdict, certify_graph, to_rt_graph};
use fluxion_ops::{Biquad, certify_sos};
use fluxion_rt::RtGraph;

use crate::Chain;

/// Read an `(n_sections, 5)` `[b0, b1, b2, a1, a2]` float32 array as a cascade.
fn sections_to_sos(sections: &PyReadonlyArray2<'_, f32>) -> PyResult<Vec<Biquad>> {
    let shape = sections.shape();
    if shape[0] == 0 || shape[1] != 5 {
        return Err(PyValueError::new_err(format!(
            "sections must be a non-empty (n_sections, 5) [b0,b1,b2,a1,a2] array, got {shape:?}"
        )));
    }
    let view = sections.as_array();
    Ok(view
        .outer_iter()
        .map(|r| Biquad {
            b0: r[0],
            b1: r[1],
            b2: r[2],
            a1: r[3],
            a2: r[4],
        })
        .collect())
}

/// A realtime, stateful executor for a [`Chain`]: streams a signal block-by-block.
///
/// Build one with [`RtChain::from_chain`] (lower + certify + pre-size) or
/// [`RtChain::from_sections`] (already-designed SOS coefficients, e.g. from
/// `import_state_dict`). Then call [`process`](RtChain::process) once per block: filter state is
/// carried across calls, so chunked streaming equals whole-signal filtering; the call is
/// allocation-free and releases the GIL while the Rust kernel runs.
///
/// One `RtChain` is one mono stream. For multichannel audio, build one `RtChain` per channel
/// (they share nothing). Concurrent calls on the *same* object from two Python threads are
/// rejected by the runtime borrow check (`RuntimeError`) — an `RtChain` is single-stream by
/// design, not a thread-safe pool.
#[pyclass]
pub(crate) struct RtChain {
    inner: RtGraph,
    max_block: usize,
    fs: Option<u32>,
    verdict: Verdict,
    margin: f32,
}

impl RtChain {
    /// Shared constructor tail: refuse an unstable cascade, pre-size, and record the certificate.
    fn build(
        mut inner: RtGraph,
        max_block: usize,
        fs: Option<u32>,
        verdict: Verdict,
        margin: f32,
    ) -> PyResult<Self> {
        if max_block == 0 {
            return Err(PyValueError::new_err("max_block must be >= 1"));
        }
        inner.prepare(max_block);
        Ok(Self {
            inner,
            max_block,
            fs,
            verdict,
            margin,
        })
    }
}

#[pymethods]
impl RtChain {
    /// Lower `chain` to a realtime executor at sample rate `fs`, ready for blocks up to
    /// `max_block` samples.
    ///
    /// The chain's coefficients are designed once at `fs` and certified on the stability ladder;
    /// an `unstable` verdict is refused (`ValueError`). Chains containing an op with no realtime
    /// lowering — `normalize` / `fade` / `reverse` / the modulated effects / feedback loops —
    /// are refused too (design the chain without them; `normalize` needs the whole signal's peak,
    /// so replace it with a static headroom `gain`). `prepare` happens here: `process` never
    /// allocates afterwards.
    #[staticmethod]
    #[pyo3(signature = (chain, fs, max_block = 4096))]
    fn from_chain(chain: &Chain, fs: u32, max_block: usize) -> PyResult<Self> {
        if fs == 0 {
            return Err(PyValueError::new_err("fs must be >= 1"));
        }
        let cert = certify_graph(&chain.graph, fs);
        if cert.verdict == Verdict::Unstable {
            return Err(PyValueError::new_err(format!(
                "chain is unstable at fs={fs} (margin {:.3e}); refusing to build a realtime executor",
                cert.margin
            )));
        }
        let inner = to_rt_graph(&chain.graph, fs).ok_or_else(|| {
            PyValueError::new_err(
                "chain cannot run in realtime: it contains a whole-signal or modulated op \
                 (normalize / fade / reverse / tremolo / overdrive / chorus / flanger / phaser) \
                 or a feedback loop",
            )
        })?;
        Self::build(inner, max_block, Some(fs), cert.verdict, cert.margin)
    }

    /// A realtime executor from already-designed SOS coefficients: a non-empty
    /// `(n_sections, 5)` `[b0, b1, b2, a1, a2]` float32 array (`a0` normalised) — e.g. the
    /// sections returned by `import_state_dict`, or `Chain.sos_coeffs(fs).reshape(-1, 5)`.
    ///
    /// The cascade is certified from its poles; an `unstable` verdict is refused (`ValueError`).
    /// No sample rate is involved (the coefficients are already discrete-time), so
    /// [`fs`](RtChain::fs) is `None`.
    #[staticmethod]
    #[pyo3(signature = (sections, max_block = 4096))]
    fn from_sections(sections: PyReadonlyArray2<'_, f32>, max_block: usize) -> PyResult<Self> {
        let sos = sections_to_sos(&sections)?;
        let cert = certify_sos(&sos);
        if cert.verdict == Verdict::Unstable {
            return Err(PyValueError::new_err(format!(
                "sections are unstable (margin {:.3e}); refusing to build a realtime executor",
                cert.margin
            )));
        }
        Self::build(
            RtGraph::filter(sos),
            max_block,
            None,
            cert.verdict,
            cert.margin,
        )
    }

    /// Filter one block: read `input`, write `output` — both 1-D **float32, C-contiguous** numpy
    /// arrays of the **same length ≤ `max_block`**, and **distinct** (in-place aliasing is
    /// rejected by the borrow check). Filter state is carried across calls; the call is
    /// allocation-free and releases the GIL while the Rust kernel runs, so a Python DSP thread
    /// doesn't block the rest of the process.
    ///
    /// The dtype is strict (a float64 array raises `TypeError` instead of silently copying):
    /// allocate your block buffers once with `np.empty(n, np.float32)` and reuse them.
    fn process(
        &mut self,
        py: Python<'_>,
        input: PyReadonlyArray1<'_, f32>,
        mut output: PyReadwriteArray1<'_, f32>,
    ) -> PyResult<()> {
        let x = input.as_slice()?;
        let y = output.as_slice_mut()?;
        if x.len() != y.len() {
            return Err(PyValueError::new_err(format!(
                "input ({}) and output ({}) must be the same length",
                x.len(),
                y.len()
            )));
        }
        if x.len() > self.max_block {
            return Err(PyValueError::new_err(format!(
                "block of {} samples exceeds max_block = {}",
                x.len(),
                self.max_block
            )));
        }
        let inner = &mut self.inner;
        py.allow_threads(|| inner.process(x, y));
        Ok(())
    }

    /// Swap the `node`-th filter's coefficients (depth-first order over the chain's filter ops,
    /// see [`filter_count`](RtChain::filter_count)) to a new `(n_sections, 5)` cascade,
    /// equal-power crossfaded over `fade_samples` — click-free live automation. `fade_samples=0`
    /// swaps with a 1-sample fade (effectively immediate).
    ///
    /// The incoming cascade is certified first; `unstable` sections are refused (`ValueError`).
    /// An out-of-range `node` raises `IndexError`.
    ///
    /// This is a **control-plane** call: it converts and certifies the sections — allocating,
    /// and holding the GIL — before handing them to the engine's (allocation-free) crossfade
    /// swap. Call it from a control thread between blocks, not from a hard-realtime audio
    /// callback; [`process`](RtChain::process) is the only call designed for that path.
    #[pyo3(signature = (node, sections, fade_samples = 0))]
    fn set_coeffs(
        &mut self,
        node: usize,
        sections: PyReadonlyArray2<'_, f32>,
        fade_samples: u32,
    ) -> PyResult<()> {
        let sos = sections_to_sos(&sections)?;
        let cert = certify_sos(&sos);
        if cert.verdict == Verdict::Unstable {
            return Err(PyValueError::new_err(format!(
                "incoming sections are unstable (margin {:.3e}); refusing the swap",
                cert.margin
            )));
        }
        if !self.inner.set_coeffs(node, &sos, fade_samples) {
            return Err(PyIndexError::new_err(format!(
                "filter node {node} out of range (filter_count = {})",
                self.inner.filter_count()
            )));
        }
        Ok(())
    }

    /// Reset all filter state and delay lines to silence (as if freshly built). Call on seek or
    /// source switch so the tail of the previous audio doesn't bleed into the new one.
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Number of addressable filter (SOS) nodes, depth-first — the valid `node` range for
    /// [`set_coeffs`](RtChain::set_coeffs).
    #[getter]
    fn filter_count(&self) -> usize {
        self.inner.filter_count()
    }

    /// The largest block `process` accepts (set at construction).
    #[getter]
    fn max_block(&self) -> usize {
        self.max_block
    }

    /// The sample rate the chain was designed at, or `None` for
    /// [`from_sections`](RtChain::from_sections) (already-discrete coefficients).
    #[getter]
    fn fs(&self) -> Option<u32> {
        self.fs
    }

    /// The construction-time stability verdict: `"certified-stable"`, `"marginally-stable"`,
    /// `"indeterminate"` or `"not-certified"` (`"unstable"` never constructs).
    #[getter]
    fn verdict(&self) -> String {
        self.verdict.to_string()
    }

    /// The stability margin (`1 − spectral radius`): positive inside the stable region, `NaN` if
    /// indeterminate.
    #[getter]
    fn margin(&self) -> f32 {
        self.margin
    }

    fn __repr__(&self) -> String {
        format!(
            "RtChain(filters={}, max_block={}, fs={:?}, verdict={})",
            self.inner.filter_count(),
            self.max_block,
            self.fs,
            self.verdict
        )
    }
}

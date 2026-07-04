//! Allocation-free streaming SOS cascade (plan task G3, core).
//!
//! Runs a *frozen* second-order-section cascade block-by-block with persistent per-section state, so
//! streaming a signal in chunks is bit-identical to filtering it whole ([`fluxion_ops::sos_filter`]).
//! No allocation in [`SosStream::process_block`] — the only state is the pre-sized `state` vector,
//! filled at construction. This is the inference kernel the realtime executor drives; freezing an
//! arbitrary `fluxion_core::Graph` to a cascade (G2) and the SIMD/general-graph executor are
//! later steps.

use std::f32::consts::FRAC_PI_2;

use fluxion_ops::Biquad;

/// A streaming SOS cascade: the sections plus each section's Direct-Form-II-Transposed state.
///
/// Supports a click-free coefficient swap ([`set_coeffs`](Self::set_coeffs), plan task G9): the
/// incoming cascade runs alongside the current one and their outputs are equal-power blended over a
/// fade window, then the new one takes over. Design the new coefficients off the audio thread; the
/// swap itself is allocation-free (after [`prepare`](Self::prepare)) when the section count is
/// unchanged — the common case for automating a filter's cutoff/Q.
#[derive(Debug, Clone)]
pub struct SosStream {
    sos: Vec<Biquad>,
    state: Vec<[f32; 2]>, // (s1, s2) per section
    // Crossfade-to-new-coefficients state (idle when `fade_left == 0`).
    next_sos: Vec<Biquad>,
    next_state: Vec<[f32; 2]>,
    fade_left: u32,
    fade_len: u32,
    scratch: Vec<f32>, // the incoming branch's output during a fade (sized by `prepare`)
}

impl SosStream {
    /// Build a stream from a frozen cascade, all state zeroed.
    pub fn new(sos: Vec<Biquad>) -> Self {
        let state = vec![[0.0f32; 2]; sos.len()];
        Self {
            sos,
            state,
            next_sos: Vec::new(),
            next_state: Vec::new(),
            fade_left: 0,
            fade_len: 0,
            scratch: Vec::new(),
        }
    }

    /// Pre-size the crossfade scratch for blocks up to `max_block`, and the incoming-coefficient
    /// storage for the current section count — so a same-order [`set_coeffs`](Self::set_coeffs) is
    /// allocation-free. Call before going realtime if you will swap coefficients live.
    pub fn prepare(&mut self, max_block: usize) {
        self.scratch.resize(max_block, 0.0);
        if self.next_sos.len() != self.sos.len() {
            self.next_sos = self.sos.clone();
            self.next_state = vec![[0.0f32; 2]; self.sos.len()];
        }
    }

    /// Swap to `new_sos` with an equal-power crossfade over `fade_samples` (a design-stage swap;
    /// design happens off the audio thread). Allocation-free when `new_sos.len()` equals the current
    /// section count (after [`prepare`](Self::prepare)); a different order reallocates the incoming
    /// storage. A swap requested mid-fade restarts the fade from the current output.
    pub fn set_coeffs(&mut self, new_sos: &[Biquad], fade_samples: u32) {
        if self.next_sos.len() == new_sos.len() {
            self.next_sos.copy_from_slice(new_sos);
        } else {
            self.next_sos = new_sos.to_vec();
            self.next_state = vec![[0.0f32; 2]; new_sos.len()];
        }
        self.next_state.iter_mut().for_each(|s| *s = [0.0; 2]);
        self.fade_len = fade_samples.max(1);
        self.fade_left = self.fade_len;
    }

    /// The Direct-Form-II-Transposed cascade inner loop over one block, carrying `state`.
    fn run(sos: &[Biquad], state: &mut [[f32; 2]], input: &[f32], output: &mut [f32]) {
        output.copy_from_slice(input);
        // Fused cascade: sample-outer, section-inner (Direct Form II Transposed).
        // The K independent per-section dependency chains pipeline in the
        // out-of-order core instead of paying each chain's latency per pass, and
        // the block is read once instead of K times. Per-(section, sample)
        // arithmetic is unchanged, so results are bit-identical to the
        // section-outer formulation (and to `fluxion_ops::sos_filter`).
        for y in output.iter_mut() {
            let mut v = *y;
            for (bq, st) in sos.iter().zip(state.iter_mut()) {
                let out = bq.b0 * v + st[0];
                st[0] = bq.b1 * v - bq.a1 * out + st[1];
                st[1] = bq.b2 * v - bq.a2 * out;
                v = out;
            }
            *y = v;
        }
    }

    /// Build a stream from frozen sections `[b0, b1, b2, a1, a2]` (e.g. `FrozenSos::sections` from
    /// `fluxion-backend::freeze`).
    pub fn from_sections(sections: &[[f32; 5]]) -> Self {
        let sos = sections
            .iter()
            .map(|c| Biquad {
                b0: c[0],
                b1: c[1],
                b2: c[2],
                a1: c[3],
                a2: c[4],
            })
            .collect();
        Self::new(sos)
    }

    /// Reset all section state to zero and cancel any in-progress crossfade.
    pub fn reset(&mut self) {
        self.state.iter_mut().for_each(|s| *s = [0.0; 2]);
        self.next_state.iter_mut().for_each(|s| *s = [0.0; 2]);
        self.fade_left = 0;
    }

    /// Filter one block into `output`, carrying state across calls. `input` and `output` must be the
    /// same length. Allocation-free. During a [`set_coeffs`](Self::set_coeffs) crossfade the current
    /// and incoming cascades are equal-power blended.
    pub fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), output.len(), "block in/out length mismatch");
        let n = input.len();

        // Fast path: no crossfade (also the fallback if `prepare` wasn't called to size scratch).
        if self.fade_left == 0 || self.scratch.len() < n {
            Self::run(&self.sos, &mut self.state, input, output);
            return;
        }

        // Crossfade: current → output, incoming → scratch, equal-power blend (cos²+sin² = 1).
        Self::run(&self.sos, &mut self.state, input, output);
        let scratch = &mut self.scratch[..n];
        Self::run(&self.next_sos, &mut self.next_state, input, scratch);
        let (done0, flen) = (
            (self.fade_len - self.fade_left) as f32,
            self.fade_len as f32,
        );
        for (i, (o, &s)) in output.iter_mut().zip(scratch.iter()).enumerate() {
            let theta = ((done0 + i as f32) / flen).min(1.0) * FRAC_PI_2;
            *o = theta.cos() * *o + theta.sin() * s;
        }
        self.fade_left = self.fade_left.saturating_sub(n as u32);
        if self.fade_left == 0 {
            // Adopt the incoming cascade (pointer swaps, no allocation).
            std::mem::swap(&mut self.sos, &mut self.next_sos);
            std::mem::swap(&mut self.state, &mut self.next_state);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SosStream;
    use fluxion_ops::{butterworth_highpass, butterworth_lowpass, sos_filter};

    #[test]
    fn streaming_in_chunks_matches_whole_signal() {
        let sos = butterworth_lowpass(6, 4_000.0, 48_000); // 3-section cascade
        let signal: Vec<f32> = (0..5_000)
            .map(|i| (0.05 * i as f32).sin() + 0.3 * (0.31 * i as f32).sin())
            .collect();
        let whole = sos_filter(&signal, &sos);

        // Odd block size that doesn't divide the length, to exercise the carried state.
        let mut stream = SosStream::new(sos);
        let mut streamed = Vec::with_capacity(signal.len());
        let mut out = vec![0.0f32; 333];
        for chunk in signal.chunks(333) {
            let out = &mut out[..chunk.len()];
            stream.process_block(chunk, out);
            streamed.extend_from_slice(out);
        }

        assert_eq!(streamed.len(), whole.len());
        for (a, b) in streamed.iter().zip(&whole) {
            assert!((a - b).abs() < 1e-5, "streamed {a} vs whole {b}");
        }
    }

    #[test]
    fn from_sections_equals_from_biquads() {
        let sos = butterworth_lowpass(4, 5_000.0, 48_000);
        let sections: Vec<[f32; 5]> = sos.iter().map(|b| [b.b0, b.b1, b.b2, b.a1, b.a2]).collect();
        let x: Vec<f32> = (0..256).map(|i| (0.1 * i as f32).sin()).collect();

        let mut a = super::SosStream::new(sos);
        let mut b = super::SosStream::from_sections(&sections);
        let (mut oa, mut ob) = (vec![0.0; 256], vec![0.0; 256]);
        a.process_block(&x, &mut oa);
        b.process_block(&x, &mut ob);
        assert_eq!(oa, ob);
    }

    #[test]
    fn coeff_crossfade_is_click_free_and_converges() {
        // DC input: a low-pass passes it (→1), a high-pass kills it (→0). Swapping lp→hp with a
        // crossfade must slide the output smoothly 1→0 (no click) and settle at 0.
        let lp = butterworth_lowpass(2, 2_000.0, 48_000);
        let hp = butterworth_highpass(2, 2_000.0, 48_000);
        let dc = vec![1.0f32; 4000];

        let mut s = SosStream::new(lp);
        s.prepare(200);
        let mut y = Vec::with_capacity(dc.len());
        let mut out = vec![0.0f32; 200];
        let mut swapped = false;
        for chunk in dc.chunks(200) {
            let o = &mut out[..chunk.len()];
            s.process_block(chunk, o);
            y.extend_from_slice(o);
            if !swapped && y.len() >= 2000 {
                s.set_coeffs(&hp, 1000); // swap at a block boundary, 1000-sample fade
                swapped = true;
            }
        }
        assert!(
            (y[1999] - 1.0).abs() < 0.02,
            "not settled to lp DC before swap: {}",
            y[1999]
        );
        // No click during the fade: every sample-to-sample step stays tiny.
        let jump = y[2000..3000]
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f32, f32::max);
        assert!(jump < 0.02, "click during crossfade: {jump}");
        assert!(
            y[3500..].iter().all(|&v| v.abs() < 0.02),
            "not settled to hp DC after fade"
        );
    }

    #[test]
    fn reset_clears_state() {
        let sos = butterworth_lowpass(2, 6_000.0, 48_000);
        let mut stream = SosStream::new(sos.clone());
        let x: Vec<f32> = (0..64).map(|i| (0.2 * i as f32).sin()).collect();
        let mut a = vec![0.0; 64];
        let mut b = vec![0.0; 64];
        stream.process_block(&x, &mut a);
        stream.reset();
        stream.process_block(&x, &mut b);
        assert_eq!(a, b, "reset returns the stream to its initial state");
    }
}

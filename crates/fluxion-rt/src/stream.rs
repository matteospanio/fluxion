//! Allocation-free streaming SOS cascade (plan task G3, core).
//!
//! Runs a *frozen* second-order-section cascade block-by-block with persistent per-section state, so
//! streaming a signal in chunks is bit-identical to filtering it whole ([`fluxion_ops::sos_filter`]).
//! No allocation in [`SosStream::process_block`] — the only state is the pre-sized `state` vector,
//! filled at construction. This is the inference kernel the realtime executor drives; freezing an
//! arbitrary [`Graph`](fluxion_core::Graph) to a cascade (G2) and the SIMD/general-graph executor are
//! later steps.

use fluxion_ops::Biquad;

/// A streaming SOS cascade: the sections plus each section's Direct-Form-II-Transposed state.
#[derive(Debug, Clone)]
pub struct SosStream {
    sos: Vec<Biquad>,
    state: Vec<[f32; 2]>, // (s1, s2) per section
}

impl SosStream {
    /// Build a stream from a frozen cascade, all state zeroed.
    pub fn new(sos: Vec<Biquad>) -> Self {
        let state = vec![[0.0f32; 2]; sos.len()];
        Self { sos, state }
    }

    /// Reset all section state to zero (e.g. between independent files).
    pub fn reset(&mut self) {
        self.state.iter_mut().for_each(|s| *s = [0.0; 2]);
    }

    /// Filter one block into `output`, carrying state across calls. `input` and `output` must be the
    /// same length. Allocation-free.
    pub fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), output.len(), "block in/out length mismatch");
        output.copy_from_slice(input);
        for (bq, st) in self.sos.iter().zip(self.state.iter_mut()) {
            let (mut s1, mut s2) = (st[0], st[1]);
            for y in output.iter_mut() {
                let x = *y;
                let out = bq.b0 * x + s1; // Direct Form II Transposed
                s1 = bq.b1 * x - bq.a1 * out + s2;
                s2 = bq.b2 * x - bq.a2 * out;
                *y = out;
            }
            *st = [s1, s2];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SosStream;
    use fluxion_ops::{butterworth_lowpass, sos_filter};

    #[test]
    fn streaming_in_chunks_matches_whole_signal() {
        let sos = butterworth_lowpass(6, 4_000.0, 48_000); // 3-section cascade
        let signal: Vec<f32> = (0..5_000).map(|i| (0.05 * i as f32).sin() + 0.3 * (0.31 * i as f32).sin()).collect();
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

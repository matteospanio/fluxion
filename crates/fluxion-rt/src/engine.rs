//! Integrated real-time block executor (plan tasks G3 + G4).
//!
//! Ties the pieces together: a frozen SOS cascade ([`SosStream`]) plus click-free gain automation
//! ([`SmoothedValue`]), fed parameter [`Command`]s from another thread over the lock-free
//! [`ring`](crate::ring) and applied at block boundaries. [`RtEngine::process_block`] allocates
//! nothing, takes no locks, and never panics on the audio thread — it is the callback body the CPAL
//! backend (G5) will drive.

use fluxion_ops::Biquad;

use crate::param::SmoothedValue;
use crate::ring::{Consumer, Producer, channel};
use crate::stream::SosStream;

/// A parameter-change command for the audio thread. `Copy` and small, so it rides the SPSC ring.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Command {
    /// Ramp the output gain to `target` (linear), over `ramp_samples` samples (`0` = immediate).
    SetGain { target: f32, ramp_samples: u32 },
}

/// A real-time block processor: cascade filter → smoothed output gain, with lock-free parameter
/// automation. Build it with [`RtEngine::new`], keep the returned [`Producer`] on the control thread.
pub struct RtEngine {
    stream: SosStream,
    gain: SmoothedValue,
    rx: Consumer<Command>,
}

impl RtEngine {
    /// Build an engine from a frozen cascade and initial gain. Returns the engine (for the audio
    /// thread) and the [`Producer`] half for sending [`Command`]s from a control thread.
    /// `queue_capacity` bounds the number of in-flight commands (rounded up to a power of two).
    pub fn new(sos: Vec<Biquad>, gain: f32, queue_capacity: usize) -> (Self, Producer<Command>) {
        let (tx, rx) = channel::<Command>(queue_capacity);
        let engine = Self {
            stream: SosStream::new(sos),
            gain: SmoothedValue::new(gain),
            rx,
        };
        (engine, tx)
    }

    /// Process one block in place: apply any pending commands (at this block boundary), run the
    /// cascade, then apply the per-sample smoothed gain. Allocation-free, lock-free — audio-safe.
    /// `input` and `output` must be the same length.
    ///
    /// Commands queued for the same block are drained in order before audio runs, so several
    /// `SetGain`s collapse to the **last** one (its ramp starts from the current gain). Send at most
    /// one gain update per block when you want each target's ramp to be heard.
    pub fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
        while let Some(cmd) = self.rx.pop() {
            match cmd {
                Command::SetGain {
                    target,
                    ramp_samples,
                } => self.gain.set_target(target, ramp_samples),
            }
        }
        self.stream.process_block(input, output);
        for y in output.iter_mut() {
            *y *= self.gain.tick();
        }
    }

    /// Reset filter state (e.g. between independent files). The gain target is left as-is.
    pub fn reset(&mut self) {
        self.stream.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::{Command, RtEngine};
    use fluxion_ops::{butterworth_lowpass, sos_filter};

    #[test]
    fn gain_command_ramps_at_block_boundary() {
        // Identity filter (no sections) so we isolate the gain path.
        let (mut eng, mut tx) = RtEngine::new(vec![], 1.0, 8);
        let input = vec![1.0f32; 8];
        let mut out = vec![0.0f32; 8];

        tx.push(Command::SetGain {
            target: 0.0,
            ramp_samples: 4,
        })
        .unwrap();
        eng.process_block(&input, &mut out);
        // gain 1.0 → 0.0 over 4 samples, applied to a constant-1 signal, then holds at 0.
        for (got, want) in out.iter().zip(&[0.75, 0.5, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0]) {
            assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        }
    }

    #[test]
    fn command_applies_only_on_next_block() {
        let (mut eng, mut tx) = RtEngine::new(vec![], 1.0, 8);
        let input = vec![1.0f32; 4];
        let (mut a, mut b) = (vec![0.0f32; 4], vec![0.0f32; 4]);

        eng.process_block(&input, &mut a); // no command yet → gain stays 1.0
        assert_eq!(a, vec![1.0; 4]);
        tx.push(Command::SetGain {
            target: 0.0,
            ramp_samples: 0,
        })
        .unwrap();
        eng.process_block(&input, &mut b); // command drained at this boundary → immediate 0
        assert_eq!(b, vec![0.0; 4]);
    }

    #[test]
    fn filters_like_sos_filter_at_unity_gain() {
        let sos = butterworth_lowpass(4, 5_000.0, 48_000);
        let (mut eng, _tx) = RtEngine::new(sos.clone(), 1.0, 4);
        let input: Vec<f32> = (0..256).map(|i| (0.1 * i as f32).sin()).collect();
        let mut out = vec![0.0f32; 256];
        eng.process_block(&input, &mut out);
        let want = sos_filter(&input, &sos);
        for (a, b) in out.iter().zip(&want) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}

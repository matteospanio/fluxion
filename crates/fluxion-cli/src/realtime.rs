//! Realtime `play` / `record` verbs (feature `realtime`, CPAL). Off by default so the base CLI
//! pulls no platform audio libraries; the stub build returns a clear "enable the feature" error.

pub(crate) use imp::{play, record};

#[cfg(not(feature = "realtime"))]
mod imp {
    fn unavailable() -> Result<(), String> {
        Err("realtime `play`/`record` need the `realtime` feature — \
             build/install with `--features realtime`"
            .into())
    }

    pub(crate) fn play(_: &[String], _: Option<u32>) -> Result<(), String> {
        unavailable()
    }

    pub(crate) fn record(_: &[String], _: f32, _: fluxion_io::WavEncoding) -> Result<(), String> {
        unavailable()
    }
}

#[cfg(feature = "realtime")]
mod imp {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};
    use std::time::{Duration, Instant};

    use fluxion::{Signal, process, transform};
    use fluxion_io::WavEncoding;
    use fluxion_rt::channel;
    use fluxion_rt::cpal_backend::{
        default_input_config, default_output_fs, run_input, run_output,
    };

    use crate::chain::parse_chain;
    use crate::verbs::{load_input, write_output};

    /// `fluxion play <in.wav> [effect...]` — process a file through the chain and play it live.
    pub(crate) fn play(args: &[String], fs: Option<u32>) -> Result<(), String> {
        let input = args
            .first()
            .ok_or("usage: fluxion play <in.wav> [effect...]")?;
        let mut signal = load_input(input)?;
        if let Some(fs) = fs {
            signal.fs = fs;
        }
        let graph = parse_chain(&args[1..])?;
        play_signal(&process(&graph, &signal))
    }

    fn play_signal(signal: &Signal) -> Result<(), String> {
        if signal.channels.iter().all(|c| c.is_empty()) {
            return Ok(());
        }
        let device_fs = default_output_fs().map_err(|e| e.to_string())?;
        // Resample to the device rate (real windowed-sinc SRC) so playback runs at the right speed.
        let resampled = transform::resample(signal, device_fs);
        let src: Arc<Vec<Vec<f32>>> = Arc::new(resampled.channels);
        let (nframes, nsrc) = (src.iter().map(Vec::len).max().unwrap_or(0), src.len());

        let cursor = Arc::new(AtomicUsize::new(0));
        let done = Arc::new(AtomicBool::new(false));
        let (s_cb, cur_cb, done_cb) = (src.clone(), cursor.clone(), done.clone());

        let stream = run_output(move |buf, ch| {
            let mut pos = cur_cb.load(Relaxed);
            for frame in buf.chunks_mut(ch) {
                if pos >= nframes {
                    frame.iter_mut().for_each(|s| *s = 0.0);
                    done_cb.store(true, Relaxed);
                    continue;
                }
                for (c, s) in frame.iter_mut().enumerate() {
                    // Map device channel → source channel: mono replicates; extra device channels
                    // reuse the last source channel.
                    *s = s_cb[c.min(nsrc - 1)].get(pos).copied().unwrap_or(0.0);
                }
                pos += 1;
            }
            cur_cb.store(pos, Relaxed);
        })
        .map_err(|e| e.to_string())?;

        eprintln!(
            "fluxion: playing {:.1}s @ {device_fs} Hz",
            nframes as f32 / device_fs as f32
        );
        while !done.load(Relaxed) {
            std::thread::sleep(Duration::from_millis(50));
        }
        std::thread::sleep(Duration::from_millis(120)); // let the final block flush
        drop(stream);
        Ok(())
    }

    /// `fluxion record [effect...] <out.wav>` — capture `--secs` from the default input, process it,
    /// and write it. Capture crosses the audio thread through the lock-free ring.
    pub(crate) fn record(args: &[String], secs: f32, enc: WavEncoding) -> Result<(), String> {
        let out = args
            .last()
            .ok_or("usage: fluxion record [effect...] <out.wav> [--secs N]")?;
        let graph = parse_chain(&args[..args.len() - 1])?;
        let secs = secs.max(0.1);

        let (dev_fs, channels) = default_input_config().map_err(|e| e.to_string())?;
        let want = (secs * dev_fs as f32) as usize * channels;
        // Ring holds ~1 s of slack beyond the target so a slow drain never overflows.
        let (mut tx, mut rx) = channel::<f32>(want + dev_fs as usize * channels);
        let stream = run_input(move |data, _| {
            for &s in data {
                let _ = tx.push(s); // drop on overflow (shouldn't happen with the slack above)
            }
        })
        .map_err(|e| e.to_string())?;

        eprintln!("fluxion: recording {secs:.1}s @ {dev_fs} Hz ({channels} ch)…");
        let mut captured = Vec::with_capacity(want);
        let start = Instant::now();
        while captured.len() < want && start.elapsed().as_secs_f32() < secs + 1.0 {
            while let Some(s) = rx.pop() {
                captured.push(s);
                if captured.len() >= want {
                    break;
                }
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        drop(stream);
        captured.truncate(want);

        // De-interleave into channels, then process + write at the capture rate.
        let mut chans = vec![Vec::with_capacity(want / channels); channels];
        for frame in captured.chunks_exact(channels) {
            for (c, &s) in frame.iter().enumerate() {
                chans[c].push(s);
            }
        }
        write_output(out, &process(&graph, &Signal::new(dev_fs, chans)), enc)
    }
}

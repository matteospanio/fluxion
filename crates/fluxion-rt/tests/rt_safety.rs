//! Real-time-safety tests (plan task G6).
//!
//! The audio thread must never allocate. We install a global allocator that flags any allocation
//! made while a thread-local "real-time section" is active, then assert the engine's hot path makes
//! zero allocations across a long run with concurrent parameter automation — and that it keeps up
//! with real time at 128 frames / 48 kHz (no xruns). Run: `cargo test -p fluxion-rt --test rt_safety`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Instant;

use fluxion_rt::{Biquad, Command, RtEngine};

// --- allocation tracker ----------------------------------------------------------------------

thread_local! {
    /// When set, allocations on *this* thread are counted as real-time-safety violations.
    static RT_ACTIVE: Cell<bool> = const { Cell::new(false) };
    /// Per-thread violation count — thread-local so tests running in parallel (e.g. one that
    /// intentionally allocates) can't contaminate each other's measurement.
    static LOCAL_VIOLATIONS: Cell<usize> = const { Cell::new(0) };
}

struct TrackingAllocator;

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note();
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note();
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[inline]
fn note() {
    // `RT_ACTIVE` is a const-initialized `Cell<bool>` → access is allocation-free, so this can't
    // recurse into the allocator.
    if RT_ACTIVE.with(|f| f.get()) {
        LOCAL_VIOLATIONS.with(|v| v.set(v.get() + 1));
    }
}

#[global_allocator]
static ALLOC: TrackingAllocator = TrackingAllocator;

/// Run `f` inside a real-time section; return its result and the number of allocations it made.
fn rt_section<R>(f: impl FnOnce() -> R) -> (R, usize) {
    let before = LOCAL_VIOLATIONS.with(|v| v.get());
    RT_ACTIVE.with(|a| a.set(true));
    let r = f();
    RT_ACTIVE.with(|a| a.set(false));
    let after = LOCAL_VIOLATIONS.with(|v| v.get());
    (r, after - before)
}

fn stable_cascade() -> Vec<Biquad> {
    // Four stable second-order sections (|a2| < 1, |a1| < 1 + a2). Exact values are irrelevant here.
    [
        [0.2929, 0.5858, 0.2929, 0.0, 0.1716],
        [0.5, 0.3, -0.1, -0.2, 0.05],
        [0.8, -0.2, 0.1, 0.1, -0.3],
        [0.6, 0.1, -0.05, -0.1, 0.02],
    ]
    .iter()
    .map(|c| Biquad {
        b0: c[0],
        b1: c[1],
        b2: c[2],
        a1: c[3],
        a2: c[4],
    })
    .collect()
}

// --- tests -----------------------------------------------------------------------------------

#[test]
fn process_block_is_allocation_free() {
    let (mut eng, mut tx) = RtEngine::new(stable_cascade(), 1.0, 64);
    let input = vec![0.1f32; 128];
    let mut out = vec![0.0f32; 128];

    // Prime thread-locals and warm up *outside* the measured section.
    let _ = rt_section(|| {});
    eng.process_block(&input, &mut out);
    tx.push(Command::SetGain {
        target: 0.5,
        ramp_samples: 64,
    })
    .unwrap();

    let (_, allocs) = rt_section(|| {
        for _ in 0..1000 {
            eng.process_block(&input, &mut out);
        }
    });
    assert_eq!(
        allocs, 0,
        "RtEngine::process_block allocated on the audio thread"
    );
}

#[test]
fn meta_tracker_detects_allocation() {
    // Guard against a false-negative test: a real allocation inside a section must be counted.
    let _ = rt_section(|| {});
    let (v, allocs) = rt_section(|| {
        let v: Vec<u8> = Vec::with_capacity(4096); // allocates
        v
    });
    std::hint::black_box(v);
    assert!(allocs >= 1, "tracker failed to observe an allocation");
}

#[test]
fn xrun_stress_128_at_48k() {
    const FS: usize = 48_000;
    const BLOCK: usize = 128;
    const SECONDS: usize = 5;
    let blocks = FS * SECONDS / BLOCK;

    let (mut eng, mut tx) = RtEngine::new(stable_cascade(), 1.0, 64);
    let input: Vec<f32> = (0..BLOCK).map(|i| 0.25 * (0.05 * i as f32).sin()).collect();
    let mut out = vec![0.0f32; BLOCK];

    // Control thread: hammer the command queue with gain automation while audio runs.
    let stop = Arc::new(AtomicBool::new(false));
    let stop_ctrl = stop.clone();
    let ctrl = thread::spawn(move || {
        let mut g = 0.0f32;
        while !stop_ctrl.load(Ordering::Relaxed) {
            let _ = tx.push(Command::SetGain {
                target: g,
                ramp_samples: BLOCK as u32,
            });
            g = 1.0 - g;
            thread::yield_now();
        }
    });

    let _ = rt_section(|| {});
    eng.process_block(&input, &mut out); // warm up

    let t0 = Instant::now();
    let (_, allocs) = rt_section(|| {
        for _ in 0..blocks {
            eng.process_block(&input, &mut out);
        }
    });
    let elapsed = t0.elapsed();

    stop.store(true, Ordering::Relaxed);
    ctrl.join().unwrap();

    assert_eq!(
        allocs, 0,
        "audio loop allocated under concurrent automation"
    );
    assert!(out.iter().all(|x| x.is_finite()), "non-finite output");
    // No xruns: {SECONDS}s of audio must process in well under {SECONDS}s of wall time.
    let audio = SECONDS as f64;
    assert!(
        elapsed.as_secs_f64() < audio,
        "did not keep up with real time: {:?} to process {audio}s of audio",
        elapsed
    );
    println!(
        "xrun stress: {blocks} blocks of {BLOCK} @ {FS} Hz in {:.1} ms ({:.0}x real time)",
        elapsed.as_secs_f64() * 1e3,
        audio / elapsed.as_secs_f64()
    );
}

//! Varispeed in an audio callback: no allocations, and it keeps up (ROADMAP R5).
//!
//! Scrubbing is the case where the ratio moves *while* the callback is running, so the two things
//! that would sink it are an allocation on the audio thread and a block that misses its deadline.
//! Both are measured here rather than argued about — with the speed moving every block throughout,
//! because a varispeed nobody is dragging is not the case that matters.
//!
//! Same shape as `resample_rt.rs` and `fluxion-rt/tests/rt_safety.rs`: a global allocator counting
//! allocations made while a thread-local flag is set. Its own test binary, because a
//! `#[global_allocator]` is per-binary.
//!
//! Run: `cargo test -p fluxion-ops --test varispeed_rt`

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::time::Instant;

use fluxion_ops::resample::Quality;
use fluxion_ops::varispeed::Varispeed;

thread_local! {
    /// When set, allocations on *this* thread are counted as violations.
    static RT_ACTIVE: Cell<bool> = const { Cell::new(false) };
    /// Per-thread count, so tests running in parallel cannot contaminate each other.
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
    // `RT_ACTIVE` is a const-initialized `Cell<bool>`, so reading it cannot itself allocate and
    // this cannot recurse into the allocator.
    if RT_ACTIVE.with(|f| f.get()) {
        LOCAL_VIOLATIONS.with(|v| v.set(v.get() + 1));
    }
}

#[global_allocator]
static ALLOC: TrackingAllocator = TrackingAllocator;

/// Run `f` inside a real-time section; return its result and how many allocations it made.
fn rt_section<R>(f: impl FnOnce() -> R) -> (R, usize) {
    let before = LOCAL_VIOLATIONS.with(|v| v.get());
    RT_ACTIVE.with(|a| a.set(true));
    let r = f();
    RT_ACTIVE.with(|a| a.set(false));
    (r, LOCAL_VIOLATIONS.with(|v| v.get()) - before)
}

const FS: u32 = 48_000;
const BLOCK: usize = 128;

fn material(frames: usize) -> Vec<f32> {
    (0..frames).map(|i| (i as f32 * 0.05).sin() * 0.5).collect()
}

/// Pre-condition: the varispeed and both buffers are built outside the section.
/// Post-condition: five seconds of scrubbing, with the speed moving every block, allocates zero
/// times and keeps well inside real time.
#[test]
fn scrubbing_allocates_nothing_and_meets_the_deadline() {
    const SECONDS: usize = 5;
    let blocks = FS as usize * SECONDS / BLOCK;

    let mut v = Varispeed::new(FS, Quality::Fast, 4.0, 512);
    let mut out = vec![0.0f32; BLOCK];
    // Enough source for the fastest speed to chew through.
    let input = material(FS as usize * SECONDS * 4 + 4096);

    // Warm up outside the measured section.
    v.process(&input[..512], &mut out);
    let _ = rt_section(|| {});

    let t0 = Instant::now();
    let ((at, made), allocations) = rt_section(|| {
        let (mut at, mut made) = (0usize, 0usize);
        for b in 0..blocks {
            // A scrub bar being dragged: a new speed every block, all the way across the range.
            v.set_speed(0.25 + 3.75 * (b as f32 / blocks as f32));
            let mut written = 0;
            while written < BLOCK && at + 512 <= input.len() {
                let (consumed, w) = v.process(&input[at..at + 512], &mut out[written..]);
                at += consumed;
                written += w;
                if consumed == 0 && w == 0 {
                    break;
                }
            }
            made += written;
        }
        (at, made)
    });
    let elapsed = t0.elapsed();

    assert_eq!(allocations, 0, "process() allocated {allocations} times");
    // It really did the work rather than returning early: five seconds of output, and more input
    // than that eaten, because most of the sweep is above 1x.
    assert!(
        made > FS as usize * SECONDS - BLOCK,
        "only produced {made} frames of the {} asked for",
        FS as usize * SECONDS
    );
    assert!(
        at > made,
        "at speeds above 1x it should eat more than it makes"
    );

    // The deadline, in aggregate, and the bar is "keeps up" — the same one `fluxion-rt`'s xrun
    // stress uses, and for the same reason: a tighter number measures the CI machine, not the code.
    // Measured here: 1.1 % of real time in release, 10.7 % unoptimized.
    let audio = SECONDS as f64;
    assert!(
        elapsed.as_secs_f64() < audio,
        "did not keep up with real time: {elapsed:?} to scrub {audio}s of audio"
    );
    println!(
        "{SECONDS}s of scrubbing in {elapsed:?} ({:.1}% of real time), 0 allocations",
        elapsed.as_secs_f64() / audio * 100.0
    );
}

/// `Hq` is the expensive end, and a host may well pick it for a tape effect that is being listened
/// to rather than scrubbed. It has to fit too.
#[test]
fn the_expensive_quality_also_fits_the_deadline() {
    const SECONDS: usize = 2;
    let blocks = FS as usize * SECONDS / BLOCK;

    let mut v = Varispeed::new(FS, Quality::Hq, 2.0, 512);
    v.set_speed(2.0);
    let mut out = vec![0.0f32; BLOCK];
    let input = material(FS as usize * SECONDS * 2 + 4096);

    v.process(&input[..512], &mut out);
    let _ = rt_section(|| {});

    let t0 = Instant::now();
    let (_, allocations) = rt_section(|| {
        let mut at = 0usize;
        for _ in 0..blocks {
            let mut written = 0;
            while written < BLOCK && at + 512 <= input.len() {
                let (consumed, w) = v.process(&input[at..at + 512], &mut out[written..]);
                at += consumed;
                written += w;
                if consumed == 0 && w == 0 {
                    break;
                }
            }
        }
    });
    let elapsed = t0.elapsed();

    assert_eq!(allocations, 0, "Hq process() allocated {allocations} times");
    // 3.6 % of real time in release, 38 % unoptimized — the widest kernel this can ask for, since
    // 2x doubles it. Same "keeps up" bar as above.
    assert!(
        elapsed.as_secs_f64() < SECONDS as f64,
        "{SECONDS}s at Hq and 2x took {elapsed:?}"
    );
    println!(
        "{SECONDS}s at Hq, 2x, in {elapsed:?} ({:.1}% of real time)",
        elapsed.as_secs_f64() / SECONDS as f64 * 100.0
    );
}

/// Guard against a false negative: the tracker has to see a real allocation.
#[test]
fn the_tracker_sees_an_allocation() {
    let _ = rt_section(|| {});
    let (v, allocations) = rt_section(|| Vec::<u8>::with_capacity(4096));
    std::hint::black_box(v);
    assert!(allocations >= 1, "the tracker missed an allocation");
}

/// `set_speed` and `reset` are called from the callback too, and neither is a moment to allocate.
#[test]
fn set_speed_and_reset_allocate_nothing() {
    let mut v = Varispeed::new(FS, Quality::Fast, 4.0, 256);
    let mut out = vec![0.0f32; BLOCK];
    v.process(&material(256), &mut out);

    let _ = rt_section(|| {});
    let (_, allocations) = rt_section(|| {
        for i in 0..1000 {
            v.set_speed(0.5 + (i % 8) as f32 * 0.25);
        }
        v.reset();
    });
    assert_eq!(
        allocations, 0,
        "set_speed/reset allocated {allocations} times"
    );
}

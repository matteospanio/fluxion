//! The streaming resampler allocates nothing after `new` (ROADMAP R1).
//!
//! A resampler is only useful in a callback if converting a block cannot block, and the one way it
//! could is the allocator. The filter table, the history, and the phase index are all sized in
//! `new`; `process` may only read and write them.
//!
//! Same shape as `fluxion-rt/tests/rt_safety.rs`: a global allocator that counts allocations made
//! while a thread-local flag is set. It lives in its own test binary because a `#[global_allocator]`
//! is per-binary and the oracle test wants an ordinary one.
//!
//! Run: `cargo test -p fluxion-ops --test resample_rt`

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use fluxion_ops::resample::{Quality, Resampler};

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

/// Pre-condition: the resampler and both buffers are built outside the section.
/// Post-condition: converting 10 s of audio in 128-frame blocks allocates zero times.
#[test]
fn process_allocates_nothing() {
    let block = 128;
    let mut r = Resampler::new(48_000, 44_100, Quality::Hq, block);
    let mut out = vec![0.0f32; r.max_output(block)];
    let input: Vec<f32> = (0..48_000 * 10)
        .map(|i| (i as f32 * 0.01).sin() * 0.5)
        .collect();

    let (frames, allocations) = rt_section(|| {
        let mut frames = 0;
        for chunk in input.chunks(block) {
            frames += r.process(chunk, &mut out);
        }
        frames
    });

    assert_eq!(allocations, 0, "process() allocated {allocations} times");
    // Sanity: it really did the work, rather than returning early on every block.
    let expected = 44_100 * 10;
    assert!(
        frames.abs_diff(expected) < 64,
        "converted {frames} frames, expected about {expected}"
    );
}

/// The same, for every quality and both directions — a downsampler and an upsampler take different
/// paths through the table, and `Fast` sizes it differently.
#[test]
fn process_allocates_nothing_in_either_direction() {
    let block = 256;
    for quality in [Quality::Fast, Quality::Hq] {
        for (from, to) in [(48_000, 44_100), (44_100, 48_000), (48_000, 16_000)] {
            let mut r = Resampler::new(from, to, quality, block);
            let mut out = vec![0.0f32; r.max_output(block)];
            let input = vec![0.25f32; from as usize];

            let (_, allocations) = rt_section(|| {
                for chunk in input.chunks(block) {
                    r.process(chunk, &mut out);
                }
            });
            assert_eq!(
                allocations, 0,
                "{from} -> {to} at {quality:?} allocated {allocations} times"
            );
        }
    }
}

/// `reset` is called between takes, which is still not a moment to allocate.
#[test]
fn reset_allocates_nothing() {
    let mut r = Resampler::new(48_000, 44_100, Quality::Hq, 128);
    let mut out = vec![0.0f32; r.max_output(128)];
    r.process(&[0.5; 128], &mut out);

    let (_, allocations) = rt_section(|| r.reset());
    assert_eq!(allocations, 0, "reset() allocated {allocations} times");
}

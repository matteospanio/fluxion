//! Lock-free single-producer / single-consumer ring buffer (plan task G1).
//!
//! The audio thread is the consumer (or producer) and must never block or allocate. This is the
//! standard SPSC ring: free-running `head`/`tail` counters (so the full capacity is usable — no
//! sacrificed slot), masked for indexing, published with release / observed with acquire so the data
//! write is visible before the index that exposes it. Single-producer-single-consumer is a
//! **contract**: exactly one [`Producer`] and one [`Consumer`], each on one thread.
//!
//! `T: Copy` keeps it allocation- and drop-free (audio samples are `Copy`); a generic-`T` version
//! with in-place drop is a later need, not this one.

use std::cell::UnsafeCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Ring<T> {
    buf: Box<[UnsafeCell<T>]>,
    mask: usize,
    head: AtomicUsize, // next write position (producer owns)
    tail: AtomicUsize, // next read position (consumer owns)
}

// SPSC discipline: the producer only writes the slot at `head` (which the consumer won't read until
// `head` is published), the consumer only reads the slot at `tail`. No slot is touched by both at
// once, so sharing the cells across the two threads is sound.
unsafe impl<T: Send> Sync for Ring<T> {}
unsafe impl<T: Send> Send for Ring<T> {}

/// Create an SPSC ring holding at least `capacity` items (rounded up to a power of two), returning
/// the producer and consumer handles. `capacity` is clamped to at least 1.
pub fn channel<T: Copy + Default>(capacity: usize) -> (Producer<T>, Consumer<T>) {
    let cap = capacity.max(1).next_power_of_two();
    let buf = (0..cap).map(|_| UnsafeCell::new(T::default())).collect();
    let ring = Arc::new(Ring {
        buf,
        mask: cap - 1,
        head: AtomicUsize::new(0),
        tail: AtomicUsize::new(0),
    });
    (Producer { ring: ring.clone() }, Consumer { ring })
}

/// The writing half. Lives on one thread.
pub struct Producer<T> {
    ring: Arc<Ring<T>>,
}
/// The reading half. Lives on one thread.
pub struct Consumer<T> {
    ring: Arc<Ring<T>>,
}

unsafe impl<T: Send> Send for Producer<T> {}
unsafe impl<T: Send> Send for Consumer<T> {}

impl<T: Copy> Producer<T> {
    /// Total slot count (a power of two).
    pub fn capacity(&self) -> usize {
        self.ring.buf.len()
    }

    /// Push one item; returns `Err(item)` if the ring is full (never blocks, never allocates).
    pub fn push(&self, item: T) -> Result<(), T> {
        let head = self.ring.head.load(Ordering::Relaxed);
        let tail = self.ring.tail.load(Ordering::Acquire);
        if head.wrapping_sub(tail) == self.ring.buf.len() {
            return Err(item); // full
        }
        // Sole writer of this slot; the consumer can't reach it until `head` is published below.
        unsafe { *self.ring.buf[head & self.ring.mask].get() = item };
        self.ring.head.store(head.wrapping_add(1), Ordering::Release);
        Ok(())
    }
}

impl<T: Copy> Consumer<T> {
    /// Pop one item, or `None` if empty (never blocks, never allocates).
    pub fn pop(&self) -> Option<T> {
        let tail = self.ring.tail.load(Ordering::Relaxed);
        let head = self.ring.head.load(Ordering::Acquire);
        if head == tail {
            return None; // empty
        }
        let item = unsafe { *self.ring.buf[tail & self.ring.mask].get() };
        self.ring.tail.store(tail.wrapping_add(1), Ordering::Release);
        Some(item)
    }

    /// Number of items currently available to read.
    pub fn len(&self) -> usize {
        let head = self.ring.head.load(Ordering::Acquire);
        let tail = self.ring.tail.load(Ordering::Relaxed);
        head.wrapping_sub(tail)
    }

    /// True if no items are available.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::channel;

    #[test]
    fn capacity_rounds_up_to_power_of_two() {
        let (p, _c) = channel::<f32>(100);
        assert_eq!(p.capacity(), 128);
    }

    #[test]
    fn fifo_order_and_full_empty() {
        let (p, c) = channel::<u32>(4); // 4 slots
        assert!(c.pop().is_none(), "empty at start");
        for i in 0..4 {
            assert!(p.push(i).is_ok());
        }
        assert_eq!(p.push(99), Err(99), "full after capacity pushes");
        for i in 0..4 {
            assert_eq!(c.pop(), Some(i), "FIFO order");
        }
        assert!(c.pop().is_none(), "empty after draining");
    }

    #[test]
    fn wraps_around_many_times() {
        let (p, c) = channel::<usize>(8);
        // Push/pop far past capacity to exercise the wrapping counters + mask.
        for i in 0..10_000 {
            assert!(p.push(i).is_ok());
            assert_eq!(c.pop(), Some(i));
        }
        assert!(c.is_empty());
    }

    #[test]
    fn concurrent_spsc_preserves_every_item() {
        use std::thread;
        let (p, c) = channel::<usize>(64);
        const N: usize = 1_000_000;
        let prod = thread::spawn(move || {
            let mut i = 0;
            while i < N {
                if p.push(i).is_ok() {
                    i += 1;
                }
            }
        });
        let mut next = 0;
        while next < N {
            if let Some(v) = c.pop() {
                assert_eq!(v, next, "items arrive in order with none lost/dup");
                next += 1;
            }
        }
        prod.join().unwrap();
    }
}

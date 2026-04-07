//! Pool-based memory management.
//!
//! A [`Pool`] is a general-purpose allocation facility — not tensor-specific.
//! Tensors and any other large buffer consumers allocate through pools.
//!
//! Two concrete implementations:
//! - [`SystemPool`] — stateless passthrough to the global allocator (tests, inline constants)
//! - [`TrackedPool`] — usage tracking, budget enforcement, incremental free

use std::alloc::Layout;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Minimum alignment for all pool allocations.
/// Covers all primitive types up to f128 and common SIMD widths.
const POOL_ALIGNMENT: usize = 16;

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum AllocationError {
    #[error(
        "allocation of {requested} bytes exceeds pool budget \
         ({in_use} of {budget} bytes in use)"
    )]
    BudgetExceeded {
        requested: usize,
        budget: usize,
        in_use: usize,
    },
    #[error("system allocator failed for {size} bytes (align {align})")]
    SystemAllocFailed { size: usize, align: usize },
}

// ---------------------------------------------------------------------------
// Pool trait
// ---------------------------------------------------------------------------

/// A general-purpose allocation facility.
///
/// `Buffer` is a GAT — its lifetime is tied to the pool via `&self` borrow.
/// Dropping a buffer returns its memory to the pool.
///
/// All implementations must be `Send + Sync`.
pub trait Pool: Send + Sync {
    /// An owned handle to an allocated region. Derefs to `[u8]`.
    type Buffer<'a>: Deref<Target = [u8]> + DerefMut + Send + Sync
    where
        Self: 'a;

    /// Allocate a zero-initialized, 16-byte-aligned buffer of `size` bytes.
    fn allocate(&self, size: usize) -> Result<Self::Buffer<'_>, AllocationError>;

    /// Bytes currently held by live buffers (not yet dropped).
    fn bytes_in_use(&self) -> usize;

    /// Maximum bytes this pool will allocate, or `None` for unlimited.
    fn budget(&self) -> Option<usize>;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Allocate `size` bytes with `POOL_ALIGNMENT`, zero-initialized.
/// Returns null on failure. Caller must not call this with size == 0.
unsafe fn alloc_aligned_zeroed(size: usize) -> *mut u8 {
    debug_assert!(size > 0);
    unsafe {
        let layout = Layout::from_size_align_unchecked(size, POOL_ALIGNMENT);
        std::alloc::alloc_zeroed(layout)
    }
}

/// Deallocate a pointer previously returned by `alloc_aligned_zeroed`.
unsafe fn dealloc_aligned(ptr: *mut u8, size: usize) {
    debug_assert!(size > 0);
    unsafe {
        let layout = Layout::from_size_align_unchecked(size, POOL_ALIGNMENT);
        std::alloc::dealloc(ptr, layout)
    }
}

// ---------------------------------------------------------------------------
// SystemPool
// ---------------------------------------------------------------------------

/// Stateless passthrough to the global allocator. No tracking, no budget.
///
/// Buffer is owned (`'static`-compatible) — suitable for inline constants,
/// graph-embedded tensor data, and tests.
#[derive(Debug, Clone, Copy)]
pub struct SystemPool;

/// Buffer allocated by [`SystemPool`]. Owns its memory outright.
pub struct SystemBuffer {
    ptr: *mut u8,
    len: usize,
}

// Safety: SystemBuffer exclusively owns its allocation — no shared state.
unsafe impl Send for SystemBuffer {}
unsafe impl Sync for SystemBuffer {}

impl Pool for SystemPool {
    type Buffer<'a> = SystemBuffer;

    fn allocate(&self, size: usize) -> Result<SystemBuffer, AllocationError> {
        if size == 0 {
            return Ok(SystemBuffer {
                ptr: std::ptr::NonNull::dangling().as_ptr(),
                len: 0,
            });
        }
        let ptr = unsafe { alloc_aligned_zeroed(size) };
        if ptr.is_null() {
            return Err(AllocationError::SystemAllocFailed {
                size,
                align: POOL_ALIGNMENT,
            });
        }
        Ok(SystemBuffer { ptr, len: size })
    }

    fn bytes_in_use(&self) -> usize {
        0
    }

    fn budget(&self) -> Option<usize> {
        None
    }
}

impl Deref for SystemBuffer {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl DerefMut for SystemBuffer {
    fn deref_mut(&mut self) -> &mut [u8] {
        if self.len == 0 {
            return &mut [];
        }
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for SystemBuffer {
    fn drop(&mut self) {
        if self.len > 0 {
            unsafe { dealloc_aligned(self.ptr, self.len) }
        }
    }
}

impl Clone for SystemBuffer {
    fn clone(&self) -> Self {
        if self.len == 0 {
            return SystemBuffer {
                ptr: std::ptr::NonNull::dangling().as_ptr(),
                len: 0,
            };
        }
        let ptr = unsafe { alloc_aligned_zeroed(self.len) };
        assert!(!ptr.is_null(), "SystemBuffer::clone allocation failed");
        unsafe { std::ptr::copy_nonoverlapping(self.ptr, ptr, self.len) };
        SystemBuffer { ptr, len: self.len }
    }
}

impl std::fmt::Debug for SystemBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SystemBuffer")
            .field("len", &self.len)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// TrackedPool
// ---------------------------------------------------------------------------

/// Pool with usage tracking and optional budget enforcement.
///
/// Every allocation is tracked. Dropping a buffer decrements the usage counter
/// and frees memory back to the system. Budget is checked on allocation —
/// exceeding it returns `AllocationError::BudgetExceeded`.
///
/// `Send + Sync` — safe to share across threads (e.g. model pool shared by
/// concurrent request handlers via `spawn_blocking`).
pub struct TrackedPool {
    /// Bytes currently allocated (live buffers). Atomic for lock-free reads
    /// and contention-free drops (most common path).
    bytes_in_use: AtomicUsize,
    /// Protects budget-check-then-allocate atomicity. Only held during
    /// `allocate()`, never during `drop()` of buffers (drops use atomic sub).
    alloc_lock: Mutex<()>,
    budget: Option<usize>,
}

/// Buffer allocated by [`TrackedPool`]. Borrows the pool — dropping the buffer
/// decrements the pool's usage counter.
pub struct TrackedBuffer<'a> {
    ptr: *mut u8,
    len: usize,
    pool: &'a TrackedPool,
}

// Safety: TrackedBuffer exclusively owns its allocation. The &TrackedPool
// reference is safe because TrackedPool is Sync (atomic + mutex).
unsafe impl Send for TrackedBuffer<'_> {}
unsafe impl Sync for TrackedBuffer<'_> {}

impl TrackedPool {
    /// Create a new pool. Pass `Some(n)` to enforce a byte budget,
    /// or `None` for unlimited.
    pub fn new(budget: Option<usize>) -> Self {
        Self {
            bytes_in_use: AtomicUsize::new(0),
            alloc_lock: Mutex::new(()),
            budget,
        }
    }
}

impl Pool for TrackedPool {
    type Buffer<'a> = TrackedBuffer<'a>;

    fn allocate(&self, size: usize) -> Result<TrackedBuffer<'_>, AllocationError> {
        if size == 0 {
            return Ok(TrackedBuffer {
                ptr: std::ptr::NonNull::dangling().as_ptr(),
                len: 0,
                pool: self,
            });
        }

        // Lock to make budget-check + bytes_in_use bump atomic with respect
        // to other allocators. Drops don't take this lock.
        // Ignore mutex poisoning — the guard protects a budget check, not
        // data that could be in an inconsistent state from a prior panic.
        let _guard = self.alloc_lock.lock().unwrap_or_else(|e| e.into_inner());

        let current = self.bytes_in_use.load(Ordering::Relaxed);
        if let Some(budget) = self.budget
            && current.checked_add(size).is_none_or(|total| total > budget)
        {
            return Err(AllocationError::BudgetExceeded {
                requested: size,
                budget,
                in_use: current,
            });
        }

        let ptr = unsafe { alloc_aligned_zeroed(size) };
        if ptr.is_null() {
            return Err(AllocationError::SystemAllocFailed {
                size,
                align: POOL_ALIGNMENT,
            });
        }

        self.bytes_in_use.fetch_add(size, Ordering::Relaxed);
        Ok(TrackedBuffer {
            ptr,
            len: size,
            pool: self,
        })
    }

    fn bytes_in_use(&self) -> usize {
        self.bytes_in_use.load(Ordering::Relaxed)
    }

    fn budget(&self) -> Option<usize> {
        self.budget
    }
}

impl Deref for TrackedBuffer<'_> {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl DerefMut for TrackedBuffer<'_> {
    fn deref_mut(&mut self) -> &mut [u8] {
        if self.len == 0 {
            return &mut [];
        }
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for TrackedBuffer<'_> {
    fn drop(&mut self) {
        if self.len > 0 {
            self.pool
                .bytes_in_use
                .fetch_sub(self.len, Ordering::Relaxed);
            unsafe { dealloc_aligned(self.ptr, self.len) }
        }
    }
}

impl std::fmt::Debug for TrackedPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrackedPool")
            .field("bytes_in_use", &self.bytes_in_use())
            .field("budget", &self.budget)
            .finish()
    }
}

impl std::fmt::Debug for TrackedBuffer<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrackedBuffer")
            .field("len", &self.len)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// ArcTrackedPool
// ---------------------------------------------------------------------------

/// A [`TrackedPool`] behind an [`Arc`] whose buffers each hold their own
/// `Arc<TrackedPool>` clone.
///
/// Because the buffer carries a live `Arc` reference to the pool for its
/// entire lifetime, the [`Pool::Buffer`] GAT type doesn't borrow the outer
/// `&self` — buffers can live for `'static` and still safely decrement the
/// pool's byte counter in `Drop`. This is what lets cached tensors sit in
/// long-lived data structures (like [`SuperGraphCache`](crate::super_graph::cache::SuperGraphCache))
/// without tying the cache's lifetime to a stack-allocated pool.
///
/// Clone is cheap (Arc bump). All clones share the same byte counter and
/// budget, so summing across clones is never needed — query any one of
/// them to get the aggregate.
#[derive(Clone, Debug)]
pub struct ArcTrackedPool {
    inner: Arc<TrackedPool>,
}

impl ArcTrackedPool {
    /// Create a new pool. Pass `Some(n)` to enforce a byte budget,
    /// or `None` for unlimited.
    pub fn new(budget: Option<usize>) -> Self {
        Self {
            inner: Arc::new(TrackedPool::new(budget)),
        }
    }

    /// Wrap an existing `Arc<TrackedPool>`. Useful when the same underlying
    /// pool is already shared elsewhere (e.g. with a stats sampler).
    pub fn from_arc(inner: Arc<TrackedPool>) -> Self {
        Self { inner }
    }

    /// Borrow the underlying `Arc<TrackedPool>` (for sampling, etc.).
    pub fn inner(&self) -> &Arc<TrackedPool> {
        &self.inner
    }
}

/// Buffer allocated by [`ArcTrackedPool`]. Owns an `Arc<TrackedPool>`
/// clone — the pool stays alive as long as the buffer exists, so the
/// buffer has no borrow lifetime and is `'static`-safe.
pub struct ArcTrackedBuffer {
    ptr: *mut u8,
    len: usize,
    /// Keeps the pool alive so `Drop` can decrement its counter.
    pool: Arc<TrackedPool>,
}

// Safety: `ArcTrackedBuffer` exclusively owns its allocation. The
// `Arc<TrackedPool>` is itself `Send + Sync`.
unsafe impl Send for ArcTrackedBuffer {}
unsafe impl Sync for ArcTrackedBuffer {}

impl Pool for ArcTrackedPool {
    type Buffer<'a>
        = ArcTrackedBuffer
    where
        Self: 'a;

    fn allocate(&self, size: usize) -> Result<ArcTrackedBuffer, AllocationError> {
        if size == 0 {
            return Ok(ArcTrackedBuffer {
                ptr: std::ptr::NonNull::dangling().as_ptr(),
                len: 0,
                pool: self.inner.clone(),
            });
        }

        // Same budget-check + alloc dance as `TrackedPool::allocate`; we
        // duplicate it here rather than calling through because the
        // returned buffer needs to own an `Arc` clone, not a borrow.
        let _guard = self
            .inner
            .alloc_lock
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let current = self.inner.bytes_in_use.load(Ordering::Relaxed);
        if let Some(budget) = self.inner.budget
            && current.checked_add(size).is_none_or(|total| total > budget)
        {
            return Err(AllocationError::BudgetExceeded {
                requested: size,
                budget,
                in_use: current,
            });
        }

        let ptr = unsafe { alloc_aligned_zeroed(size) };
        if ptr.is_null() {
            return Err(AllocationError::SystemAllocFailed {
                size,
                align: POOL_ALIGNMENT,
            });
        }

        self.inner.bytes_in_use.fetch_add(size, Ordering::Relaxed);
        Ok(ArcTrackedBuffer {
            ptr,
            len: size,
            pool: self.inner.clone(),
        })
    }

    fn bytes_in_use(&self) -> usize {
        self.inner.bytes_in_use()
    }

    fn budget(&self) -> Option<usize> {
        self.inner.budget()
    }
}

impl Deref for ArcTrackedBuffer {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl DerefMut for ArcTrackedBuffer {
    fn deref_mut(&mut self) -> &mut [u8] {
        if self.len == 0 {
            return &mut [];
        }
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for ArcTrackedBuffer {
    fn drop(&mut self) {
        if self.len > 0 {
            self.pool
                .bytes_in_use
                .fetch_sub(self.len, Ordering::Relaxed);
            unsafe { dealloc_aligned(self.ptr, self.len) }
        }
    }
}

impl std::fmt::Debug for ArcTrackedBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArcTrackedBuffer")
            .field("len", &self.len)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- SystemPool --

    #[test]
    fn system_pool_allocate_and_readwrite() {
        let pool = SystemPool;
        let mut buf = pool.allocate(64).unwrap();
        assert_eq!(buf.len(), 64);
        assert!(buf.iter().all(|&b| b == 0), "should be zero-initialized");
        buf[0] = 0xAB;
        buf[63] = 0xCD;
        assert_eq!(buf[0], 0xAB);
        assert_eq!(buf[63], 0xCD);
    }

    #[test]
    fn system_pool_zero_size() {
        let pool = SystemPool;
        let buf = pool.allocate(0).unwrap();
        assert_eq!(buf.len(), 0);
        assert_eq!(&*buf, &[] as &[u8]);
    }

    #[test]
    fn system_pool_alignment() {
        let pool = SystemPool;
        for &size in &[1, 7, 16, 31, 64, 4096] {
            let buf = pool.allocate(size).unwrap();
            let addr = buf.as_ptr() as usize;
            assert_eq!(
                addr % POOL_ALIGNMENT,
                0,
                "buffer of size {size} not aligned to {POOL_ALIGNMENT}"
            );
        }
    }

    #[test]
    fn system_pool_no_tracking() {
        let pool = SystemPool;
        let _buf = pool.allocate(1024).unwrap();
        assert_eq!(pool.bytes_in_use(), 0);
        assert_eq!(pool.budget(), None);
    }

    // -- TrackedPool --

    #[test]
    fn tracked_pool_basic_tracking() {
        let pool = TrackedPool::new(None);
        assert_eq!(pool.bytes_in_use(), 0);

        let buf1 = pool.allocate(100).unwrap();
        assert_eq!(pool.bytes_in_use(), 100);

        let buf2 = pool.allocate(200).unwrap();
        assert_eq!(pool.bytes_in_use(), 300);

        drop(buf1);
        assert_eq!(pool.bytes_in_use(), 200);

        drop(buf2);
        assert_eq!(pool.bytes_in_use(), 0);
    }

    #[test]
    fn tracked_pool_readwrite() {
        let pool = TrackedPool::new(None);
        let mut buf = pool.allocate(32).unwrap();
        assert!(buf.iter().all(|&b| b == 0));
        buf[0] = 0xFF;
        buf[31] = 0x01;
        assert_eq!(buf[0], 0xFF);
        assert_eq!(buf[31], 0x01);
    }

    #[test]
    fn tracked_pool_zero_size() {
        let pool = TrackedPool::new(None);
        let buf = pool.allocate(0).unwrap();
        assert_eq!(buf.len(), 0);
        assert_eq!(pool.bytes_in_use(), 0);
        drop(buf);
        assert_eq!(pool.bytes_in_use(), 0);
    }

    #[test]
    fn tracked_pool_alignment() {
        let pool = TrackedPool::new(None);
        for &size in &[1, 7, 16, 31, 64, 4096] {
            let buf = pool.allocate(size).unwrap();
            let addr = buf.as_ptr() as usize;
            assert_eq!(
                addr % POOL_ALIGNMENT,
                0,
                "buffer of size {size} not aligned to {POOL_ALIGNMENT}"
            );
        }
    }

    #[test]
    fn tracked_pool_budget_enforcement() {
        let pool = TrackedPool::new(Some(256));
        assert_eq!(pool.budget(), Some(256));

        let buf1 = pool.allocate(200).unwrap();
        assert_eq!(pool.bytes_in_use(), 200);

        // This should fail — would push to 300 > 256
        let err = pool.allocate(100).unwrap_err();
        assert!(
            matches!(err, AllocationError::BudgetExceeded { .. }),
            "expected BudgetExceeded, got {err:?}"
        );
        assert_eq!(pool.bytes_in_use(), 200);

        // Free the first buffer, now 100 should fit
        drop(buf1);
        assert_eq!(pool.bytes_in_use(), 0);

        let _buf2 = pool.allocate(100).unwrap();
        assert_eq!(pool.bytes_in_use(), 100);
    }

    #[test]
    fn tracked_pool_budget_exact_fit() {
        let pool = TrackedPool::new(Some(100));
        let _buf = pool.allocate(100).unwrap();
        assert_eq!(pool.bytes_in_use(), 100);

        // Exactly at budget — one more byte should fail
        let err = pool.allocate(1).unwrap_err();
        assert!(matches!(err, AllocationError::BudgetExceeded { .. }));
    }

    #[test]
    fn tracked_pool_unlimited() {
        let pool = TrackedPool::new(None);
        assert_eq!(pool.budget(), None);

        // Should succeed regardless of size (modulo actual system memory)
        let buf = pool.allocate(1_000_000).unwrap();
        assert_eq!(pool.bytes_in_use(), 1_000_000);
        drop(buf);
        assert_eq!(pool.bytes_in_use(), 0);
    }

    #[test]
    fn tracked_pool_concurrent_drops() {
        let pool = TrackedPool::new(None);

        let mut buffers: Vec<TrackedBuffer<'_>> = Vec::new();
        for _ in 0..100 {
            buffers.push(pool.allocate(1024).unwrap());
        }
        assert_eq!(pool.bytes_in_use(), 100 * 1024);

        // Use scoped threads so buffers (which borrow the pool) can be moved
        // into spawned threads without requiring 'static.
        std::thread::scope(|s| {
            for buf in buffers {
                s.spawn(move || {
                    drop(buf);
                });
            }
        });

        assert_eq!(pool.bytes_in_use(), 0);
    }

    #[test]
    fn tracked_pool_zero_budget() {
        let pool = TrackedPool::new(Some(0));
        // Zero-size succeeds (returns before budget check)
        let buf = pool.allocate(0).unwrap();
        assert_eq!(pool.bytes_in_use(), 0);
        drop(buf);

        // Any non-zero allocation fails
        let err = pool.allocate(1).unwrap_err();
        assert!(matches!(err, AllocationError::BudgetExceeded { .. }));
    }

    #[test]
    fn tracked_pool_concurrent_allocations() {
        let pool = TrackedPool::new(Some(100 * 1024));

        std::thread::scope(|s| {
            for _ in 0..10 {
                s.spawn(|| {
                    let mut bufs = Vec::new();
                    for _ in 0..10 {
                        bufs.push(pool.allocate(1024).unwrap());
                    }
                    // drop all at end of scope
                });
            }
        });

        assert_eq!(pool.bytes_in_use(), 0);
    }

    // -- Pool trait works in generic code --

    fn allocate_and_sum<P: Pool>(pool: &P, size: usize) -> u8 {
        let mut buf = pool.allocate(size).unwrap();
        for (i, byte) in buf.iter_mut().enumerate() {
            *byte = (i % 256) as u8;
        }
        buf.iter().copied().fold(0u8, |a, b| a.wrapping_add(b))
    }

    #[test]
    fn generic_over_pool() {
        let sum_sys = allocate_and_sum(&SystemPool, 256);
        let tracked = TrackedPool::new(None);
        let sum_tracked = allocate_and_sum(&tracked, 256);
        let arc_tracked = ArcTrackedPool::new(None);
        let sum_arc = allocate_and_sum(&arc_tracked, 256);
        assert_eq!(sum_sys, sum_tracked);
        assert_eq!(sum_sys, sum_arc);
    }

    // -- ArcTrackedPool --

    #[test]
    fn arc_tracked_pool_basic_tracking() {
        let pool = ArcTrackedPool::new(None);
        assert_eq!(pool.bytes_in_use(), 0);

        let buf1 = pool.allocate(100).unwrap();
        assert_eq!(pool.bytes_in_use(), 100);

        let buf2 = pool.allocate(200).unwrap();
        assert_eq!(pool.bytes_in_use(), 300);

        drop(buf1);
        assert_eq!(pool.bytes_in_use(), 200);

        drop(buf2);
        assert_eq!(pool.bytes_in_use(), 0);
    }

    #[test]
    fn arc_tracked_pool_clones_share_counter() {
        let pool = ArcTrackedPool::new(None);
        let pool_clone = pool.clone();

        let _buf1 = pool.allocate(100).unwrap();
        // Counter is visible through any clone.
        assert_eq!(pool_clone.bytes_in_use(), 100);

        let _buf2 = pool_clone.allocate(50).unwrap();
        assert_eq!(pool.bytes_in_use(), 150);
    }

    #[test]
    fn arc_tracked_pool_buffer_outlives_pool_handle() {
        // The whole point of ArcTrackedPool: a buffer keeps the pool
        // alive by itself, so dropping the originating handle is fine.
        let pool = ArcTrackedPool::new(None);
        let buf = pool.allocate(64).unwrap();
        assert_eq!(pool.bytes_in_use(), 64);
        drop(pool);
        // `buf` still holds the only live Arc — counter can no longer
        // be observed from outside, but dropping the buffer frees
        // memory cleanly without use-after-free.
        drop(buf);
    }

    #[test]
    fn arc_tracked_pool_budget_enforcement() {
        let pool = ArcTrackedPool::new(Some(256));
        assert_eq!(pool.budget(), Some(256));

        let buf1 = pool.allocate(200).unwrap();
        assert_eq!(pool.bytes_in_use(), 200);

        let err = pool.allocate(100).unwrap_err();
        assert!(matches!(err, AllocationError::BudgetExceeded { .. }));
        assert_eq!(pool.bytes_in_use(), 200);

        drop(buf1);
        let _buf2 = pool.allocate(100).unwrap();
        assert_eq!(pool.bytes_in_use(), 100);
    }

    #[test]
    fn arc_tracked_pool_readwrite() {
        let pool = ArcTrackedPool::new(None);
        let mut buf = pool.allocate(16).unwrap();
        assert!(buf.iter().all(|&b| b == 0));
        buf[0] = 0xAB;
        buf[15] = 0xCD;
        assert_eq!(buf[0], 0xAB);
        assert_eq!(buf[15], 0xCD);
    }

    #[test]
    fn arc_tracked_pool_zero_size() {
        let pool = ArcTrackedPool::new(None);
        let buf = pool.allocate(0).unwrap();
        assert_eq!(buf.len(), 0);
        assert_eq!(pool.bytes_in_use(), 0);
        drop(buf);
        assert_eq!(pool.bytes_in_use(), 0);
    }
}

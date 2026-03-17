//! A sorted collection mapping non-overlapping `u64` ranges to values.
//!
//! Backed by parallel sorted `Vec`s (struct-of-arrays layout) with
//! binary-search lookup — O(log n) for point queries, O(1) amortized
//! insert when entries are added in order.
//!
//! The SoA layout means `values()` returns `&[V]` directly, and the
//! binary search only touches the `starts` array (cache-friendly).

use std::fmt;

/// A map from non-overlapping `u64` ranges to values of type `V`.
///
/// Ranges are half-open: `[start, start + count)`. Lookup by any `u64`
/// within a range returns the associated value and the offset within that
/// range.
///
/// Entries are stored sorted by `start`. Insertions must not overlap
/// existing ranges (checked in debug builds). Appending in order is O(1);
/// out-of-order insertion is O(n) due to the shift, but this is rare in
/// practice — atom IDs are allocated monotonically.
#[derive(Clone)]
pub struct RangeMap<V> {
    starts: Vec<u64>,
    counts: Vec<u64>,
    values: Vec<V>,
}

impl<V> Default for RangeMap<V> {
    fn default() -> Self {
        Self {
            starts: Vec::new(),
            counts: Vec::new(),
            values: Vec::new(),
        }
    }
}

impl<V> RangeMap<V> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            starts: Vec::with_capacity(cap),
            counts: Vec::with_capacity(cap),
            values: Vec::with_capacity(cap),
        }
    }

    /// Insert a range `[start, start+count)` → `value`.
    ///
    /// Panics (debug) if the range overlaps an existing entry or if count is 0.
    pub fn insert(&mut self, start: u64, count: u64, value: V) {
        debug_assert!(count > 0, "RangeMap: count must be > 0");
        let end = start + count;

        // Fast path: appending in order (the common case for atom ID allocation).
        if let Some(&last_start) = self.starts.last() {
            let last_end = last_start + self.counts[self.counts.len() - 1];
            if start >= last_end {
                self.starts.push(start);
                self.counts.push(count);
                self.values.push(value);
                return;
            }
        } else {
            self.starts.push(start);
            self.counts.push(count);
            self.values.push(value);
            return;
        }

        // Slow path: find insertion point and check for overlap.
        let pos = self.starts.partition_point(|&s| s < start);

        // Check overlap with the entry before.
        if pos > 0 {
            debug_assert!(
                self.starts[pos - 1] + self.counts[pos - 1] <= start,
                "RangeMap: overlapping insert [{}, {}) vs existing [{}, {})",
                start,
                end,
                self.starts[pos - 1],
                self.starts[pos - 1] + self.counts[pos - 1],
            );
        }
        // Check overlap with the entry after.
        if pos < self.starts.len() {
            debug_assert!(
                end <= self.starts[pos],
                "RangeMap: overlapping insert [{}, {}) vs existing [{}, {})",
                start,
                end,
                self.starts[pos],
                self.starts[pos] + self.counts[pos],
            );
        }

        self.starts.insert(pos, start);
        self.counts.insert(pos, count);
        self.values.insert(pos, value);
    }

    /// Look up a point `id`, returning `(value, offset)` where
    /// `offset = id - range_start`.
    pub fn get(&self, id: u64) -> Option<(&V, u64)> {
        let idx = self.index_of(id)?;
        Some((&self.values[idx], id - self.starts[idx]))
    }

    /// Mutable version of `get`.
    pub fn get_mut(&mut self, id: u64) -> Option<(&mut V, u64)> {
        let idx = self.index_of(id)?;
        let offset = id - self.starts[idx];
        Some((&mut self.values[idx], offset))
    }

    /// Returns the entry index containing `id`, or `None`.
    fn index_of(&self, id: u64) -> Option<usize> {
        // Binary search: find the last entry whose start <= id.
        let pos = self.starts.partition_point(|&s| s <= id);
        if pos == 0 {
            return None;
        }
        let idx = pos - 1;
        if id < self.starts[idx] + self.counts[idx] {
            Some(idx)
        } else {
            None
        }
    }

    /// Look up by entry index (insertion order). O(1).
    pub fn get_by_index(&self, idx: usize) -> Option<(u64, u64, &V)> {
        if idx < self.starts.len() {
            Some((self.starts[idx], self.counts[idx], &self.values[idx]))
        } else {
            None
        }
    }

    /// Mutable lookup by entry index. O(1).
    pub fn get_by_index_mut(&mut self, idx: usize) -> Option<(u64, u64, &mut V)> {
        if idx < self.starts.len() {
            Some((self.starts[idx], self.counts[idx], &mut self.values[idx]))
        } else {
            None
        }
    }

    /// Find the entry index for a point `id`. O(log n).
    pub fn find_index(&self, id: u64) -> Option<usize> {
        self.index_of(id)
    }

    /// Number of ranges stored.
    pub fn len(&self) -> usize {
        self.starts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.starts.is_empty()
    }

    /// Direct access to the values slice.
    pub fn values(&self) -> &[V] {
        &self.values
    }

    /// Mutable access to the values slice.
    pub fn values_mut(&mut self) -> &mut [V] {
        &mut self.values
    }

    /// Iterate `(start, count, &value)` in sorted order.
    pub fn iter(&self) -> impl Iterator<Item = (u64, u64, &V)> {
        self.starts
            .iter()
            .zip(self.counts.iter())
            .zip(self.values.iter())
            .map(|((&s, &c), v)| (s, c, v))
    }

    /// Iterate `(start, count, &mut value)` in sorted order.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (u64, u64, &mut V)> {
        self.starts
            .iter()
            .zip(self.counts.iter())
            .zip(self.values.iter_mut())
            .map(|((&s, &c), v)| (s, c, v))
    }

    /// Check if `id` falls within any stored range.
    pub fn contains(&self, id: u64) -> bool {
        self.index_of(id).is_some()
    }
}

impl<V: fmt::Debug> fmt::Debug for RangeMap<V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list()
            .entries((0..self.starts.len()).map(|i| {
                format!(
                    "[{}..{}) → {:?}",
                    self.starts[i],
                    self.starts[i] + self.counts[i],
                    self.values[i]
                )
            }))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty() {
        let m: RangeMap<&str> = RangeMap::new();
        assert!(m.is_empty());
        assert_eq!(m.len(), 0);
        assert!(m.get(0).is_none());
        assert!(!m.contains(0));
    }

    #[test]
    fn single_range() {
        let mut m = RangeMap::new();
        m.insert(10, 5, "hello");
        assert_eq!(m.len(), 1);
        assert!(!m.contains(9));
        assert_eq!(m.get(10), Some((&"hello", 0)));
        assert_eq!(m.get(12), Some((&"hello", 2)));
        assert_eq!(m.get(14), Some((&"hello", 4)));
        assert!(!m.contains(15));
    }

    #[test]
    fn sequential_insert() {
        let mut m = RangeMap::new();
        m.insert(0, 100, "a");
        m.insert(100, 200, "b");
        m.insert(300, 50, "c");

        assert_eq!(m.get(0), Some((&"a", 0)));
        assert_eq!(m.get(99), Some((&"a", 99)));
        assert_eq!(m.get(100), Some((&"b", 0)));
        assert_eq!(m.get(299), Some((&"b", 199)));
        assert_eq!(m.get(300), Some((&"c", 0)));
        assert_eq!(m.get(349), Some((&"c", 49)));
        assert!(m.get(350).is_none());
    }

    #[test]
    fn gap_between_ranges() {
        let mut m = RangeMap::new();
        m.insert(0, 10, "a");
        m.insert(20, 10, "b");

        assert_eq!(m.get(5), Some((&"a", 5)));
        assert!(m.get(10).is_none());
        assert!(m.get(15).is_none());
        assert!(m.get(19).is_none());
        assert_eq!(m.get(20), Some((&"b", 0)));
    }

    #[test]
    fn out_of_order_insert() {
        let mut m = RangeMap::new();
        m.insert(100, 10, "b");
        m.insert(0, 10, "a");
        m.insert(50, 10, "mid");

        assert_eq!(m.get(5), Some((&"a", 5)));
        assert_eq!(m.get(55), Some((&"mid", 5)));
        assert_eq!(m.get(105), Some((&"b", 5)));
        assert_eq!(m.len(), 3);
    }

    #[test]
    fn get_mut() {
        let mut m = RangeMap::new();
        m.insert(0, 10, 0u32);
        if let Some((val, offset)) = m.get_mut(5) {
            assert_eq!(offset, 5);
            *val = 42;
        }
        assert_eq!(m.get(5), Some((&42u32, 5)));
    }

    #[test]
    fn find_index_and_get_by_index() {
        let mut m = RangeMap::new();
        m.insert(0, 100, "first");
        m.insert(100, 100, "second");
        m.insert(200, 100, "third");

        assert_eq!(m.find_index(50), Some(0));
        assert_eq!(m.find_index(150), Some(1));
        assert_eq!(m.find_index(250), Some(2));
        assert_eq!(m.find_index(300), None);

        let (start, count, val) = m.get_by_index(1).unwrap();
        assert_eq!(start, 100);
        assert_eq!(count, 100);
        assert_eq!(*val, "second");
    }

    #[test]
    fn values_slice() {
        let mut m = RangeMap::new();
        m.insert(0, 10, "a");
        m.insert(10, 20, "b");
        m.insert(30, 5, "c");

        assert_eq!(m.values(), &["a", "b", "c"]);
        assert_eq!(m.values().len(), 3);
    }

    #[test]
    fn iter() {
        let mut m = RangeMap::new();
        m.insert(0, 10, "a");
        m.insert(10, 20, "b");
        m.insert(30, 5, "c");

        let collected: Vec<_> = m.iter().collect();
        assert_eq!(collected.len(), 3);
        assert_eq!(collected[0], (0, 10, &"a"));
        assert_eq!(collected[1], (10, 20, &"b"));
        assert_eq!(collected[2], (30, 5, &"c"));
    }

    #[test]
    fn large_sequential() {
        let mut m = RangeMap::new();
        for i in 0..10_000u64 {
            m.insert(i * 100, 100, i);
        }
        assert_eq!(m.len(), 10_000);
        assert_eq!(m.values().len(), 10_000);
        // Spot checks.
        assert_eq!(m.get(0), Some((&0, 0)));
        assert_eq!(m.get(50), Some((&0, 50)));
        assert_eq!(m.get(100), Some((&1, 0)));
        assert_eq!(m.get(999_950), Some((&9999, 50)));
        assert!(m.get(1_000_000).is_none());
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "overlapping")]
    fn overlap_panics() {
        let mut m = RangeMap::new();
        m.insert(0, 10, "a");
        m.insert(5, 10, "b"); // overlaps [0, 10)
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "count must be > 0")]
    fn zero_count_panics() {
        let mut m: RangeMap<()> = RangeMap::new();
        m.insert(0, 0, ());
    }

    #[test]
    fn adjacent_ranges() {
        let mut m = RangeMap::new();
        m.insert(0, 10, "a");
        m.insert(10, 10, "b"); // immediately adjacent, not overlapping
        assert_eq!(m.get(9), Some((&"a", 9)));
        assert_eq!(m.get(10), Some((&"b", 0)));
    }
}

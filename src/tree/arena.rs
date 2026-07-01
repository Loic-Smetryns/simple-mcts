use std::num::NonZeroU32;

/// A contiguous memory arena for cache-friendly data storage.
///
/// The `Arena` allows you to allocate items in a single underlying `Vec`,
/// returning a strongly typed [`ArenaIndex`] instead of a standard reference.
/// This approach avoids complex lifetime management, prevents memory fragmentation,
/// and guarantees excellent data locality.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Arena<T>{
    storage: Vec<T>
}

/// A strongly typed, memory-optimized index into an [`Arena`].
///
/// Under the hood, this uses a `NonZeroU32` shifted by 1. This guarantees that
/// the index itself takes 4 bytes, and an `Option<ArenaIndex>` also takes exactly
/// 4 bytes thanks to Rust's null-pointer optimization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ArenaIndex(NonZeroU32);

impl ArenaIndex {
    /// Creates a new `ArenaIndex` from a standard `usize`.
    ///
    /// # Arguments
    ///
    /// * `n` - The 1-based internal index (usually `Vec::len()`).
    ///
    /// # Panics
    ///
    /// Panics if `n` exceeds `u32::MAX`, meaning the arena has reached its absolute maximum capacity.
    /// Also panics if `n` is `0`, as the internal `NonZeroU32` cannot be null.
    pub fn from_usize(n: usize) -> Self {
        let index = u32::try_from(n).expect("Arena capacity exceeded (u32::MAX)");
        ArenaIndex(NonZeroU32::new(index).expect("The index 0 is not valid"))
    }

    /// Converts the `ArenaIndex` back into a 0-based `usize` for `Vec` indexing.
    ///
    /// # Returns
    ///
    /// The standard `usize` index corresponding to the actual position in the arena's storage.
    pub fn to_usize(&self) -> usize {
        (self.0.get() - 1) as usize
    }
}

impl <T> Arena<T>{
    /// Creates a new, empty `Arena`.
    ///
    /// The arena will not allocate memory until elements are pushed onto it.
    ///
    /// # Returns
    ///
    /// A new `Arena<T>` instance.
    #[must_use]
    pub fn new() -> Arena<T>{
        Arena{ storage: Vec::new() }
    }

    /// Creates a new, empty `Arena` with at least the specified capacity.
    ///
    /// The arena will be able to hold `capacity` elements without reallocating.
    ///
    /// # Arguments
    ///
    /// * `capacity` - The initial capacity of the underlying storage.
    ///
    /// # Returns
    ///
    /// A new `Arena<T>` instance pre-allocated with the given capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Arena<T> {
        Arena{ storage: Vec::with_capacity(capacity) }
    }

    /// Allocates a new element in the arena and returns its index.
    ///
    /// # Arguments
    ///
    /// * `data` - The data to be inserted into the arena.
    ///
    /// # Returns
    ///
    /// An [`ArenaIndex`] that uniquely identifies the inserted element.
    ///
    /// # Panics
    ///
    /// Panics if the total number of elements exceeds `u32::MAX - 1`.
    pub fn alloc(&mut self, data: T) -> ArenaIndex{
        self.storage.push(data);
        ArenaIndex::from_usize(self.storage.len())
    }

    /// Returns a reference to the element at the given index, or `None` if out of bounds.
    ///
    /// # Arguments
    ///
    /// * `index` - The [`ArenaIndex`] of the element to retrieve.
    ///
    /// # Returns
    ///
    /// `Some(&T)` if the index is valid, otherwise `None`.
    pub fn get(&self, index: ArenaIndex) -> Option<&T>{
        self.storage.get(index.to_usize())
    }

    /// Returns a mutable reference to the element at the given index, or `None` if out of bounds.
    ///
    /// # Arguments
    ///
    /// * `index` - The [`ArenaIndex`] of the element to retrieve.
    ///
    /// # Returns
    ///
    /// `Some(&mut T)` if the index is valid, otherwise `None`.
    pub fn get_mut(&mut self, index: ArenaIndex) -> Option<&mut T>{
        self.storage.get_mut(index.to_usize())
    }

    /// Clears the arena, removing all values.
    ///
    /// Note that this method has no effect on the allocated capacity of the underlying storage.
    /// All previously issued [`ArenaIndex`]es will become invalid or point to wrong/new data
    /// if the arena is reused.
    pub fn clear(&mut self){
        self.storage.clear()
    }

    /// Returns the number of elements currently allocated in the arena.
    ///
    /// # Returns
    ///
    /// A `usize` representing the total count of items in the underlying storage.
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    /// Returns the total number of elements the arena can hold without reallocating.
    ///
    /// # Returns
    ///
    /// A `usize` representing the current capacity of the underlying storage.
    pub fn capacity(&self) -> usize{
        self.storage.capacity()
    }
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T : Default> Arena<T>{
    /// Allocates a new default-initialized element in the arena and returns its index.
    ///
    /// # Returns
    ///
    /// An [`ArenaIndex`] pointing to the newly inserted `T::default()`.
    #[allow(dead_code)]
    pub fn alloc_default(&mut self) -> ArenaIndex{
        self.storage.push(T::default());
        ArenaIndex::from_usize(self.storage.len())
    }
}

impl<T> std::ops::Index<ArenaIndex> for Arena<T> {
    type Output = T;

    /// Returns a reference to the element at the given index.
    ///
    /// # Panics
    ///
    /// Panics if the `index` is out of bounds (which theoretically shouldn't happen
    /// unless an index from a different or cleared arena is used).
    fn index(&self, index: ArenaIndex) -> &Self::Output {
        &self.storage[index.to_usize()]
    }
}

impl<T> std::ops::IndexMut<ArenaIndex> for Arena<T> {
    /// Returns a mutable reference to the element at the given index.
    ///
    /// # Panics
    ///
    /// Panics if the `index` is out of bounds.
    fn index_mut(&mut self, index: ArenaIndex) -> &mut Self::Output {
        &mut self.storage[index.to_usize()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_index_size() {
        assert_eq!(size_of::<ArenaIndex>(), size_of::<Option<ArenaIndex>>());
        assert_eq!(size_of::<ArenaIndex>(), 4);
    }

    #[test]
    fn test_new_and_len() {
        let arena: Arena<i32> = Arena::new();
        assert_eq!(arena.len(), 0);
        assert_eq!(arena.capacity(), 0);
    }

    #[test]
    fn test_default_and_len() {
        let arena: Arena<i32> = Arena::default();
        assert_eq!(arena.len(), 0);
        assert_eq!(arena.capacity(), 0);
    }

    #[test]
    fn test_with_capacity() {
        let arena: Arena<i32> = Arena::with_capacity(100);
        assert_eq!(arena.len(), 0);
        assert!(arena.capacity() >= 100);
    }

    #[test]
    fn test_alloc_and_indexes() {
        let mut arena = Arena::new();
        let id1 = arena.alloc(10);
        let id2 = arena.alloc(20);

        assert_eq!(arena.len(), 2);
        assert_eq!(id1.to_usize(), 0);
        assert_eq!(id2.to_usize(), 1);
    }

    #[test]
    fn test_alloc_default() {
        let mut arena: Arena<i32> = Arena::new();
        let id = arena.alloc_default();

        assert_eq!(arena.len(), 1);
        assert_eq!(arena.get(id), Some(&0));
    }

    #[test]
    fn test_get_valid_and_invalid() {
        let mut arena = Arena::new();
        let id = arena.alloc(42);

        assert_eq!(arena.get(id), Some(&42));

        let out_of_bounds_id = ArenaIndex::from_usize(5);
        assert_eq!(arena.get(out_of_bounds_id), None);
    }

    #[test]
    fn test_get_mut_valid_and_invalid() {
        let mut arena = Arena::new();
        let id = arena.alloc(42);

        if let Some(value) = arena.get_mut(id) {
            *value = 100;
        }
        assert_eq!(arena.get(id), Some(&100));

        let out_of_bounds_id = ArenaIndex::from_usize(5);
        assert_eq!(arena.get_mut(out_of_bounds_id), None);
    }

    #[test]
    fn test_index_operator() {
        let mut arena = Arena::new();
        let id = arena.alloc("Test");
        assert_eq!(arena[id], "Test");
    }

    #[test]
    fn test_index_mut_operator() {
        let mut arena = Arena::new();
        let id = arena.alloc("Test");
        arena[id] = "Modifié";
        assert_eq!(arena[id], "Modifié");
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_index_out_of_bounds_panics() {
        let arena: Arena<i32> = Arena::new();
        let fake_id = ArenaIndex::from_usize(1);
        let _val = arena[fake_id]; // Doit paniquer
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_index_mut_out_of_bounds_panics() {
        let mut arena: Arena<i32> = Arena::new();
        let fake_id = ArenaIndex::from_usize(1);
        arena[fake_id] = 10;
    }

    #[test]
    fn test_clear_keeps_capacity() {
        let mut arena = Arena::with_capacity(50);
        arena.alloc(1);
        arena.alloc(2);

        let cap_before = arena.capacity();
        assert!(cap_before >= 50);

        arena.clear();

        assert_eq!(arena.len(), 0);
        assert_eq!(arena.capacity(), cap_before);
    }

    #[test]
    fn test_arena_index_conversions() {
        let id = ArenaIndex::from_usize(1);
        assert_eq!(id.to_usize(), 0);

        let id_large = ArenaIndex::from_usize(100);
        assert_eq!(id_large.to_usize(), 99);
    }

    #[test]
    #[should_panic(expected = "The index 0 is not valid")]
    fn test_arena_index_zero_panics() {
        let _ = ArenaIndex::from_usize(0);
    }

    #[test]
    #[should_panic(expected = "Arena capacity exceeded")]
    fn test_arena_index_overflow_panics() {
        let overflow_value = (u32::MAX as usize) + 1;
        let _ = ArenaIndex::from_usize(overflow_value);
    }

    #[test]
    fn test_derives_and_equality() {
        let mut arena1 = Arena::new();
        let mut arena2 = Arena::new();

        arena1.alloc(42);
        arena2.alloc(42);

        assert_eq!(arena1, arena2);

        let arena_cloned = arena1.clone();
        assert_eq!(arena1, arena_cloned);

        arena1.alloc(10);
        assert_ne!(arena1, arena2);
    }
}
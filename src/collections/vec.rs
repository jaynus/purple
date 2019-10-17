use super::alloc::{Layout, UnstableLayoutMethods};
use crate::*;
use derivative::Derivative;
use std::iter::FromIterator;
use std::marker::PhantomData;

pub struct Vec<'a, T> {
    arena: &'a Arena,
    buffer: ScopedRawBuffer<'a>,
    capacity: usize,
    size: usize,
    _marker: PhantomData<T>,
}
impl<'a, T> Vec<'a, T> {
    pub fn with_capacity_in(arena: &'a Arena, capacity: usize) -> Self {
        let buffer = arena.alloc_scoped_raw(Layout::array::<T>(capacity).unwrap());

        Self {
            arena,
            buffer,
            capacity,
            size: 0,
            _marker: Default::default(),
        }
    }

    pub fn from_std_in(arena: &'a Arena, v: std::vec::Vec<T>) -> Self {
        let mut r = Self::with_capacity_in(arena, v.len());
        v.into_iter().for_each(|i| {
            r.push(i).unwrap();
        });
        r
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.size
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.size > 0
    }

    pub fn push(&mut self, value: T) -> Result<(), ArenaError> {
        if self.size >= self.capacity {
            return Err(ArenaError::CapacityExceeded);
        }

        let write_ptr = unsafe {
            self.buffer
                .ptr
                .as_ptr()
                .add(self.size * std::mem::size_of::<T>())
        };

        unsafe {
            std::ptr::copy_nonoverlapping(&value as *const T, write_ptr as *mut T, 1);
        }

        self.size += 1;

        Ok(())
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut T {
        self.buffer.ptr.as_ptr().cast()
    }

    pub(crate) fn as_ptr(&mut self) -> *mut T {
        self.buffer.ptr.as_ptr().cast()
    }

    pub fn remove(&mut self, index: usize) -> T {
        let len = self.len();
        assert!(index < len);
        unsafe {
            // infallible
            let ret;
            {
                // the place we are taking from.
                let ptr = self.as_mut_ptr().add(index);
                // copy it out, unsafely having a copy of the value on
                // the stack and in the vector at the same time.
                ret = std::ptr::read(ptr);

                // Shift everything down to fill in that spot.
                std::ptr::copy(ptr.offset(1), ptr, len - index - 1);
            }
            self.size = len - 1;
            ret
        }
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.len() == 0 {
            None
        } else {
            unsafe {
                self.size -= 1;
                //Some(self.get(self.len()).unwrap()) TODO: Return old values
                None
            }
        }
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.size {
            return None;
        }

        Some(unsafe {
            &*(self
                .buffer
                .ptr
                .as_ptr()
                .add(index * std::mem::size_of::<T>()) as *const T)
        })
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index >= self.size {
            return None;
        }

        Some(unsafe {
            &mut *(self
                .buffer
                .ptr
                .as_ptr()
                .add(index * std::mem::size_of::<T>()) as *mut T)
        })
    }

    pub fn iter(&self) -> VecIter<'_, T> {
        VecIter::new(self)
    }
    pub fn iter_mut(&mut self) -> VecIterMut<'_, T> {
        unsafe {
            VecIterMut {
                begin: self.buffer.ptr.as_ptr().cast(),
                ptr: self.buffer.ptr.as_ptr().cast(),
                end: self
                    .buffer
                    .ptr
                    .as_ptr()
                    .add(self.size * std::mem::size_of::<T>())
                    .cast(),
                _marker: Default::default(),
            }
        }
    }

    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts::<T>(self.buffer.ptr.as_ptr().cast(), self.size) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut::<T>(self.buffer.ptr.as_ptr().cast(), self.size) }
    }

    pub fn truncate(&mut self, len: usize) {
        let current_len = self.len();
        unsafe {
            let mut ptr = self.as_mut_ptr().add(self.len());
            // Set the final length at the end, keeping in mind that
            // dropping an element might panic. Works around a missed
            // optimization, as seen in the following issue:
            // https://github.com/rust-lang/rust/issues/51802
            let mut local_len = SetLenOnDrop::new(&mut self.size);

            // drop any extra elements
            for _ in len..current_len {
                local_len.decrement_len(1);
                ptr = ptr.offset(-1);
                std::ptr::drop_in_place(ptr);
            }
        }
    }
}

impl<'a, T> AsRef<Vec<'a, T>> for Vec<'a, T> {
    fn as_ref(&self) -> &Vec<'a, T> {
        self
    }
}

impl<'a, T> AsMut<Vec<'a, T>> for Vec<'a, T> {
    fn as_mut(&mut self) -> &mut Vec<'a, T> {
        self
    }
}

impl<'a, T> AsRef<[T]> for Vec<'a, T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<'a, T> AsMut<[T]> for Vec<'a, T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<'a, T> Drop for Vec<'a, T> {
    fn drop(&mut self) {
        for i in 0..self.size {
            unsafe {
                std::intrinsics::drop_in_place(
                    self.buffer.ptr.as_ptr().add(i * std::mem::size_of::<T>()) as *mut T,
                );
            }
        }
    }
}

impl<'a, T: std::fmt::Debug> std::fmt::Debug for Vec<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.as_slice())
    }
}

impl<'a, T: PartialEq> PartialEq for Vec<'a, T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<'a, T: PartialEq> PartialEq<std::vec::Vec<T>> for Vec<'a, T> {
    fn eq(&self, other: &std::vec::Vec<T>) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<'a, T: PartialEq> PartialEq<[T]> for Vec<'a, T> {
    fn eq(&self, other: &[T]) -> bool {
        self.as_slice() == other
    }
}

impl<'a, T: PartialEq> Vec<'a, T> {
    #[inline]
    pub fn dedup_by_key<F, K>(&mut self, mut key: F)
    where
        F: FnMut(&mut T) -> K,
        K: PartialEq,
    {
        self.dedup_by(|a, b| key(a) == key(b))
    }

    pub fn dedup_by<F>(&mut self, same_bucket: F)
    where
        F: FnMut(&mut T, &mut T) -> bool,
    {
        let len = {
            let (dedup, _) = partition_dedup_by(self.as_mut_slice(), same_bucket);
            dedup.len()
        };
        self.truncate(len);
    }

    pub fn remove_item(&mut self, item: &T) -> Option<T> {
        let pos = self.iter().position(|x| *x == *item)?;
        Some(self.remove(pos))
    }
}

pub struct VecIter<'a, T> {
    inner: &'a Vec<'a, T>,
    cur: usize,
}

impl<'a, T> VecIter<'a, T> {
    pub fn new(inner: &'a Vec<'a, T>) -> Self {
        Self { inner, cur: 0 }
    }
}

impl<'a, T> Iterator for VecIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.cur;
        if self.cur >= self.inner.size {
            return None;
        }
        self.cur += 1;

        self.inner.get(next)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.inner.size))
    }
}

impl<'a, T> DoubleEndedIterator for VecIter<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let next = self.cur;

        if self.cur.checked_sub(1).is_none() {
            return None;
        }

        self.cur -= 1;

        self.inner.get(next)
    }
}

pub struct VecIterMut<'a, T> {
    begin: *mut T,
    ptr: *mut T,
    end: *mut T,
    _marker: PhantomData<&'a T>,
}

impl<'a, T> Iterator for VecIterMut<'a, T> {
    type Item = &'a mut T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let fetch = self.ptr;
        if fetch >= self.end {
            return None;
        }

        self.ptr = unsafe { self.ptr.add(std::mem::size_of::<T>()) };

        Some(unsafe { &mut *fetch.cast() })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (
            0,
            Some(unsafe { self.end.sub(self.ptr as usize) as usize / std::mem::size_of::<T>() }),
        )
    }
}

impl<'a, T> DoubleEndedIterator for VecIterMut<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let fetch = self.ptr;
        if self.ptr < self.begin {
            return None;
        }

        self.ptr = unsafe { self.ptr.sub(std::mem::size_of::<T>()) };

        Some(unsafe { &mut *fetch.cast() })
    }
}

pub trait FromIteratorIn<'a, T> {
    fn from_iter_in<I: 'a + IntoIterator<Item = T> + ExactSizeIterator>(
        arena: &'a Arena,
        iter: I,
    ) -> Self;
}
impl<'a, T> FromIteratorIn<'a, T> for Vec<'a, T> {
    fn from_iter_in<I: IntoIterator<Item = T> + ExactSizeIterator>(
        arena: &'a Arena,
        iter: I,
    ) -> Self {
        let mut c = Vec::with_capacity_in(&arena, iter.len());

        for i in iter {
            c.push(i);
        }

        c
    }
}

pub trait CollectIn<'a, T> {
    fn collect_in<B: FromIteratorIn<'a, T>>(self, arena: &'a Arena) -> B;
}
impl<'a, T: Sized, I> CollectIn<'a, T> for I
where
    I: 'a + Iterator<Item = T> + ExactSizeIterator,
{
    fn collect_in<B: FromIteratorIn<'a, T>>(self, arena: &'a Arena) -> B {
        B::from_iter_in(arena, self)
    }
}

pub(crate) fn partition_dedup_by<T: PartialEq, F>(
    slice: &mut [T],
    mut same_bucket: F,
) -> (&mut [T], &mut [T])
where
    F: FnMut(&mut T, &mut T) -> bool,
{
    // Although we have a mutable reference to `self`, we cannot make
    // *arbitrary* changes. The `same_bucket` calls could panic, so we
    // must ensure that the slice is in a valid state at all times.
    //
    // The way that we handle this is by using swaps; we iterate
    // over all the elements, swapping as we go so that at the end
    // the elements we wish to keep are in the front, and those we
    // wish to reject are at the back. We can then split the slice.
    // This operation is still O(n).
    //
    // Example: We start in this state, where `r` represents "next
    // read" and `w` represents "next_write`.
    //
    //           r
    //     +---+---+---+---+---+---+
    //     | 0 | 1 | 1 | 2 | 3 | 3 |
    //     +---+---+---+---+---+---+
    //           w
    //
    // Comparing self[r] against self[w-1], this is not a duplicate, so
    // we swap self[r] and self[w] (no effect as r==w) and then increment both
    // r and w, leaving us with:
    //
    //               r
    //     +---+---+---+---+---+---+
    //     | 0 | 1 | 1 | 2 | 3 | 3 |
    //     +---+---+---+---+---+---+
    //               w
    //
    // Comparing self[r] against self[w-1], this value is a duplicate,
    // so we increment `r` but leave everything else unchanged:
    //
    //                   r
    //     +---+---+---+---+---+---+
    //     | 0 | 1 | 1 | 2 | 3 | 3 |
    //     +---+---+---+---+---+---+
    //               w
    //
    // Comparing self[r] against self[w-1], this is not a duplicate,
    // so swap self[r] and self[w] and advance r and w:
    //
    //                       r
    //     +---+---+---+---+---+---+
    //     | 0 | 1 | 2 | 1 | 3 | 3 |
    //     +---+---+---+---+---+---+
    //                   w
    //
    // Not a duplicate, repeat:
    //
    //                           r
    //     +---+---+---+---+---+---+
    //     | 0 | 1 | 2 | 3 | 1 | 3 |
    //     +---+---+---+---+---+---+
    //                       w
    //
    // Duplicate, advance r. End of slice. Split at w.

    let len = slice.len();
    if len <= 1 {
        return (slice, &mut []);
    }

    let ptr = slice.as_mut_ptr();
    let mut next_read: usize = 1;
    let mut next_write: usize = 1;

    unsafe {
        // Avoid bounds checks by using raw pointers.
        while next_read < len {
            let ptr_read = ptr.add(next_read);
            let prev_ptr_write = ptr.add(next_write - 1);
            if !same_bucket(&mut *ptr_read, &mut *prev_ptr_write) {
                if next_read != next_write {
                    let ptr_write = prev_ptr_write.offset(1);
                    std::mem::swap(&mut *ptr_read, &mut *ptr_write);
                }
                next_write += 1;
            }
            next_read += 1;
        }
    }

    slice.split_at_mut(next_write)
}

// Set the length of the vec when the `SetLenOnDrop` value goes out of scope.
//
// The idea is: The length field in SetLenOnDrop is a local variable
// that the optimizer will see does not alias with any stores through the Vec's data
// pointer. This is a workaround for alias analysis issue #32155
struct SetLenOnDrop<'a> {
    len: &'a mut usize,
    local_len: usize,
}

impl<'a> SetLenOnDrop<'a> {
    #[inline]
    fn new(len: &'a mut usize) -> Self {
        SetLenOnDrop {
            local_len: *len,
            len: len,
        }
    }

    #[inline]
    fn increment_len(&mut self, increment: usize) {
        self.local_len += increment;
    }

    #[inline]
    fn decrement_len(&mut self, decrement: usize) {
        self.local_len -= decrement;
    }
}

impl Drop for SetLenOnDrop<'_> {
    #[inline]
    fn drop(&mut self) {
        *self.len = self.local_len;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    pub struct TestType {
        value: i32,
    }

    #[test]
    fn from_iter_in() {
        let mut arena = Arena::with_capacity(5_242_880);

        let test_values = [
            TestType { value: 1 },
            TestType { value: 2 },
            TestType { value: 3 },
        ];

        let vec = Vec::from_iter_in(&arena, test_values.into_iter());
        assert_eq!(test_values.len(), vec.iter().count());
        vec.iter()
            .enumerate()
            .for_each(|(i, v)| assert_eq!(**v, *test_values.get(i).unwrap()));

        let collected = test_values
            .iter()
            .map(|v| *v)
            .collect_in::<Vec<'_, TestType>>(&arena);
        assert_eq!(test_values.len(), collected.iter().count());
        collected
            .iter()
            .enumerate()
            .for_each(|(i, v)| assert_eq!(*v, *test_values.get(i).unwrap()));
    }

    #[test]
    fn iterators() {
        let mut arena = Arena::with_capacity(5_242_880);
        let mut vec = Vec::with_capacity_in(&arena, 500);

        let test_values = [
            TestType { value: 1 },
            TestType { value: 2 },
            TestType { value: 3 },
        ];

        test_values.into_iter().for_each(|v| vec.push(*v).unwrap());

        vec.iter()
            .enumerate()
            .for_each(|(i, v)| assert_eq!(*v, *test_values.get(i).unwrap()));

        vec.iter_mut()
            .enumerate()
            .for_each(|(i, v)| assert_eq!(*v, *test_values.get(i).unwrap()));

        // Test reverse iteration
        vec.iter()
            .rev()
            .enumerate()
            .for_each(|(i, v)| assert_eq!(*v, *test_values.get(i).unwrap()));

        vec.iter_mut()
            .rev()
            .enumerate()
            .for_each(|(i, v)| assert_eq!(*v, *test_values.get(i).unwrap()));

        assert_eq!(test_values.len(), vec.iter().count());
        assert_eq!(test_values.len(), vec.iter().count());

        //assert_eq!(test_values.len(), vec.iter().rev().count());
        //assert_eq!(test_values.len(), vec.iter_mut().rev().count());
    }

    #[test]
    fn push_read_vec() {
        let mut arena = Arena::with_capacity(5_242_880);
        let mut vec = Vec::with_capacity_in(&arena, 500);

        let test_values = [
            TestType { value: 1 },
            TestType { value: 2 },
            TestType { value: 3 },
        ];

        test_values.into_iter().for_each(|v| vec.push(*v).unwrap());
        test_values
            .into_iter()
            .enumerate()
            .for_each(|(i, v)| assert_eq!(*v, *vec.get(i).unwrap()));
    }
}

use super::alloc::{Layout, UnstableLayoutMethods};
use crate::*;
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

    pub fn as_slice_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut::<T>(self.buffer.ptr.as_ptr().cast(), self.size) }
    }
}

impl<'a, T> AsRef<[T]> for Vec<'a, T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<'a, T> AsMut<[T]> for Vec<'a, T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.as_slice_mut()
    }
}

impl<'a, T> Drop for Vec<'a, T> {
    fn drop(&mut self) {
        for i in (0..self.size) {
            unsafe {
                std::intrinsics::drop_in_place(
                    self.buffer.ptr.as_ptr().add(i * std::mem::size_of::<T>()) as *mut T,
                );
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    pub struct TestType {
        value: i32,
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

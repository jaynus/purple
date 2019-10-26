use super::alloc::{Layout, UnstableLayoutMethods};
use crate::*;

pub struct BitVec<'a> {
    arena: &'a Arena,
    buffer: ScopedRawBuffer<'a>,
    capacity: usize,
    size: usize,
}
impl<'a> BitVec<'a> {
    /// Capacity for a BitVec is in bytes
    /// However, it must be 32-bit aligned as we internally store in integers
    pub fn with_capacity_in(arena: &'a Arena, capacity: usize) -> Self {
        assert!(capacity % 4 == 0);

        let buffer = arena.alloc_scoped_raw(Layout::array::<u32>(capacity / 4).unwrap());

        Self {
            arena,
            buffer,
            capacity,
            size: 0,
        }
    }

    #[inline]
    pub unsafe fn set_unchecked(&mut self, index: usize, value: bool) {
        let (offset, bit) = calc_bit(index);

        let ptr = self.as_mut_ptr().add(offset);
        if value {
            *ptr |= 1 << bit;
        } else {
            *ptr &= (1 << bit);
        }
    }

    #[inline]
    pub unsafe fn toggle_unchecked(&mut self, index: usize) {
        let (offset, bit) = calc_bit(index);

        let ptr = self.as_mut_ptr().add(offset);
        *ptr ^= (1 << bit);
    }

    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> bool {
        let (offset, bit) = calc_bit(index);

        let ptr = self.as_ptr().add(offset);
        ((*ptr >> bit) & 1) != 0
    }

    #[inline]
    pub fn set(&mut self, index: usize, value: bool) -> Result<(), ArenaError> {
        if check_bounds(index, self.capacity) {
            Ok(unsafe { self.set_unchecked(index, value) })
        } else {
            Err(ArenaError::OutOfBounds)
        }
    }

    #[inline]
    pub fn toggle(&mut self, index: usize) -> Result<(), ArenaError> {
        if check_bounds(index, self.capacity) {
            Ok(unsafe { self.toggle_unchecked(index) })
        } else {
            Err(ArenaError::OutOfBounds)
        }
    }

    #[inline]
    pub fn get(&self, index: usize) -> Option<bool> {
        if check_bounds(index, self.capacity) {
            Some(unsafe { self.get_unchecked(index) })
        } else {
            None
        }
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    #[inline]
    //pub fn len(&self) -> usize {
    //    self.size
    //}

    pub(crate) fn as_mut_ptr(&mut self) -> *mut u32 {
        self.buffer.ptr.as_ptr().cast()
    }

    pub(crate) fn as_ptr(&self) -> *const u32 {
        self.buffer.ptr.as_ptr().cast()
    }
}

#[inline(always)]
fn check_bounds(index: usize, capacity: usize) -> bool {
    let (offset, _) = calc_bit(index);

    if offset >= capacity {
        false
    } else {
        true
    }
}

#[inline(always)]
// Calculate a bit index to a u32 offset + shift
fn calc_bit(index: usize) -> (usize, u32) {
    let bit = index % 32;

    ((index / 32), bit as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_bitvec(size: usize, arena: &Arena) {
        let mut vec = BitVec::with_capacity_in(&arena, size);
        for n in 0..32 {
            assert!(vec.get(n).unwrap() == false);
        }

        for n in 0..32 {
            vec.set(n, true).unwrap();
            assert!(vec.get(n).unwrap());
        }

        for n in 0..32 {
            vec.toggle(n).unwrap();
            assert!(vec.get(n).unwrap() == false);
        }

        // OutOfBounds
        assert!(vec.get(size * 32 * 2).is_none());
        assert!(vec.set(size * 32 * 2, true).is_err());
    }

    #[test]
    fn size_cases() {
        let arena = Arena::with_capacity(5_242_880);
        test_bitvec(4, &arena);
        test_bitvec(32, &arena);
        test_bitvec(64, &arena);
    }
}

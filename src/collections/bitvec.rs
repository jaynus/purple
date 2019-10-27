use super::alloc::*;
use crate::*;

#[cfg(feature = "nightly")]
use packed_simd::u32x4;

use std::ops::{BitAnd, BitOr, Index};

pub struct BitVec<'a> {
    arena: &'a Arena,
    buffer: ScopedRawBuffer<'a>,
    capacity: usize,
    size: usize,
}
impl<'a> BitVec<'a> {
    /// Capacity for a BitVec is in BITS
    /// However, it must be 32-bit aligned as we internally store in integers
    pub fn with_capacity_in(arena: &'a Arena, mut capacity: usize) -> Self {
        let intrin_size = std::mem::size_of::<u32>() * 4;
        let step_bits = intrin_size * 8;
        capacity = (capacity + step_bits) - (capacity % step_bits);

        let internal_capacity = capacity / 8;

        let mut layout = Layout::from_size_align(internal_capacity, intrin_size).unwrap();

        let buffer = arena.alloc_scoped_raw(layout);

        Self {
            arena,
            buffer,
            capacity,
            size: 0,
        }
    }

    pub fn as_mut_slice(&mut self) -> &[u32] {
        unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr(), self.capacity() / 32) }
    }

    pub fn as_slice(&self) -> &[u32] {
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.capacity() / 32) }
    }

    pub fn or_subslice(&mut self, dst: usize, src: usize, len: usize) {
        const BITS_PER_SIMD: usize = (std::mem::size_of::<u32>() * 4) * 8;

        if len % BITS_PER_SIMD == 0 {
            #[cfg(features = "nightly")]
            {
                unsafe {
                    let left_ptr: *mut u32x4 = self.as_mut_ptr().add(dst / 32).cast();
                    let right_ptr: *const u32x4 = self.as_ptr().add(src / 32).cast();
                    *left_ptr = *left_ptr | *right_ptr;
                }

                return;
            }

            #[target_feature(enable = "sse2")]
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                #[cfg(target_arch = "x86")]
                use std::arch::x86::*;
                #[cfg(target_arch = "x86_64")]
                use std::arch::x86_64::*;

                unsafe {
                    let left_ptr: *mut __m128i = self.as_mut_ptr().add(dst / 32).cast();
                    let right_ptr: *const __m128i = self.as_ptr().add(src / 32).cast();

                    *left_ptr = _mm_or_si128(*left_ptr, *right_ptr);
                }
                return;
            }
        }

        let slice = self.buffer.as_mut_slice::<u32>();

        let dst = dst / 32;
        let src = src / 32;

        let mut bit = 0;
        while bit < len {
            let src_idx = src + (bit / 32);
            let dst_idx = dst + (bit / 32);
            if len - bit >= 32 {
                slice[dst_idx] = slice[dst_idx] | slice[src_idx];
                bit += 32;
            } else {
                slice[dst_idx] = slice[dst_idx] >> bit as u32 | slice[src_idx];
                bit += 1;
            }
        }
    }

    #[inline]
    pub fn or(mut self, other: Self) -> BitVec<'a> {
        #[cfg(features = "nightly")]
        {
            assert!(self.capacity() == other.capacity());

            let left_ptr: *mut u32x4 = self.as_mut_ptr().cast();
            let right_ptr: *const u32x4 = other.as_ptr().cast();

            for n in 0..(self.capacity() / (32 * u32x4::lanes())) {
                unsafe {
                    *left_ptr.add(n) = *left_ptr.add(n) | *right_ptr.add(n);

                    left_ptr.add(n);
                };
            }

            return self;
        }

        #[target_feature(enable = "sse2")]
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;

            let left_ptr: *mut __m128i = self.as_mut_ptr().cast();
            let right_ptr: *const __m128i = other.as_ptr().cast();

            for n in 0..(self.capacity() / (32 * 4)) {
                unsafe {
                    *left_ptr.add(n) = _mm_or_si128(*left_ptr.add(n), *right_ptr.add(n));
                };
            }

            return self;
        }

        unimplemented!()
    }

    #[inline]
    pub fn and(mut self, other: Self) -> BitVec<'a> {
        assert!(self.capacity() == other.capacity());

        #[cfg(features = "nightly")]
        {
            let left_ptr: *mut u32x4 = self.as_mut_ptr().cast();
            let right_ptr: *const u32x4 = other.as_ptr().cast();

            for n in 0..(self.capacity() / (32 * u32x4::lanes())) {
                unsafe {
                    *left_ptr.add(n) = *left_ptr.add(n) & *right_ptr.add(n);
                };
            }

            return self;
        }

        #[target_feature(enable = "sse2")]
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;

            let left_ptr: *mut __m128i = self.as_mut_ptr().cast();
            let right_ptr: *const __m128i = other.as_ptr().cast();

            for n in 0..(self.capacity() / (32 * 4)) {
                unsafe {
                    *left_ptr.add(n) = _mm_and_si128(*left_ptr.add(n), *right_ptr.add(n));
                };
            }

            return self;
        }

        unimplemented!()
    }

    #[inline]
    pub unsafe fn set_unchecked(&mut self, index: usize, value: bool) {
        let (offset, bit) = calc_bit(index);

        let ptr = self.as_mut_ptr().add(offset);
        let num: u32 = value as u32;

        *ptr = (*ptr & !(1 << bit)) | (num << bit);
    }

    #[inline]
    pub unsafe fn toggle_unchecked(&mut self, index: usize) {
        let (offset, bit) = calc_bit(index);

        *self.as_mut_ptr().add(offset) ^= 1 << bit;
    }

    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> bool {
        let (offset, bit) = calc_bit(index);

        ((*self.as_ptr().add(offset) >> bit) & 1) != 0
    }

    #[inline]
    pub unsafe fn replace_unchecked(&mut self, index: usize, value: bool) -> bool {
        let (offset, bit) = calc_bit(index);

        let ptr = self.as_mut_ptr().add(offset);
        let old = (*ptr >> bit) & 1 != 0;

        *ptr ^= 1 << bit;

        old
    }

    #[inline]
    pub fn set(&mut self, index: usize, value: bool) -> Result<(), ArenaError> {
        if check_bounds(index, self.capacity()) {
            Ok(unsafe { self.set_unchecked(index, value) })
        } else {
            Err(ArenaError::OutOfBounds)
        }
    }

    #[inline]
    pub fn toggle(&mut self, index: usize) -> Result<(), ArenaError> {
        if check_bounds(index, self.capacity()) {
            Ok(unsafe { self.toggle_unchecked(index) })
        } else {
            Err(ArenaError::OutOfBounds)
        }
    }

    #[inline]
    pub fn get(&self, index: usize) -> Option<bool> {
        if check_bounds(index, self.capacity()) {
            Some(unsafe { self.get_unchecked(index) })
        } else {
            None
        }
    }

    #[inline]
    pub fn replace(&mut self, index: usize, value: bool) -> Result<bool, ArenaError> {
        if check_bounds(index, self.capacity()) {
            Ok(unsafe { self.replace_unchecked(index, value) })
        } else {
            Err(ArenaError::OutOfBounds)
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

impl<'a> BitOr for BitVec<'a> {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        self.or(rhs)
    }
}

impl<'a> BitAnd for BitVec<'a> {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self {
        self.and(rhs)
    }
}

impl<'a> Index<usize> for BitVec<'a> {
    type Output = bool;

    fn index(&self, index: usize) -> &Self::Output {
        const TRUE: bool = true;
        const FALSE: bool = true;

        if self.get(index).unwrap() {
            &TRUE
        } else {
            &FALSE
        }
    }
}

#[inline(always)]
fn check_bounds(index: usize, capacity: usize) -> bool {
    if index >= capacity {
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

    #[test]
    fn test_or_subslice() {
        let arena = Arena::with_capacity(5_242_880);

        // Single simd instruction
        let mut one = BitVec::with_capacity_in(&arena, 256);

        (128..256).for_each(|n| one.set(n, true).unwrap());
        one.or_subslice(0, 128, 128);

        (0..256).for_each(|n| {
            assert!(one.get(n).unwrap());
        });
    }

    #[test]
    fn test_and() {
        let arena = Arena::with_capacity(5_242_880);

        // Single simd instruction
        let mut one = BitVec::with_capacity_in(&arena, 256);
        let mut two = BitVec::with_capacity_in(&arena, 256);

        (0..128).for_each(|n| one.set(n, true).unwrap());
        (128..256).for_each(|n| two.set(n, true).unwrap());

        let and = one & two;

        (0..256).for_each(|n| {
            assert!(and.get(n).unwrap() == false);
        });
    }

    #[test]
    fn test_or() {
        let arena = Arena::with_capacity(5_242_880);

        // Single simd instruction
        let mut one = BitVec::with_capacity_in(&arena, 256);
        let mut two = BitVec::with_capacity_in(&arena, 256);

        (0..128).for_each(|n| one.set(n, true).unwrap());
        (128..256).for_each(|n| two.set(n, true).unwrap());

        let or = one | two;

        (0..256).for_each(|n| {
            assert!(or.get(n).unwrap());
        });
    }

    fn test_bitvec(size: usize, arena: &Arena) {
        let mut vec = BitVec::with_capacity_in(&arena, size);
        for n in 0..size {
            assert!(vec.get(n).unwrap() == false);
        }

        for n in 0..size {
            vec.set(n, true).unwrap();
            assert!(vec.get(n).unwrap());
        }

        for n in 0..size {
            vec.toggle(n).unwrap();
            assert!(vec.get(n).unwrap() == false);
        }

        assert!(vec.get(vec.capacity() * 2).is_none());
        assert!(vec.set(vec.capacity() * 2, true).is_err());
    }

    #[test]
    fn size_cases() {
        let arena = Arena::with_capacity(5_242_880);
        test_bitvec(4, &arena);
        test_bitvec(32, &arena);
        test_bitvec(128, &arena);
        test_bitvec(4058, &arena);
        test_bitvec(123123, &arena);
        test_bitvec(111, &arena);
        test_bitvec(2, &arena);
    }
}

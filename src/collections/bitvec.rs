use super::alloc::*;
use crate::*;

#[cfg(feature = "nightly")]
use packed_simd::u64x2;

use std::ops::{BitAnd, BitOr, Index};

/// Store up to u32::MAX packed bytes and address them one by one.
pub struct BitVec<'a> {
    arena: &'a Arena,
    buffer: ScopedRawBuffer<'a>,
    capacity: u32,
    size: u32,
}

impl<'a> BitVec<'a> {
    /// Capacity for a BitVec is in BITS
    /// However, it must be 32-bit aligned as we internally store in integers
    pub fn with_size_in(arena: &'a Arena, size: u32) -> Self {
        let intrin_size = std::mem::size_of::<u64>() as u32 * 4;
        let step_bits = intrin_size * 8;
        let capacity = (size + step_bits) - (size % step_bits);
        let capacity_bytes = capacity / 8;

        let layout = Layout::from_size_align(capacity_bytes as _, intrin_size as _).unwrap();
        let buffer = arena.alloc_scoped_raw(layout);

        // zero buffer memory
        unsafe {
            let buffer_ptr: *mut u64 = buffer.ptr.as_ptr().cast();
            for i in 0..capacity as usize / 64 {
                std::ptr::write_volatile(buffer_ptr.add(i), 0);
            }
        }

        Self {
            arena,
            buffer,
            capacity,
            size,
        }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &[u64] {
        unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr(), self.capacity() as usize / 64) }
    }

    #[inline]
    pub fn as_slice(&self) -> &[u64] {
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.capacity() as usize / 64) }
    }

    /// `src`, `dst` and `len` must be 64 bit aligned
    pub fn or_subslice_aligned(&mut self, dst: u32, src: u32, len: u32) {
        debug_assert!(dst % 64 == 0);
        debug_assert!(src % 64 == 0);

        let dst = (dst / 64) as usize;
        let src = (src / 64) as usize;
        let len = len as usize;

        const BITS_PER_SIMD: usize = (std::mem::size_of::<u64>() * 2) * 8;

        if len % BITS_PER_SIMD == 0 {
            #[cfg(features = "nightly")]
            {
                unsafe {
                    let left_ptr: *mut u64x2 = self.as_mut_ptr().add(dst).cast();
                    let right_ptr: *const u64x2 = self.as_ptr().add(src).cast();
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
                    let left_ptr: *mut __m128i = self.as_mut_ptr().add(dst).cast();
                    let right_ptr: *const __m128i = self.as_ptr().add(src).cast();

                    *left_ptr = _mm_or_si128(*left_ptr, *right_ptr);
                }
                return;
            }
        }

        let slice = self.buffer.as_mut_slice::<u64>();

        let mut bit = 0;
        while bit < len {
            let src_idx = src + (bit / 64);
            let dst_idx = dst + (bit / 64);
            if len - bit >= 64 {
                slice[dst_idx] = slice[dst_idx] | slice[src_idx];
                bit += 64;
            } else {
                slice[dst_idx] = slice[dst_idx] >> bit as u64 | slice[src_idx];
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

            for n in 0..(self.capacity() / (32 * 4)) as usize {
                unsafe {
                    *left_ptr.add(n) = _mm_or_si128(*left_ptr.add(n), *right_ptr.add(n));
                };
            }

            return self;
        }
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

            self
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

            for n in 0..(self.capacity() / (32 * 4)) as usize {
                unsafe {
                    *left_ptr.add(n) = _mm_and_si128(*left_ptr.add(n), *right_ptr.add(n));
                };
            }

            return self;
        }
    }

    #[inline]
    pub unsafe fn set_unchecked(&mut self, index: u32, value: bool) {
        let (offset, bit) = calc_bit(index);

        let ptr = self.as_mut_ptr().add(offset);
        let num: u64 = value as u64;

        *ptr = (*ptr & !(1 << bit)) | (num << bit);
    }

    #[inline]
    pub unsafe fn toggle_unchecked(&mut self, index: u32) {
        let (offset, bit) = calc_bit(index);

        *self.as_mut_ptr().add(offset) ^= 1 << bit;
    }

    #[inline]
    pub unsafe fn get_unchecked(&self, index: u32) -> bool {
        let (offset, bit) = calc_bit(index);

        ((*self.as_ptr().add(offset) >> bit) & 1) != 0
    }

    #[inline]
    pub unsafe fn replace_unchecked(&mut self, index: u32, value: bool) -> bool {
        let (offset, bit) = calc_bit(index);

        let ptr = self.as_mut_ptr().add(offset);
        let old = (*ptr >> bit) & 1 != 0;

        let num: u64 = value as u64;
        *ptr = (*ptr & !(1 << bit)) | (num << bit);

        old
    }

    #[inline]
    pub fn set(&mut self, index: u32, value: bool) -> Result<(), ArenaError> {
        if check_bounds(index, self.capacity()) {
            Ok(unsafe { self.set_unchecked(index, value) })
        } else {
            Err(ArenaError::OutOfBounds)
        }
    }

    #[inline]
    pub fn toggle(&mut self, index: u32) -> Result<(), ArenaError> {
        if check_bounds(index, self.capacity()) {
            Ok(unsafe { self.toggle_unchecked(index) })
        } else {
            Err(ArenaError::OutOfBounds)
        }
    }

    #[inline]
    pub fn get(&self, index: u32) -> Option<bool> {
        if check_bounds(index, self.capacity()) {
            Some(unsafe { self.get_unchecked(index) })
        } else {
            None
        }
    }

    #[inline]
    pub fn replace(&mut self, index: u32, value: bool) -> Result<bool, ArenaError> {
        if check_bounds(index, self.capacity()) {
            Ok(unsafe { self.replace_unchecked(index, value) })
        } else {
            Err(ArenaError::OutOfBounds)
        }
    }

    #[inline]
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    #[inline]
    pub fn len(&self) -> u32 {
        self.size
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut u64 {
        self.buffer.ptr.as_ptr().cast()
    }

    pub(crate) fn as_ptr(&self) -> *const u64 {
        self.buffer.ptr.as_ptr().cast()
    }

    pub fn iter(&'a self) -> BitVecIter<'a> {
        BitVecIter::new(self)
    }
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
pub struct BitVecIter<'a> {
    inner: &'a BitVec<'a>,
    head: u32,
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
impl<'a> BitVecIter<'a> {
    fn new(inner: &'a BitVec<'a>) -> Self {
        Self { inner, head: 0 }
    }

    fn next(&mut self) -> Option<u32> {
        for i in self.head..self.inner.len() {
            if unsafe { self.inner.get_unchecked(i) } {
                self.head = i + 1;
                return Some(i);
            }
        }
        None
    }
}

const DECODE_BUFFER_CAPACITY: usize = 512;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub struct BitVecIter<'a> {
    inner: &'a BitVec<'a>,
    head: usize,
    decode_buffer: ScopedRawBuffer<'a>,
    decode_head: usize,
    decode_tail: usize,
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl<'a> BitVecIter<'a> {
    fn new(inner: &'a BitVec<'a>) -> Self {
        let buffer_size = DECODE_BUFFER_CAPACITY * std::mem::size_of::<u32>();
        let align_size = std::mem::size_of::<u64>() * 4;

        let layout = Layout::from_size_align(buffer_size, align_size).unwrap();
        let decode_buffer = inner.arena.alloc_scoped_raw(layout);

        Self {
            inner,
            head: 0,
            decode_buffer,
            decode_head: 0,
            decode_tail: 0,
        }
    }

    fn decode_next(&mut self) {
        let remaining_bits = self.inner.len() as usize - self.head;
        let chunk_size = DECODE_BUFFER_CAPACITY.min(remaining_bits);

        debug_assert!(self.head % 64 == 0);
        let chunk_slice_start = self.head / 64;
        let chunk_slice_end = chunk_slice_start + (chunk_size + 63) / 64;

        let decoded = unsafe {
            bitmap_decode_sse2(
                self.head as u32,
                &self.inner.as_slice()[dbg!(chunk_slice_start..chunk_slice_end)],
                self.decode_buffer.as_mut_slice::<u32>(),
            )
        };

        self.head += chunk_size;
        self.decode_head = 0;
        self.decode_tail = decoded;
    }

    fn next(&mut self) -> Option<u32> {
        if self.decode_head != self.decode_tail {
            let val = unsafe {
                *self
                    .decode_buffer
                    .as_slice::<u32>()
                    .get_unchecked(self.decode_head)
            };
            self.decode_head += 1;
            Some(val)
        } else {
            if self.head < self.inner.len() as usize {
                self.decode_next();
                if self.decode_head != self.decode_tail {
                    let val = unsafe {
                        *self
                            .decode_buffer
                            .as_slice::<u32>()
                            .get_unchecked(self.decode_head)
                    };
                    self.decode_head += 1;
                    return Some(val);
                }
            }
            None
        }
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

impl<'a> Index<u32> for BitVec<'a> {
    type Output = bool;

    fn index(&self, index: u32) -> &Self::Output {
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
fn check_bounds(index: u32, capacity: u32) -> bool {
    if index >= capacity {
        false
    } else {
        true
    }
}

#[inline(always)]
// Calculate a bit index to a u32 offset + shift
fn calc_bit(index: u32) -> (usize, u32) {
    let bit = index % 64;

    ((index / 64) as usize, bit as u32)
}

#[target_feature(enable = "sse2")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn bitmap_decode_sse2(start_offset: u32, bitmap: &[u64], out: &mut [u32]) -> usize {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    #[repr(align(16))]
    struct Align16<T>(T);

    const LUT_INDICES: Align16<[[u32; 4]; 16]> = {
        Align16([
            [0, 0, 0, 0], // 0000 .... 0 0 0 0
            [0, 0, 0, 0], // 0001 X... 0 0 0 0
            [1, 0, 0, 0], // 0010 Y... 0 1 0 0
            [0, 1, 0, 0], // 0011 XY.. 0 0 0 0
            [2, 0, 0, 0], // 0100 Z... 0 0 2 0
            [0, 2, 0, 0], // 0101 XZ.. 0 0 1 0
            [1, 2, 0, 0], // 0110 YZ.. 0 1 1 0
            [0, 1, 2, 0], // 0111 XYZ. 0 0 0 0
            [3, 0, 0, 0], // 1000 W... 0 0 0 3
            [0, 3, 0, 0], // 1001 XW.. 0 0 0 2
            [1, 3, 0, 0], // 1010 YW.. 0 1 0 2
            [0, 1, 3, 0], // 1011 XYW. 0 0 0 1
            [2, 3, 0, 0], // 1100 ZW.. 0 0 2 2
            [0, 2, 3, 0], // 1101 XZW. 0 0 1 1
            [1, 2, 3, 0], // 1110 YZW. 0 1 1 1
            [0, 1, 2, 3], // 1111 XYZW 0 0 0 0
        ])
    };

    // NOTE: u16 primitive chosen as the best performer in benchmarks
    const LUT_POPCNT: Align16<[u16; 16]> = {
        Align16([
            0, // 0000 .... 0 0 0 0
            1, // 0001 X... 0 0 0 0
            1, // 0010 Y... 0 1 0 0
            2, // 0011 XY.. 0 0 0 0
            1, // 0100 Z... 0 0 2 0
            2, // 0101 XZ.. 0 0 1 0
            2, // 0110 YZ.. 0 1 1 0
            3, // 0111 XYZ. 0 0 0 0
            1, // 1000 W... 0 0 0 3
            2, // 1001 XW.. 0 0 0 2
            2, // 1010 YW.. 0 1 0 2
            3, // 1011 XYW. 0 0 0 1
            2, // 1100 ZW.. 0 0 2 2
            3, // 1101 XZW. 0 0 1 1
            3, // 1110 YZW. 0 1 1 1
            4, // 1111 XYZW 0 0 0 0
        ])
    };

    #[inline(always)]
    unsafe fn lookup_index(mask: isize) -> __m128i {
        _mm_load_si128(LUT_INDICES.0.as_ptr().offset(mask) as _)
    }

    let mut out_pos = 0;

    let bitmap_iter = bitmap.into_iter();

    debug_assert!(out.len() >= bitmap_iter.len() * 64);

    let mut base: __m128i = _mm_set1_epi32(start_offset as i32);

    for bits in bitmap_iter {
        if *bits == 0 {
            base = _mm_add_epi32(base, _mm_set1_epi32(64));
            continue;
        }

        for i in 0..4 {
            let move_mask = (*bits >> (i * 16)) as u16;

            // pack the elements to the left using the move mask
            let movemask_a = move_mask & 0xF;
            let movemask_b = (move_mask >> 4) & 0xF;
            let movemask_c = (move_mask >> 8) & 0xF;
            let movemask_d = (move_mask >> 12) & 0xF;

            let mut a = lookup_index(movemask_a as isize);
            let mut b = lookup_index(movemask_b as isize);
            let mut c = lookup_index(movemask_c as isize);
            let mut d = lookup_index(movemask_d as isize);

            // offset by bit index
            a = _mm_add_epi32(base, a);
            b = _mm_add_epi32(base, b);
            c = _mm_add_epi32(base, c);
            d = _mm_add_epi32(base, d);

            // increase the base
            base = _mm_add_epi32(base, _mm_set1_epi32(16));

            // correct lookups
            b = _mm_add_epi32(_mm_set1_epi32(4), b);
            c = _mm_add_epi32(_mm_set1_epi32(8), c);
            d = _mm_add_epi32(_mm_set1_epi32(12), d);

            // get the number of elements being output
            let adv_a = LUT_POPCNT.0.get_unchecked(movemask_a as usize);
            let adv_b = LUT_POPCNT.0.get_unchecked(movemask_b as usize);
            let adv_c = LUT_POPCNT.0.get_unchecked(movemask_c as usize);
            let adv_d = LUT_POPCNT.0.get_unchecked(movemask_d as usize);

            let adv_ab = adv_a + adv_b;
            let adv_abc = adv_ab + adv_c;
            let adv_abcd = adv_abc + adv_d;

            let out_ptr = out.get_unchecked_mut(out_pos) as *mut u32;
            out_pos += adv_abcd as usize;

            // perform the store
            _mm_storeu_si128(out_ptr as *mut _, a);
            _mm_storeu_si128(out_ptr.offset(*adv_a as isize) as _, b);
            _mm_storeu_si128(out_ptr.offset(adv_ab as isize) as _, c);
            _mm_storeu_si128(out_ptr.offset(adv_abc as isize) as _, d);
        }
    }
    out_pos
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_or_subslice() {
        let arena = Arena::with_capacity(5_242_880);

        // Single simd instruction
        let mut one = BitVec::with_size_in(&arena, 256);

        (128..256).for_each(|n| one.set(n, true).unwrap());
        one.or_subslice_aligned(0, 128, 128);

        (0..256).for_each(|n| {
            assert!(one.get(n).unwrap());
        });
    }

    #[test]
    fn test_and() {
        let arena = Arena::with_capacity(5_242_880);

        // Single simd instruction
        let mut one = BitVec::with_size_in(&arena, 256);
        let mut two = BitVec::with_size_in(&arena, 256);

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
        let mut one = BitVec::with_size_in(&arena, 256);
        let mut two = BitVec::with_size_in(&arena, 256);

        (0..128).for_each(|n| one.set(n, true).unwrap());
        (128..256).for_each(|n| two.set(n, true).unwrap());

        let or = one | two;

        (0..256).for_each(|n| {
            assert!(or.get(n).unwrap());
        });
    }

    #[test]
    fn test_iter() {
        let arena = Arena::with_capacity(5_242_880);

        let mut vec = BitVec::with_size_in(&arena, 2048);

        (0..16).for_each(|n| vec.set(n, true).unwrap());
        (256..512)
            .filter(|n| n % 2 == 0 && n % 3 == 0)
            .for_each(|n| vec.set(n, true).unwrap());

        let mut iter = vec.iter();
        for n in 0..16 {
            assert_eq!(Some(n), iter.next());
        }

        for n in 256..512 {
            if n % 2 == 0 && n % 3 == 0 {
                assert_eq!(Some(n), iter.next());
            }
        }
        assert_eq!(None, iter.next());
    }

    fn test_bitvec(size: u32, arena: &Arena) {
        let mut vec = BitVec::with_size_in(&arena, size);
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

use super::vec::Vec;
use crate::*;
use std::fmt;
use std::iter;
use std::marker::PhantomData;
use std::mem;
use std::slice;

pub type Word = u64;
pub const WORD_BYTES: usize = mem::size_of::<Word>();
pub const WORD_BITS: usize = WORD_BYTES * 8;

/// A fixed-size bitset type with a dense representation. It does not support
/// resizing after creation; use `GrowableBitSet` for that.
///
/// `T` is an index type, typically a newtyped `usize` wrapper, but it can also
/// just be `usize`.
///
/// All operations that involve an element will panic if the element is equal
/// to or greater than the domain size. All operations that involve two bitsets
/// will panic if the bitsets have differing domain sizes.
#[derive(Clone, Eq, PartialEq)]
pub struct BitSet<'a, T> {
    domain_size: usize,
    words: Vec<'a, Word>,
    marker: PhantomData<T>,
}

impl<'a, T> BitSet<'a, T> {
    /// Creates a new, empty bitset with a given `domain_size`.
    #[inline]
    pub fn new_empty_in(arena: &'a Arena, domain_size: usize) -> BitSet<T> {
        let num_words = num_words(domain_size);
        BitSet {
            domain_size,
            words: Vec::with_capacity_in(&arena, domain_size),
            marker: PhantomData,
        }
    }

    /// Creates a new, filled bitset with a given `domain_size`.
    #[inline]
    pub fn new_filled_in(arena: &'a Arena, domain_size: usize) -> BitSet<T> {
        let num_words = num_words(domain_size);
        let mut result = BitSet {
            domain_size,
            words: Vec::with_capacity_in(&arena, domain_size),
            marker: PhantomData,
        };
        result.clear_excess_bits();
        result
    }

    /// Gets the domain size.
    pub fn domain_size(&self) -> usize {
        self.domain_size
    }

    /// Clear all elements.
    #[inline]
    pub fn clear(&mut self) {
        for word in &mut self.words {
            *word = 0;
        }
    }

    /// Clear excess bits in the final word.
    fn clear_excess_bits(&mut self) {
        let num_bits_in_final_word = self.domain_size % WORD_BITS;
        if num_bits_in_final_word > 0 {
            let mask = (1 << num_bits_in_final_word) - 1;
            let final_word_idx = self.words.len() - 1;
            self.words[final_word_idx] &= mask;
        }
    }

    /// Efficiently overwrite `self` with `other`.
    pub fn overwrite(&mut self, other: &BitSet<T>) {
        assert!(self.domain_size == other.domain_size);
        self.words.clone_from_slice(&other.words);
    }

    /// Count the number of set bits in the set.
    pub fn count(&self) -> usize {
        self.words.iter().map(|e| e.count_ones() as usize).sum()
    }

    /// Returns `true` if `self` contains `elem`.
    #[inline]
    pub fn contains(&self, elem: T) -> bool {
        assert!(elem.index() < self.domain_size);
        let (word_index, mask) = word_index_and_mask(elem);
        (self.words[word_index] & mask) != 0
    }

    /// Is `self` is a (non-strict) superset of `other`?
    #[inline]
    pub fn superset(&self, other: &BitSet<T>) -> bool {
        assert_eq!(self.domain_size, other.domain_size);
        self.words
            .iter()
            .zip(&other.words)
            .all(|(a, b)| (a & b) == *b)
    }

    /// Is the set empty?
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|a| *a == 0)
    }

    /// Insert `elem`. Returns whether the set has changed.
    #[inline]
    pub fn insert(&mut self, elem: T) -> bool {
        assert!(elem.index() < self.domain_size);
        let (word_index, mask) = word_index_and_mask(elem);
        let word_ref = &mut self.words[word_index];
        let word = *word_ref;
        let new_word = word | mask;
        *word_ref = new_word;
        new_word != word
    }

    /// Sets all bits to true.
    pub fn insert_all(&mut self) {
        for word in &mut self.words {
            *word = !0;
        }
        self.clear_excess_bits();
    }

    /// Returns `true` if the set has changed.
    #[inline]
    pub fn remove(&mut self, elem: T) -> bool {
        assert!(elem.index() < self.domain_size);
        let (word_index, mask) = word_index_and_mask(elem);
        let word_ref = &mut self.words[word_index];
        let word = *word_ref;
        let new_word = word & !mask;
        *word_ref = new_word;
        new_word != word
    }

    /// Sets `self = self | other` and returns `true` if `self` changed
    /// (i.e., if new bits were added).
    pub fn union(&mut self, other: &impl UnionIntoBitSet<T>) -> bool {
        other.union_into(self)
    }

    /// Sets `self = self - other` and returns `true` if `self` changed.
    /// (i.e., if any bits were removed).
    pub fn subtract(&mut self, other: &impl SubtractFromBitSet<T>) -> bool {
        other.subtract_from(self)
    }

    /// Sets `self = self & other` and return `true` if `self` changed.
    /// (i.e., if any bits were removed).
    pub fn intersect(&mut self, other: &BitSet<T>) -> bool {
        assert_eq!(self.domain_size, other.domain_size);
        bitwise(&mut self.words, &other.words, |a, b| a & b)
    }

    /// Gets a slice of the underlying words.
    pub fn words(&self) -> &[Word] {
        &self.words
    }

    /// Iterates over the indices of set bits in a sorted order.
    #[inline]
    pub fn iter(&self) -> BitIter<'_, T> {
        BitIter {
            cur: None,
            iter: self.words.iter().enumerate(),
            marker: PhantomData,
        }
    }

    /// Set `self = self | other`. In contrast to `union` returns `true` if the set contains at
    /// least one bit that is not in `other` (i.e. `other` is not a superset of `self`).
    ///
    /// This is an optimization for union of a hybrid bitset.
    fn reverse_union_sparse(&mut self, sparse: &SparseBitSet<T>) -> bool {
        assert!(sparse.domain_size == self.domain_size);
        self.clear_excess_bits();

        let mut not_already = false;
        // Index of the current word not yet merged.
        let mut current_index = 0;
        // Mask of bits that came from the sparse set in the current word.
        let mut new_bit_mask = 0;
        for (word_index, mask) in sparse.iter().map(|x| word_index_and_mask(*x)) {
            // Next bit is in a word not inspected yet.
            if word_index > current_index {
                self.words[current_index] |= new_bit_mask;
                // Were there any bits in the old word that did not occur in the sparse set?
                not_already |= (self.words[current_index] ^ new_bit_mask) != 0;
                // Check all words we skipped for any set bit.
                not_already |= self.words[current_index + 1..word_index]
                    .iter()
                    .any(|&x| x != 0);
                // Update next word.
                current_index = word_index;
                // Reset bit mask, no bits have been merged yet.
                new_bit_mask = 0;
            }
            // Add bit and mark it as coming from the sparse set.
            // self.words[word_index] |= mask;
            new_bit_mask |= mask;
        }
        self.words[current_index] |= new_bit_mask;
        // Any bits in the last inspected word that were not in the sparse set?
        not_already |= (self.words[current_index] ^ new_bit_mask) != 0;
        // Any bits in the tail? Note `clear_excess_bits` before.
        not_already |= self.words[current_index + 1..].iter().any(|&x| x != 0);

        not_already
    }
}

/// This is implemented by all the bitsets so that BitSet::union() can be
/// passed any type of bitset.
pub trait UnionIntoBitSet<T> {
    // Performs `other = other | self`.
    fn union_into(&self, other: &mut BitSet<T>) -> bool;
}

/// This is implemented by all the bitsets so that BitSet::subtract() can be
/// passed any type of bitset.
pub trait SubtractFromBitSet<T> {
    // Performs `other = other - self`.
    fn subtract_from(&self, other: &mut BitSet<T>) -> bool;
}

impl<T> UnionIntoBitSet<T> for BitSet<T> {
    fn union_into(&self, other: &mut BitSet<T>) -> bool {
        assert_eq!(self.domain_size, other.domain_size);
        bitwise(&mut other.words, &self.words, |a, b| a | b)
    }
}

impl<T> SubtractFromBitSet<T> for BitSet<T> {
    fn subtract_from(&self, other: &mut BitSet<T>) -> bool {
        assert_eq!(self.domain_size, other.domain_size);
        bitwise(&mut other.words, &self.words, |a, b| a & !b)
    }
}

impl<T> fmt::Debug for BitSet<T> {
    fn fmt(&self, w: &mut fmt::Formatter<'_>) -> fmt::Result {
        w.debug_list().entries(self.iter()).finish()
    }
}

impl<T> ToString for BitSet<T> {
    fn to_string(&self) -> String {
        let mut result = String::new();
        let mut sep = '[';

        // Note: this is a little endian printout of bytes.

        // i tracks how many bits we have printed so far.
        let mut i = 0;
        for word in &self.words {
            let mut word = *word;
            for _ in 0..WORD_BYTES {
                // for each byte in `word`:
                let remain = self.domain_size - i;
                // If less than a byte remains, then mask just that many bits.
                let mask = if remain <= 8 { (1 << remain) - 1 } else { 0xFF };
                assert!(mask <= 0xFF);
                let byte = word & mask;

                result.push_str(&format!("{}{:02x}", sep, byte));

                if remain <= 8 {
                    break;
                }
                word >>= 8;
                i += 8;
                sep = '-';
            }
            sep = '|';
        }
        result.push(']');

        result
    }
}

pub struct BitIter<'a, T> {
    cur: Option<(Word, usize)>,
    iter: iter::Enumerate<slice::Iter<'a, Word>>,
    marker: PhantomData<T>,
}

impl<'a, T> Iterator for BitIter<'a, T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        loop {
            if let Some((ref mut word, offset)) = self.cur {
                let bit_pos = word.trailing_zeros() as usize;
                if bit_pos != WORD_BITS {
                    let bit = 1 << bit_pos;
                    *word ^= bit;
                    return Some(T::new(bit_pos + offset));
                }
            }

            let (i, word) = self.iter.next()?;
            self.cur = Some((*word, WORD_BITS * i));
        }
    }
}

#[inline]
fn bitwise<Op>(out_vec: &mut [Word], in_vec: &[Word], op: Op) -> bool
where
    Op: Fn(Word, Word) -> Word,
{
    assert_eq!(out_vec.len(), in_vec.len());
    let mut changed = false;
    for (out_elem, in_elem) in out_vec.iter_mut().zip(in_vec.iter()) {
        let old_val = *out_elem;
        let new_val = op(old_val, *in_elem);
        *out_elem = new_val;
        changed |= old_val != new_val;
    }
    changed
}

/*

/// A resizable bitset type with a dense representation.
///
/// `T` is an index type, typically a newtyped `usize` wrapper, but it can also
/// just be `usize`.
///
/// All operations that involve an element will panic if the element is equal
/// to or greater than the domain size.
#[derive(Clone, Debug, PartialEq)]
pub struct GrowableBitSet<T> {
    bit_set: BitSet<T>,
}

impl<T> GrowableBitSet<T> {
    /// Ensure that the set can hold at least `min_domain_size` elements.
    pub fn ensure(&mut self, min_domain_size: usize) {
        if self.bit_set.domain_size < min_domain_size {
            self.bit_set.domain_size = min_domain_size;
        }

        let min_num_words = num_words(min_domain_size);
        if self.bit_set.words.len() < min_num_words {
            self.bit_set.words.resize(min_num_words, 0)
        }
    }

    pub fn new_empty() -> GrowableBitSet<T> {
        GrowableBitSet {
            bit_set: BitSet::new_empty(0),
        }
    }

    pub fn with_capacity(capacity: usize) -> GrowableBitSet<T> {
        GrowableBitSet {
            bit_set: BitSet::new_empty(capacity),
        }
    }

    /// Returns `true` if the set has changed.
    #[inline]
    pub fn insert(&mut self, elem: T) -> bool {
        self.ensure(elem.index() + 1);
        self.bit_set.insert(elem)
    }

    #[inline]
    pub fn contains(&self, elem: T) -> bool {
        let (word_index, mask) = word_index_and_mask(elem);
        if let Some(word) = self.bit_set.words.get(word_index) {
            (word & mask) != 0
        } else {
            false
        }
    }
}
*/

#[inline]
fn num_words<T>(domain_size: T) -> usize {
    (domain_size.index() + WORD_BITS - 1) / WORD_BITS
}

#[inline]
fn word_index_and_mask<T>(elem: T) -> (usize, Word) {
    let elem = elem.index();
    let word_index = elem / WORD_BITS;
    let mask = 1 << (elem % WORD_BITS);
    (word_index, mask)
}

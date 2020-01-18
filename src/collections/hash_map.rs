/*
use crate::*;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

use super::Vec;


pub struct Entry<K, T> {
    key: K,
    value: Option<T>,
}

impl<K: Hash, T> Hash for Entry<K, T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.key.hash(state);
    }
}

pub struct HashMap<'a, K: Hash, T, H = std::collections::hash_map::DefaultHasher> {
    arena: &'a Arena,
    data: Vec<'a, Entry<K, T>>,
    capacity: usize,
    size: usize,
    _marker: PhantomData<(H, K)>,
}

struct HashKey {}

impl<'a, K: Hash, T, H: Hasher> HashMap<'a, K, T, H> {
    /// HashMap allocates size_of<T> * n + ((size_of<u64> + usize) * n) bytes
    pub fn with_capacity_in(arena: &'a Arena, capacity: usize) -> Self {
        let data = Vec::with_capacity_in(arena, capacity);

        Self {
            arena,
            data,
            capacity,
            size: 0,
            _marker: Default::default(),
        }
    }

    pub fn insert(key: K, value: T) -> Option<T> {
        unimplemented!()
    }
}
*/

use std::{
    alloc::{alloc_zeroed, Layout},
    marker::PhantomData,
    ops::{Deref, DerefMut},
    ptr::NonNull,
    sync::atomic::{AtomicPtr, Ordering},
};

pub mod collections;

#[derive(Debug)]
pub enum ArenaError {
    NotDisposed,
    CapacityExceeded,
    OutOfBounds,
}

pub trait Dispose {
    /// Dispose consumes the scoped buffer because its pointers become invalidated on disposal.
    fn dispose(self) -> Result<(), ArenaError>;
    fn tail(&self) -> NonNull<u8>;
    fn ptr(&self) -> NonNull<u8>;
}

/// This buffer maintains a pointer to the current `tail` of the arena. The aim of this type is to allow deallocation
/// *if* no other allocations have occured between this scoped allocation and its disposal. This allows us to avoid
/// runtime checks in all allocations or a mutex, while still allowing for the case of a temporary allocation and
/// deallocation inside the arena in a short scoped amount of time.
///
/// SAFETY: Drop implementation will silently leak frame-scoped memory.
/// This buffer may leak memory until the next `reset`
struct ScopedRawBuffer<'a> {
    arena: &'a Arena,
    ptr: NonNull<u8>,
    tail: NonNull<u8>,
}
impl<'a> ScopedRawBuffer<'a> {
    fn new(arena: &'a Arena, ptr: NonNull<u8>, tail: NonNull<u8>) -> Self {
        Self { arena, ptr, tail }
    }
}
impl<'a> Dispose for ScopedRawBuffer<'a> {
    fn dispose(mut self) -> Result<(), ArenaError> {
        self.arena.try_dispose(&mut self)
    }
    fn tail(&self) -> NonNull<u8> {
        self.tail
    }
    fn ptr(&self) -> NonNull<u8> {
        self.ptr
    }
}
impl<'a> Drop for ScopedRawBuffer<'a> {
    // We cant call our regular consuming `dispose` because of drop impl. However, because its drop being called
    // we can still assume we are consumed and so it is safe to invalidate ourselves here.
    fn drop(&mut self) {
        self.arena.try_dispose(self);
    }
}

/// Typed wrapper for `ScopedRawBuffer`
pub struct ScopedBuffer<'a, T> {
    inner: ScopedRawBuffer<'a>,
    _marker: PhantomData<T>,
}
impl<'a, T> ScopedBuffer<'a, T> {
    fn new(inner: ScopedRawBuffer<'a>) -> Self {
        Self {
            inner,
            _marker: Default::default(),
        }
    }
}
impl<'a, T> Dispose for ScopedBuffer<'a, T> {
    fn dispose(self) -> Result<(), ArenaError> {
        self.inner.dispose()
    }
    fn tail(&self) -> NonNull<u8> {
        self.inner.tail()
    }
    fn ptr(&self) -> NonNull<u8> {
        self.inner.ptr()
    }
}
impl<'a, T> AsRef<T> for ScopedBuffer<'a, T> {
    fn as_ref(&self) -> &T {
        self.deref()
    }
}
impl<'a, T> Deref for ScopedBuffer<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &*(self.inner.ptr.as_ptr().cast()) }
    }
}
impl<'a, T> DerefMut for ScopedBuffer<'a, T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *(self.inner.ptr.as_ptr().cast()) }
    }
}

pub struct Arena {
    head: NonNull<u8>,
    tail: AtomicPtr<u8>,
    size: usize,
}
impl Arena {
    pub fn with_capacity(size: usize) -> Self {
        let (head, current) = {
            let head = unsafe {
                NonNull::new_unchecked(alloc_zeroed(Layout::from_size_align_unchecked(size, 16)))
            };
            (head, AtomicPtr::new(head.as_ptr()))
        };

        Self {
            head,
            tail: current,
            size,
        }
    }

    /// Realloc just allocs another chunk at the end of the arena, thus wasting memory but being fast
    ///
    pub(crate) unsafe fn realloc<T>(
        &self,
        src: NonNull<T>,
        src_layout: Layout,
        dst_layout: Layout,
    ) -> NonNull<T> {
        let (dst, _, _) = self.bump(dst_layout);
        let src_size = Self::align_size(src_layout);

        std::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_ptr(), src_size);

        dst
    }

    #[inline(always)]
    pub fn alloc_scoped<'a, T>(&'a self, value: T) -> ScopedBuffer<'a, T> {
        #[inline(always)]
        unsafe fn inner_writer<T, F>(ptr: NonNull<T>, f: F)
        where
            F: FnOnce() -> T,
        {
            // Taken from bumpalo

            // This function is translated as:
            // - allocate space for a T on the stack
            // - call f() with the return value being put onto this stack space
            // - memcpy from the stack to the heap
            //
            // Ideally we want LLVM to always realize that doing a stack
            // allocation is unnecessary and optimize the code so it writes
            // directly into the heap instead. It seems we get it to realize
            // this most consistently if we put this critical line into it's
            // own function instead of inlining it into the surrounding code.
            std::ptr::write(ptr.as_ptr(), f())
        }
        let raw = self.alloc_scoped_raw(Layout::new::<T>());
        unsafe {
            inner_writer(raw.ptr.cast(), || value);
        }
        ScopedBuffer::new(raw)
    }

    #[inline(always)]
    fn alloc_scoped_raw<'a>(&'a self, layout: Layout) -> ScopedRawBuffer<'a> {
        let (ptr, tail, size) = self.bump(layout);
        ScopedRawBuffer::new(self, ptr, tail)
    }

    pub(crate) fn alloc_raw<T>(&self, layout: Layout) -> NonNull<T> {
        log::trace!("alloc_raw: {:?}", layout);
        self.bump::<T>(layout).0
    }

    pub fn alloc_slice<'a, T>(&self, layout: Layout) -> &'a mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.bump::<T>(layout).0.as_ptr(), layout.size()) }
    }

    #[inline(always)]
    pub fn alloc<T>(&self, value: T) -> &mut T {
        self.alloc_with(|| value)
    }

    #[inline(always)]
    pub fn alloc_with<F, T>(&self, f: F) -> &mut T
    where
        F: FnOnce() -> T,
    {
        #[inline(always)]
        unsafe fn inner_writer<T, F>(ptr: NonNull<T>, f: F)
        where
            F: FnOnce() -> T,
        {
            // Taken from bumpalo

            // This function is translated as:
            // - allocate space for a T on the stack
            // - call f() with the return value being put onto this stack space
            // - memcpy from the stack to the heap
            //
            // Ideally we want LLVM to always realize that doing a stack
            // allocation is unnecessary and optimize the code so it writes
            // directly into the heap instead. It seems we get it to realize
            // this most consistently if we put this critical line into it's
            // own function instead of inlining it into the surrounding code.
            std::ptr::write(ptr.as_ptr(), f())
        }

        let layout = Layout::new::<T>();
        let (ptr, _, _) = self.bump(layout);

        unsafe {
            inner_writer(ptr, f);
            &mut *ptr.as_ptr()
        }
    }

    /// Tries to dispose the provided Disposable. This will succeed if the tail has not moved since allocation.
    /// Otherwise, it will fail. Failure indicates that the memory is 'leaked' until `reset` is called.
    fn try_dispose<T: Dispose>(&self, disposable: &mut T) -> Result<(), ArenaError> {
        self.try_raw_dispose(disposable.tail(), disposable.ptr())
    }

    /// Tries to dispose the provided Disposable. This will succeed if the tail has not moved since allocation.
    /// Otherwise, it will fail. Failure indicates that the memory is 'leaked' until `reset` is called.
    pub(crate) fn try_raw_dispose(
        &self,
        tail: NonNull<u8>,
        ptr: NonNull<u8>,
    ) -> Result<(), ArenaError> {
        match self.tail.compare_exchange_weak(
            tail.as_ptr(),
            ptr.as_ptr(),
            Ordering::SeqCst,
            Ordering::SeqCst,
        ) {
            Ok(returned) => {
                if returned == tail.as_ptr() {
                    Ok(())
                } else {
                    Err(ArenaError::NotDisposed)
                }
            }
            Err(_) => Err(ArenaError::NotDisposed),
        }
    }

    #[inline]
    pub unsafe fn reset(&mut self) {
        *self.tail.get_mut() = self.head.as_ptr();
    }

    #[inline(always)]
    pub(crate) fn bump<T>(&self, layout: Layout) -> (NonNull<T>, NonNull<u8>, usize) {
        log::trace!("bump: {:?}", layout);

        let size = Self::align_size(layout);

        let mut current = self.tail.load(Ordering::SeqCst);
        let mut new_tail = unsafe { current.add(size) };
        loop {
            match self.tail.compare_exchange_weak(
                current,
                new_tail,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(returned_current) => {
                    if current == returned_current {
                        break;
                    } else {
                        new_tail = unsafe { returned_current.add(size) };
                        current = returned_current;
                    }
                }
                Err(returned_current) => {
                    new_tail = unsafe { returned_current.add(size) };
                    current = returned_current;
                }
            }
        }
        unsafe {
            (
                NonNull::new_unchecked(current as *mut T),
                NonNull::new_unchecked(new_tail),
                size,
            )
        }
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        std::ptr::eq(self.head.as_ptr(), self.tail.load(Ordering::SeqCst))
    }

    #[inline(always)]
    fn size(&self) -> usize {
        self.size
    }

    #[inline(always)]
    fn consumed(&self) -> usize {
        unsafe {
            self.tail
                .load(Ordering::SeqCst)
                .sub(self.head.as_ptr() as usize) as usize
        }
    }

    #[inline(always)]
    fn align_size(layout: Layout) -> usize {
        layout.size() + (layout.size() % layout.align())
    }
}
unsafe impl Send for Arena {}
unsafe impl Sync for Arena {}

impl std::fmt::Debug for Arena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Arena (capacity = {}, consumed = {})",
            self.size,
            self.consumed()
        )
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use rayon::prelude::*;
    use std::ptr;

    pub struct TestType {
        int: u32,
        buffer: [u8; 1024],
    }
    impl Default for TestType {
        fn default() -> Self {
            Self {
                int: 123,
                buffer: [41; 1024],
            }
        }
    }

    pub fn get_tail(arena: &Arena) -> *mut u8 {
        arena.tail.load(Ordering::SeqCst)
    }

    #[test]
    fn scoped_allocators() {
        let mut arena = Arena::with_capacity(5_242_880);
        let size = Layout::new::<TestType>().size();

        // Confirm manual dispos
        let scoped = arena.alloc_scoped(TestType::default());
        assert_eq!(arena.consumed(), size);
        assert!(scoped.dispose().is_ok());
        assert!(arena.is_empty());

        // confirm an implicit-Drop
        {
            let _scoped = arena.alloc_scoped(TestType::default());
            assert_eq!(arena.consumed(), size);
        }
        assert!(arena.is_empty());

        // Confirm a failed drop
        let scoped = arena.alloc_scoped(TestType::default());
        assert_eq!(arena.consumed(), size);
        let _ = arena.alloc::<u8>(1);
        assert_eq!(arena.consumed(), size + 1);
        assert!(scoped.dispose().is_err());
        assert_eq!(arena.consumed(), size + 1);

        unsafe {
            arena.reset();
        }
        assert!(arena.is_empty());
    }

    #[test]
    fn threaded_alloc_reset() {
        let mut arena = Arena::with_capacity(5_242_880);

        (0..500).into_par_iter().for_each(|_| {
            let _ = arena.alloc(TestType::default());
        });
        unsafe {
            assert!(ptr::eq(
                get_tail(&arena),
                arena
                    .head
                    .as_ptr()
                    .add(Layout::new::<TestType>().size() * 500)
            ));
        }
        unsafe {
            arena.reset();
        }
        assert!(arena.is_empty());
    }

    #[test]
    fn loop_alloc() {
        let arena = Arena::with_capacity(5_242_880);

        (0..500).into_iter().for_each(|_| {
            let _ = arena.alloc(TestType::default());
        });
        unsafe {
            // Did we actually grow correctly?
            assert!(ptr::eq(
                get_tail(&arena),
                arena
                    .head
                    .as_ptr()
                    .add(Layout::new::<TestType>().size() * 500)
            ));
        }
    }

    #[test]
    fn simple() {
        let arena = Arena::with_capacity(5_242_880);
        unsafe {
            // Test single byte
            let ptr = arena.alloc::<u8>(1);
            assert!(ptr::eq(arena.head.as_ptr().add(1), get_tail(&arena)));
            assert!(ptr::eq((ptr as *mut u8).add(1), get_tail(&arena)));

            // Test struct with array
            let ptr = arena.alloc::<TestType>(TestType::default());
            assert!(ptr::eq(
                (ptr as *mut TestType as *mut u8).add(Layout::new::<TestType>().size()),
                get_tail(&arena)
            ));
        }
    }
}

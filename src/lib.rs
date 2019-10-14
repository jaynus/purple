use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::sync::atomic::{AtomicPtr, Ordering};

pub enum ArenaError {}

pub struct Arena {
    head: AtomicPtr<u8>,
    current: AtomicPtr<u8>,
    size: usize,
}
impl Arena {
    pub fn with_capacity(size: usize) -> Self {
        let (head, current) = {
            let arena = unsafe { alloc_zeroed(Layout::from_size_align_unchecked(size, 16)) };
            (AtomicPtr::new(arena), AtomicPtr::new(arena))
        };

        Self {
            head,
            current,
            size,
        }
    }

    pub fn alloc<T>(&self, value: T) -> &mut T {
        self.alloc_with(|| value)
    }

    #[inline(always)]
    pub fn alloc_with<F, T>(&self, f: F) -> &mut T
    where
        F: FnOnce() -> T,
    {
        #[inline(always)]
        unsafe fn inner_writer<T, F>(ptr: *mut T, f: F)
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
            std::ptr::write(ptr, f())
        }

        let layout = Layout::new::<T>();
        let size = layout.size() + (layout.size() % layout.align());

        let mut current = self.current.load(Ordering::SeqCst);

        loop {
            if self
                .current
                .compare_exchange_weak(
                    current,
                    unsafe { current.add(size) },
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                )
                .unwrap()
                == current
            {
                break;
            } else {
                current = self.current.load(Ordering::SeqCst);
            }
        }

        unsafe {
            let res = current as *mut T;

            inner_writer(res, f);

            &mut *res
        }
    }

    pub fn reset(&mut self) {
        *self.current.get_mut() = self.head.load(Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn simple() {
        let arena = Arena::with_capacity(524_288_000);
    }
}

use std::cell::{Cell, RefCell, Ref, RefMut};
use std::cmp::{max};
use std::rc::{Rc};

pub struct LazyVec<T> {
  buf:  Rc<RefCell<Option<Vec<T>>>>,
  capacity: Cell<usize>,
}

impl<T> LazyVec<T> {
  pub fn new() -> LazyVec<T> {
    LazyVec{
      buf:  Rc::new(RefCell::new(None)),
      capacity: Cell::new(0),
    }
  }

  pub fn require_capacity(&self, min_cap: usize) {
    assert!(self.buf.borrow().is_none());
    self.capacity.set(max(min_cap, self.capacity.get()));
  }

  pub fn borrow(&self) -> Ref<Vec<T>> {
    if self.buf.borrow().is_none() {
      let mut buf = self.buf.borrow_mut();
      *buf = Some(Vec::with_capacity(self.capacity.get()));
    }
    Ref::map(self.buf.borrow(), |maybe_buf| maybe_buf.as_ref().unwrap())
  }

  pub fn borrow_mut(&self) -> RefMut<Vec<T>> {
    if self.buf.borrow().is_none() {
      let mut buf = self.buf.borrow_mut();
      *buf = Some(Vec::with_capacity(self.capacity.get()));
    }
    RefMut::map(self.buf.borrow_mut(), |maybe_buf| maybe_buf.as_mut().unwrap())
  }
}

use std::cell::{RefCell};
use std::rc::{Rc};

#[derive(Clone)]
pub struct CommonOperatorOutput<T> where T: Copy {
  pub batch_size:   usize,
  pub out_buf:      Rc<RefCell<Vec<T>>>,
  pub out_grad:     Rc<RefCell<Option<Vec<T>>>>,
  pub out_r_buf:    Rc<RefCell<Option<Vec<T>>>>,
  pub out_r_grad:   Rc<RefCell<Option<Vec<T>>>>,
}

#[derive(Clone, Copy, Debug)]
pub enum ActivationKind {
  Identity,
  Rect,
  Logistic,
  Tanh,
}

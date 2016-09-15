use super::{OpCapability};

use std::cell::{RefCell};
use std::rc::{Rc};

#[derive(Clone)]
pub struct CommonOperatorOutput<T> where T: Copy {
  pub batch_size:   usize,
  pub out_buf:      Rc<RefCell<Vec<T>>>,
  pub out_grad:     Option<Rc<RefCell<Vec<T>>>>,
  pub out_r_buf:    Option<Rc<RefCell<Vec<T>>>>,
  pub out_r_grad:   Option<Rc<RefCell<Vec<T>>>>,
}

impl CommonOperatorOutput<f32> {
  pub fn new(batch_size: usize, frame_size: usize, cap: OpCapability) -> Self {
    let mut out_buf = Vec::with_capacity(batch_size * frame_size);
    unsafe { out_buf.set_len(batch_size * frame_size) };
    let out_grad = if cap.enable_backward() {
      let mut out_grad = Vec::with_capacity(batch_size * frame_size);
      unsafe { out_grad.set_len(batch_size * frame_size) };
      Some(Rc::new(RefCell::new(out_grad)))
    } else {
      None
    };
    CommonOperatorOutput{
      batch_size:   batch_size,
      out_buf:      Rc::new(RefCell::new(out_buf)),
      out_grad:     out_grad,
      out_r_buf:    None,
      out_r_grad:   None,
    }
  }
}

#[derive(Clone, Copy, Debug)]
pub enum ActivationKind {
  Identity,
  Rect,
  Logistic,
  Tanh,
}

#[derive(Clone, Copy, Debug)]
pub enum ParamInitKind {
  Disabled,
  Uniform{lo: f32, hi: f32},
  Normal{mean: f32, std: f32},
  Xavier,
  Kaiming,
}

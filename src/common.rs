use operator::{OpCapability};
use sharedmem::{RwMem};

use std::cell::{RefCell};
use std::rc::{Rc};

pub trait ArmOutput {
  type Output;

  fn _output(&self, arm: usize) -> Self::Output;
}

#[derive(Clone)]
pub struct CommonOperatorOutput<T> where T: Copy {
  pub batch_size:   usize,
  /*pub out_buf:      Rc<RefCell<Vec<T>>>,
  pub out_grad:     Option<Rc<RefCell<Vec<T>>>>,
  pub out_r_buf:    Option<Rc<RefCell<Vec<T>>>>,
  pub out_r_grad:   Option<Rc<RefCell<Vec<T>>>>,*/
  pub out_buf:      RwMem<T>,
  pub out_grad:     Option<RwMem<T>>,
  pub out_r_buf:    Option<RwMem<T>>,
  pub out_r_grad:   Option<RwMem<T>>,
}

impl CommonOperatorOutput<f32> {
  pub fn new(batch_size: usize, frame_size: usize, cap: OpCapability) -> Self {
    let mut out_buf = Vec::with_capacity(batch_size * frame_size);
    unsafe { out_buf.set_len(batch_size * frame_size) };
    let out_buf = RwMem::new(out_buf);
    let out_grad = if cap.enable_backward() {
      let mut out_grad = Vec::with_capacity(batch_size * frame_size);
      unsafe { out_grad.set_len(batch_size * frame_size) };
      Some(RwMem::new(out_grad))
    } else {
      None
    };
    CommonOperatorOutput{
      batch_size:   batch_size,
      out_buf:      out_buf,
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

#[derive(Clone, Copy, Debug)]
pub enum PoolKind {
  Max,
  Average,
}

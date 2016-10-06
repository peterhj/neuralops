use common::{ActivationKind};

use nnpack::{NnpackHandle, NnpackPthreadPool};
use nnpack::ffi::*;

use libc::{size_t};
use std::ptr::{null_mut};
use std::rc::{Rc};

#[link(name = "neuralops_extkernels", kind = "static")]
extern "C" {
  pub fn neuralops_rect_fwd(
      batch_sz: size_t,
      dim: size_t,
      in_buf: *const f32,
      out_buf: *mut f32,
  );
  pub fn neuralops_rect_bwd(
      batch_sz: size_t,
      dim: size_t,
      in_buf: *const f32,
      out_grad: *const f32,
      in_grad: *mut f32,
  );
}

pub struct ActivateKernel {
  batch_sz: usize,
  dim:      usize,
  act_kind: ActivationKind,
  //nnp_h:    NnpackHandle,
  //nnp_pool: Rc<NnpackPthreadPool>,
}

impl ActivateKernel {
  pub fn new(batch_sz: usize, dim: usize, act_kind: ActivationKind, nnp_pool: Rc<NnpackPthreadPool>) -> ActivateKernel {
    ActivateKernel{
      batch_sz: batch_sz,
      dim:      dim,
      act_kind: act_kind,
      //nnp_h:    NnpackHandle::new(),
      //nnp_pool: nnp_pool,
    }
  }

  pub fn forward(&mut self, batch_sz: usize, in_buf: &[f32], out_buf: &mut [f32]) {
    assert!(batch_sz <= self.batch_sz);
    match self.act_kind {
      ActivationKind::Identity => {
        out_buf.copy_from_slice(in_buf);
      }
      ActivationKind::Rect => {
        /*let status = unsafe { nnp_relu_output(
            batch_sz,
            self.dim,
            in_buf.as_ptr(),
            out_buf.as_mut_ptr(),
            0.0,
            //self.nnp_pool.as_raw(),
            null_mut(),
        ) };
        assert!(status.is_ok());*/
        unsafe { neuralops_rect_fwd(
            batch_sz,
            self.dim,
            in_buf.as_ptr(),
            out_buf.as_mut_ptr(),
        ) };
      }
      ActivationKind::Logistic => {
        unimplemented!();
      }
      ActivationKind::Tanh => {
        unimplemented!();
      }
    }
  }

  pub fn backward(&mut self, batch_sz: usize, in_buf: &[f32], out_grad: &[f32], in_grad: &mut [f32]) {
    match self.act_kind {
      ActivationKind::Identity => {
        in_grad.copy_from_slice(out_grad);
      }
      ActivationKind::Rect => {
        /*let status = unsafe { nnp_relu_input_gradient(
            batch_sz,
            self.dim,
            out_grad.as_ptr(),
            in_buf.as_ptr(),
            in_grad.as_mut_ptr(),
            0.0,
            //self.nnp_pool.as_raw(),
            null_mut(),
        ) };
        assert!(status.is_ok());*/
        unsafe { neuralops_rect_bwd(
            batch_sz,
            self.dim,
            in_buf.as_ptr(),
            out_grad.as_ptr(),
            in_grad.as_mut_ptr(),
        ) };
      }
      ActivationKind::Logistic => {
        unimplemented!();
      }
      ActivationKind::Tanh => {
        unimplemented!();
      }
    }
  }
}
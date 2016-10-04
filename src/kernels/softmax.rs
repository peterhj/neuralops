use nnpack::{NnpackHandle, NnpackPthreadPool};
use nnpack::ffi::*;

use std::ptr::{null_mut};
use std::rc::{Rc};

pub struct SoftmaxKernel {
  batch_sz: usize,
  dim:      usize,
  _nnp_h:   NnpackHandle,
  nnp_pool: Rc<NnpackPthreadPool>,
}

impl SoftmaxKernel {
  pub fn new(batch_sz: usize, dim: usize, nnp_pool: Rc<NnpackPthreadPool>) -> SoftmaxKernel {
    SoftmaxKernel{
      batch_sz: batch_sz,
      dim:      dim,
      _nnp_h:   NnpackHandle::new(),
      nnp_pool: nnp_pool,
    }
  }

  pub fn forward(&mut self, batch_sz: usize, in_buf: &[f32], out_buf: &mut [f32]) {
    assert!(batch_sz <= self.batch_sz);
    let status = unsafe { nnp_softmax_output(
        batch_sz,
        self.dim,
        in_buf.as_ptr(),
        out_buf.as_mut_ptr(),
        //self.nnp_pool.as_raw(),
        null_mut(),
    ) };
    assert!(status.is_ok());
  }
}

use prelude::*;
use kernels::ffi::*;

use densearray::prelude::*;

pub struct ActivateKernel {
  batch_sz: usize,
  dim:      usize,
  act_kind: ActivationKind,
}

impl ActivateKernel {
  pub fn new(batch_sz: usize, dim: usize, act_kind: ActivationKind) -> ActivateKernel {
    ActivateKernel{
      batch_sz: batch_sz,
      dim:      dim,
      act_kind: act_kind,
    }
  }

  pub fn forward(&mut self, batch_sz: usize, in_buf: &[f32], out_buf: &mut [f32]) {
    assert!(batch_sz <= self.batch_sz);
    match self.act_kind {
      ActivationKind::Identity => {
        out_buf.copy_from_slice(in_buf);
      }
      ActivationKind::Rect => {
        unsafe { neuralops_rect_fwd(
            batch_sz,
            self.dim,
            in_buf.as_ptr(),
            out_buf.as_mut_ptr(),
        ) };
      }
      ActivationKind::LeakyRect(_) => {
        unimplemented!();
      }
      ActivationKind::Logistic => {
        unsafe { neuralops_logistic_fwd(
            batch_sz,
            self.dim,
            in_buf.as_ptr(),
            out_buf.as_mut_ptr(),
        ) };
      }
      ActivationKind::Tanh => {
        unimplemented!();
      }
    }
  }

  pub fn backward(&mut self, batch_sz: usize, out_buf: &[f32], out_grad: &[f32], in_grad: &mut [f32]) {
    match self.act_kind {
      ActivationKind::Identity => {
        in_grad.copy_from_slice(out_grad);
      }
      ActivationKind::Rect => {
        unsafe { neuralops_rect_bwd(
            batch_sz,
            self.dim,
            out_buf.as_ptr(),
            out_grad.as_ptr(),
            in_grad.as_mut_ptr(),
        ) };
      }
      ActivationKind::LeakyRect(_) => {
        unimplemented!();
      }
      ActivationKind::Logistic => {
        unsafe { neuralops_logistic_bwd(
            batch_sz,
            self.dim,
            out_buf.as_ptr(),
            out_grad.as_ptr(),
            in_grad.as_mut_ptr(),
        ) };
      }
      ActivationKind::Tanh => {
        unimplemented!();
      }
    }
  }
}

pub struct ParallelActivateKernel {
  batch_sz: usize,
  dim:      usize,
  act_kind: ActivationKind,
}

impl ParallelActivateKernel {
  pub fn new(batch_sz: usize, dim: usize, act_kind: ActivationKind) -> ParallelActivateKernel {
    ParallelActivateKernel{
      batch_sz: batch_sz,
      dim:      dim,
      act_kind: act_kind,
    }
  }

  pub fn forward(&mut self, batch_sz: usize, in_buf: &[f32], out_buf: &mut [f32]) {
    assert!(batch_sz <= self.batch_sz);
    match self.act_kind {
      ActivationKind::Identity => {
        //out_buf.copy_from_slice(in_buf);
        out_buf.reshape_mut(batch_sz * self.dim).parallel_copy(in_buf.reshape(batch_sz * self.dim));
      }
      ActivationKind::Rect => {
        unsafe { neuralops_omp_rect_fwd(
            batch_sz,
            self.dim,
            in_buf.as_ptr(),
            out_buf.as_mut_ptr(),
        ) };
      }
      ActivationKind::LeakyRect(_) => {
        unimplemented!();
      }
      ActivationKind::Logistic => {
        unsafe { neuralops_omp_logistic_fwd(
            batch_sz,
            self.dim,
            in_buf.as_ptr(),
            out_buf.as_mut_ptr(),
        ) };
      }
      ActivationKind::Tanh => {
        unimplemented!();
      }
    }
  }

  pub fn backward(&mut self, batch_sz: usize, out_buf: &[f32], out_grad: &[f32], in_grad: &mut [f32]) {
    match self.act_kind {
      ActivationKind::Identity => {
        //in_grad.copy_from_slice(out_grad);
        in_grad.reshape_mut(batch_sz * self.dim).parallel_copy(out_grad.reshape(batch_sz * self.dim));
      }
      ActivationKind::Rect => {
        unsafe { neuralops_omp_rect_bwd(
            batch_sz,
            self.dim,
            out_buf.as_ptr(),
            out_grad.as_ptr(),
            in_grad.as_mut_ptr(),
        ) };
      }
      ActivationKind::LeakyRect(_) => {
        unimplemented!();
      }
      ActivationKind::Logistic => {
        unsafe { neuralops_omp_logistic_bwd(
            batch_sz,
            self.dim,
            out_buf.as_ptr(),
            out_grad.as_ptr(),
            in_grad.as_mut_ptr(),
        ) };
      }
      ActivationKind::Tanh => {
        unimplemented!();
      }
    }
  }
}

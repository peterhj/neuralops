use common::{ActivationKind};

pub mod activate;
pub mod batchnorm;
pub mod conv;
pub mod image;
pub mod interpolate;
pub mod pool;
pub mod softmax;

pub fn activate_fwd(act_kind: ActivationKind, in_buf: &[f32], out_buf: &mut [f32]) {
  match act_kind {
    ActivationKind::Identity => {
      out_buf.copy_from_slice(in_buf);
    }
    ActivationKind::Rect => {
      rect_fwd(in_buf, out_buf);
    }
    ActivationKind::Logistic => {
      unimplemented!();
    }
    ActivationKind::Tanh => {
      unimplemented!();
    }
    _ => unimplemented!(),
  }
}

pub fn activate_bwd(act_kind: ActivationKind, in_buf: &[f32], out_grad: &[f32], in_grad: &mut [f32]) {
  match act_kind {
    ActivationKind::Identity => {
      in_grad.copy_from_slice(out_grad);
    }
    ActivationKind::Rect => {
      rect_bwd(in_buf, out_grad, in_grad);
    }
    ActivationKind::Logistic => {
      unimplemented!();
    }
    ActivationKind::Tanh => {
      unimplemented!();
    }
    _ => unimplemented!(),
  }
}

pub fn rect_fwd(in_buf: &[f32], out_buf: &mut [f32]) {
  let n = in_buf.len();
  assert_eq!(n, out_buf.len());
  for i in 0 .. n {
    //out_buf[i] = in_buf[i].max(0.0);
    if in_buf[i] > 0.0 {
      out_buf[i] = in_buf[i];
    } else {
      out_buf[i] = 0.0;
    }
  }
}

pub fn rect_bwd(in_buf: &[f32], out_grad: &[f32], in_grad: &mut [f32]) {
  let n = in_buf.len();
  assert_eq!(n, out_grad.len());
  assert_eq!(n, in_grad.len());
  for i in 0 .. n {
    if in_buf[i] > 0.0 {
      in_grad[i] = out_grad[i];
    } else {
      in_grad[i] = 0.0
    }
  }
}

/*pub fn pool_fwd(pool_kind: PoolKind, in_buf: &[f32], out_buf: &mut [f32]) {
}*/

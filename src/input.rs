use common::{CommonOperatorOutput};

use operator::{Operator, InternalOperator, OpPhase};
use operator::data::{ClassSample};

//use anymap::{AnyMap};
//use typemap::{TypeMap};

use std::marker::{PhantomData};

pub struct InputOperator {
  batch_cap:    usize,
  frame_sz:     usize,
  in_buf:       Vec<f32>,
  out:          CommonOperatorOutput<f32>,
}

impl Operator<f32, ClassSample<f32>> for InputOperator {
  //type Sample = ClassSample<f32>;

  fn load_data(&mut self, samples: &[ClassSample<f32>]) {
    let batch_size = samples.len();
    assert!(batch_size <= self.batch_cap);
    for (idx, sample) in samples.iter().enumerate() {
      assert_eq!(self.frame_sz, sample.input.len());
      self.in_buf[idx * self.frame_sz .. (idx+1) * self.frame_sz].copy_from_slice(&sample.input);
    }
    self.out.batch_size = batch_size;
  }
}

impl Operator<f32, ClassSample<u8>> for InputOperator {
  //type Sample = ClassSample<f32>;

  fn load_data(&mut self, samples: &[ClassSample<u8>]) {
    let batch_size = samples.len();
    assert!(batch_size <= self.batch_cap);
    for (idx, sample) in samples.iter().enumerate() {
      assert_eq!(self.frame_sz, sample.input.len());
      for j in 0 .. self.frame_sz {
        self.in_buf[idx * self.frame_sz + j] = sample.input[j] as f32;
      }
    }
    self.out.batch_size = batch_size;
  }
}

/*impl<S> Operator<f32, S> for InputOperator<S> where S: 'static + SampleCastAs<ClassSample<f32>> {
  type Sample = ClassSample<f32>;

  fn load_data(&mut self, samples: &[S]) {
    match self.cast_map.get::<S>() {
      None    => {}
      Some(_) => {}
    }
    unimplemented!();
  }
}*/

/*impl Operator<f32> for InputOperator {
  type Sample = ClassSample<f32>;

  fn load_data<S>(&mut self, samples: &[S]) where S: SampleCastAs<ClassSample<f32>> {
    /*match self.cast_map.get::<S>() {
      None    => {}
      Some(_) => {}
    }*/
    unimplemented!();
  }
}*/

impl InternalOperator<f32> for InputOperator {
  type Output = CommonOperatorOutput<f32>;

  fn output(&self, _arm: usize) -> CommonOperatorOutput<f32> {
    assert_eq!(0, _arm);
    self.out.clone()
  }

  fn forward(&mut self, _phase: OpPhase) {
    self.out.out_buf.borrow_mut().copy_from_slice(&self.in_buf);
  }

  fn backward(&mut self) {
  }
}

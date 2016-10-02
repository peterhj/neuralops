use super::prelude::*;
use super::ops::*;

use operator::prelude::*;

use std::collections::{HashMap};

pub struct GraphOperatorConfig {
}

impl GraphOperatorConfig {
  pub fn simple_input(&mut self, name: &str, prev_name: &str, cfg: SimpleInputOperatorConfig) -> &mut Self {
    unimplemented!();
  }

  pub fn affine(&mut self, name: &str, prev_name: &str, cfg: AffineOperatorConfig) -> &mut Self {
    unimplemented!();
  }

  pub fn conv2d(&mut self, name: &str, prev_name: &str, cfg: Conv2dOperatorConfig) -> &mut Self {
    unimplemented!();
  }

  pub fn softmax_nll_class_loss(&mut self, name: &str, prev_name: &str, cfg: ClassLossConfig) -> &mut Self {
    unimplemented!();
  }
}

pub struct GraphOperator {
}

impl GraphOperator {
  pub fn new(cfg: GraphOperatorConfig) -> GraphOperator {
    unimplemented!();
  }
}

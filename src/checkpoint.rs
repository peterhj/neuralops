use operator::{CheckpointFormat};

pub struct CaffeProtobufFormat {}

impl CheckpointFormat for CaffeProtobufFormat {}

pub struct TensorflowChkptFormat {}

impl CheckpointFormat for TensorflowChkptFormat {}

pub struct MsgpackFormat {}

impl CheckpointFormat for MsgpackFormat {}

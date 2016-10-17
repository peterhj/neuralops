use data::{IndexedDataShard, OwnedSample};

use varraydb::{VarrayDb};

use std::path::{Path};

pub struct VarrayDbShard {
  db:           VarrayDb,
  start_idx:    usize,
  end_idx:      usize,
}

impl VarrayDbShard {
  pub fn open(prefix: &Path) -> VarrayDbShard {
    let db = VarrayDb::open(prefix).unwrap();
    let len = db.len();
    VarrayDbShard{
      db:           db,
      start_idx:    0,
      end_idx:      len,
    }
  }

  pub fn open_partition(prefix: &Path, part: usize, num_parts: usize) -> VarrayDbShard {
    let mut shard = Self::open(prefix);
    unimplemented!();
  }

  pub fn open_range(prefix: &Path, start_idx: usize, end_idx: usize) -> VarrayDbShard {
    let mut shard = Self::open(prefix);
    shard.start_idx = start_idx;
    shard.end_idx = end_idx;
    shard
  }
}

impl IndexedDataShard<OwnedSample<u8>> for VarrayDbShard {
  fn len(&self) -> usize {
    self.end_idx - self.start_idx
  }

  fn get(&mut self, offset: usize) -> OwnedSample<u8> {
    let idx = self.start_idx + offset;
    assert!(idx >= self.start_idx);
    assert!(idx < self.end_idx);
    let value = self.db.get(idx);
    OwnedSample{
      input:    value.to_owned(),
    }
  }
}

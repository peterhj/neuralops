use data::{IndexedDataShard};

use operator::prelude::*;
//use varraydb::{VarrayDb};
use varraydb::shared::{SharedVarrayDb};

use std::path::{PathBuf};

/*pub struct VarrayDbShard {
  prefix:       PathBuf,
  db:           VarrayDb,
  start_idx:    usize,
  end_idx:      usize,
}

impl VarrayDbShard {
  pub fn open(prefix: PathBuf) -> VarrayDbShard {
    let db = VarrayDb::open(&prefix).unwrap();
    let len = db.len();
    VarrayDbShard{
      prefix:       prefix,
      db:           db,
      start_idx:    0,
      end_idx:      len,
    }
  }

  pub fn open_partition(prefix: PathBuf, part: usize, num_parts: usize) -> VarrayDbShard {
    let mut shard = Self::open(prefix);
    unimplemented!();
  }

  pub fn open_range(prefix: PathBuf, start_idx: usize, end_idx: usize) -> VarrayDbShard {
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
}*/

pub struct SharedVarrayDbShard {
  prefix:       PathBuf,
  db:           SharedVarrayDb,
  start_idx:    usize,
  end_idx:      usize,
}

impl SharedVarrayDbShard {
  pub fn open(prefix: PathBuf) -> SharedVarrayDbShard {
    let db = SharedVarrayDb::open(&prefix).unwrap();
    let len = db.len();
    SharedVarrayDbShard{
      prefix:       prefix,
      db:           db,
      start_idx:    0,
      end_idx:      len,
    }
  }

  pub fn open_partition(prefix: PathBuf, part: usize, num_parts: usize) -> SharedVarrayDbShard {
    //let shard = Self::open(prefix);
    unimplemented!();
  }

  pub fn open_range(prefix: PathBuf, start_idx: usize, end_idx: usize) -> SharedVarrayDbShard {
    let mut shard = Self::open(prefix);
    shard.start_idx = start_idx;
    shard.end_idx = end_idx;
    shard
  }
}

impl IndexedDataShard<SampleItem> for SharedVarrayDbShard {
  fn len(&self) -> usize {
    self.end_idx - self.start_idx
  }

  fn get(&mut self, offset: usize) -> SampleItem {
    let idx = self.start_idx + offset;
    assert!(idx >= self.start_idx);
    assert!(idx < self.end_idx);
    let value = self.db.get(idx);
    let mut item = SampleItem::new();
    item.kvs.insert::<SampleSharedSliceDataKey<u8>>(value);
    //item.insert::<SampleSharedExtractInputKey<f32>>(value);
    item
  }
}

use std::sync::{Arc};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{SyncSender, Receiver, sync_channel};
use std::thread::{JoinHandle, sleep, spawn};
use std::time::{Duration};

pub struct RRQueueSink<S, Iter> {
  num_workers:  usize,
  max_fill:     usize,
  fill_count:   Arc<AtomicUsize>,
  iter:         Iter,
  rr_offset:    usize,
  src_txs:      Vec<SyncSender<Option<S>>>,
}

impl<S, Iter> RRQueueSink<S, Iter> where Iter: Iterator<Item=S> {
  /*pub fn new() -> Self {
    unimplemented!();
  }*/

  pub fn runloop(&mut self) {
    let sleep_duration = Duration::from_millis(5);
    loop {
      if self.fill_count.load(Ordering::Acquire) >= self.max_fill {
        sleep(sleep_duration);
        continue;
      }
      let item = self.iter.next();
      let is_term = item.is_none();
      self.fill_count.fetch_add(1, Ordering::AcqRel);
      match self.src_txs[self.rr_offset].send(item) {
        Err(e) => {
          println!("WARNING: src send error: {:?}", e);
          break;
        }
        Ok(_) => {}
      }
      self.rr_offset = (self.rr_offset + 1) % self.num_workers;
      if is_term {
        break;
      }
    }
  }
}

pub struct RRQueueSrc<S> {
  is_term:      bool,
  rr_rank:      usize,
  src_rx:       Receiver<Option<S>>,
}

impl<S> Iterator for RRQueueSrc<S> {
  type Item = S;

  fn next(&mut self) -> Option<S> {
    if self.is_term {
      return None;
    }
    match self.src_rx.recv() {
      Err(e) => {
        println!("WARNING: src recv error: {:?}", e);
        return None;
      }
      Ok(item) => {
        if item.is_none() {
          self.is_term = true;
        }
        item
      }
    }
  }
}

pub struct RRQueueWorker<S, WorkIter> {
  work_iter:    WorkIter,
  work_tx:      SyncSender<Option<S>>,
}

impl<S, WorkIter> RRQueueWorker<S, WorkIter> where WorkIter: Iterator<Item=S> {
  /*pub fn new() -> Self {
    unimplemented!();
  }*/

  pub fn runloop(&mut self) {
    loop {
      let item = self.work_iter.next();
      let is_term = item.is_none();
      match self.work_tx.send(item) {
        Err(e) => {
          println!("WARNING: worker send error: {:?}", e);
          break;
        }
        Ok(_) => {}
      }
      if is_term {
        break;
      }
    }
  }
}

pub struct RoundRobinQueueDataIter<S> {
  num_workers:  usize,
  max_fill:     usize,
  fill_count:   Arc<AtomicUsize>,
  is_term:      bool,
  rr_offset:    usize,
  work_rxs:     Vec<Receiver<Option<S>>>,
  worker_hs:    Vec<JoinHandle<()>>,
  sink_h:       Option<JoinHandle<()>>,
}

impl<S> Drop for RoundRobinQueueDataIter<S> {
  fn drop(&mut self) {
    // XXX(20161129): Do not wait to join the threads.
    /*for h in self.worker_hs.drain(..) {
      h.join().unwrap();
    }
    self.sink_h.take().unwrap().join().unwrap();*/
  }
}

impl<S> RoundRobinQueueDataIter<S> where S: Send + 'static {
  pub fn new<F, WorkIter, Iter>(num_workers: usize, max_fill: usize, mut builder: F, iter: Iter) -> Self
  where F: FnMut(RRQueueSrc<S>) -> WorkIter,
        WorkIter: Iterator<Item=S> + Send + 'static,
        Iter: Iterator<Item=S> + Send + 'static,
  {
    assert!(num_workers <= max_fill);
    let max_fill_per_worker = (num_workers + max_fill - 1) / max_fill;
    let fill_count = Arc::new(AtomicUsize::new(0));
    let mut src_txs = Vec::with_capacity(num_workers);
    let mut work_rxs = Vec::with_capacity(num_workers);
    let mut worker_hs = Vec::with_capacity(num_workers);
    for rr_rank in 0 .. num_workers {
      let (src_tx, src_rx) = sync_channel(2 * max_fill_per_worker);
      let (work_tx, work_rx) = sync_channel(2 * max_fill_per_worker);
      src_txs.push(src_tx);
      work_rxs.push(work_rx);
      let src = RRQueueSrc{
        is_term:    false,
        rr_rank:    rr_rank,
        src_rx:     src_rx,
      };
      let work_iter = builder(src);
      worker_hs.push(spawn(move || {
        let mut worker = RRQueueWorker{
          work_iter:    work_iter,
          work_tx:      work_tx,
        };
        worker.runloop();
      }));
    }
    let sink_h = {
      let fill_count = fill_count.clone();
      spawn(move || {
        let mut sink = RRQueueSink{
          num_workers:  num_workers,
          max_fill:     max_fill,
          fill_count:   fill_count,
          iter:         iter,
          rr_offset:    0,
          src_txs:      src_txs,
        };
        sink.runloop();
      })
    };
    RoundRobinQueueDataIter{
      num_workers:  num_workers,
      max_fill:     max_fill,
      fill_count:   fill_count,
      is_term:      false,
      rr_offset:    0,
      work_rxs:     work_rxs,
      worker_hs:    worker_hs,
      sink_h:       Some(sink_h),
    }
  }
}

impl<S> Iterator for RoundRobinQueueDataIter<S> where S: Send {
  type Item = S;

  fn next(&mut self) -> Option<S> {
    if self.is_term {
      return None;
    }
    match self.work_rxs[self.rr_offset].recv() {
      Err(e) => {
        println!("WARNING: iter recv error: {:?}", e);
        return None;
      }
      Ok(item) => {
        if item.is_none() {
          self.is_term = true;
        }
        self.fill_count.fetch_sub(1, Ordering::AcqRel);
        self.rr_offset = (self.rr_offset + 1) % self.num_workers;
        item
      }
    }
  }
}

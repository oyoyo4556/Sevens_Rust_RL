use crate::common::Experience;
use rand::seq::IndexedRandom;

pub struct ReplayBuffer {
    capacity:usize,
    buffer:Vec<Experience>,
    pos:usize,
}

impl ReplayBuffer {
    pub fn new(capacity:usize) -> Self {
        Self {
            capacity,
            buffer:Vec::with_capacity(capacity),
            pos:0,
        }
    }

    pub fn add(&mut self,exp:Experience) {
        if self.buffer.len() < self.capacity {
            self.buffer.push(exp)
        } else {
            self.buffer[self.pos] = exp;
            self.pos =(self.pos + 1) % self.capacity;
        }
    }

    pub fn sample(&self,batch_size:usize) -> Vec<&Experience> {
        let mut rng = rand::rng();
        self.buffer.choose_multiple(&mut rng,batch_size).collect()
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }
}
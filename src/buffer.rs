use crate::common::Experience;
use rand::seq::IndexedRandom;
use crate::sumtree::SumTree;
use rand::Rng;

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

pub struct PrioritizedReplayBuffer {
    pub capacity:usize,
    pub tree:SumTree,
    buffer:Vec<Option<Experience>>,
    cursor:usize,
    full:bool,
    pub alpha:f32,
    pub beta:f32,
    pub beta_increment:f32,
    pub max_priority:f32,
}

impl PrioritizedReplayBuffer {
    pub fn new(capacity:usize,alpha:f32,beta:f32,beta_increment:f32) -> Self{
        let cap = if capacity.is_power_of_two(){
            capacity
        } else {
            capacity.next_power_of_two()
        };
        //capcityは2の累乗にしないといけない(二分木だから)。capacityに大きすぎるものはいれないでね
        Self {
            capacity:cap,
            tree:SumTree::new(cap),
            buffer:vec![None;cap],
            cursor:0,
            full:false,
            alpha,
            beta,
            beta_increment,
            max_priority:1.0,
        }
    }

    pub fn add(&mut self,exp:Experience) {
        let p = self.max_priority.powf(self.alpha);
        self.buffer[self.cursor] = Some(exp);
        self.tree.update(self.cursor,p);

        self.cursor = (self.cursor + 1) % self.capacity;
        if self.cursor == 0 {
            self.full = true;
        }
    }

    pub fn update_priorities(&mut self,indices:&[usize],errors:&[f32]){
        for (&idx,&error) in indices.iter().zip(errors.iter()) {
            let priority = error.abs() + 1e-5;
            if priority > self.max_priority {
                self.max_priority = priority;
            }
            let p_alpha = priority.powf(self.alpha);
            self.tree.update(idx,p_alpha);
        }
    }

    pub fn sample(& mut self,batch_size:usize) -> (Vec<&Experience>,Vec<usize>,Vec<f32>){

        let mut samples = Vec::with_capacity(batch_size);
        let mut indices = Vec::with_capacity(batch_size);
        let mut weights = Vec::with_capacity(batch_size);

        let total_p = self.tree.total_priority();
        let n = self.size() as f32;
        let segment = total_p/batch_size as f32;

        let min_prob = self.tree.min_priority() / total_p;
        let max_weight = (1.0 / (n * min_prob)).powf(self.beta);

        let  mut rng = rand::rng();

        for i in 0..batch_size {
            let a = segment * i as f32;
            let b = segment * (i+1) as f32;
            //Noneを引いたときの対策でloop処理
            loop {
                let s = rng.random_range(a..b);
                let data_idx = self.tree.get_leaf(s);

                if let Some(exp) = self.buffer[data_idx].as_ref() {
                    samples.push(exp);
                    indices.push(data_idx);

                    let p_i = self.tree.get_priority(data_idx);
                    let prob = p_i /total_p;
                    let weight = (1.0 / (n * prob)).powf(self.beta);
                    weights.push(weight / max_weight); //最大値で正規化

                    break;
                }
            }
        }

        //betaの更新。ifにしているのは1.0になった後に比較計算させないため。CPUが予測しやすいので早くなると思いたい。
        //minは1.0を超えないようにするために必要。
        if self.beta < 1.0 {
            self.beta = (self.beta + self.beta_increment).min(1.0);
        }


        (samples,indices,weights)
    }

    pub fn size(&self) -> usize {
        if self.full {
            self.capacity
        } else {
            self.cursor
        }
    }
}
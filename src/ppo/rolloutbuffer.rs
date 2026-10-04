use crate::common::PPOExperience;
use rand::seq::SliceRandom;

pub struct RolloutBuffer {
    pub exps: Vec<PPOExperience>,
    pub advantages: Vec<f32>,
    pub returns: Vec<f32>,
}

impl RolloutBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            exps: Vec::with_capacity(capacity),
            advantages: Vec::with_capacity(capacity),
            returns: Vec::with_capacity(capacity),
        }
    }

    /// バッファのクリア
    pub fn clear(&mut self) {
        self.exps.clear();
        self.advantages.clear();
        self.returns.clear();
    }

    /// 各Workerから集まった経験を一括追加
    pub fn push_batch(&mut self, mut new_exps: Vec<PPOExperience>) {
        self.exps.append(&mut new_exps);
    }

    pub fn compute_gae(&mut self, gamma: f32, gae_lambda: f32) {
        let len = self.exps.len();
        self.advantages = vec![0.0; len];
        self.returns = vec![0.0; len];

        let mut gae = 0.0;
        for t in (0..len).rev() {
            let exp = &self.exps[t];
            let next_value = if t == len - 1 || exp.done {
                0.0
            } else {
                self.exps[t + 1].value
            };

            let mask = if exp.done { 0.0 } else { 1.0 };
            let delta = exp.reward + gamma * next_value * mask - exp.value;
            gae = delta + gamma * gae_lambda * mask * gae;

            self.advantages[t] = gae;
            self.returns[t] = gae + exp.value;
        }

        // Advantage の正規化
        let mean = self.advantages.iter().sum::<f32>() / len as f32;
        let variance = self.advantages.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / len as f32;
        let std = (variance + 1e-8).sqrt();

        for a in self.advantages.iter_mut() {
            *a = (*a - mean) / std;
        }
    }

    pub fn get_batches<'a>(&'a self, batch_size: usize) -> Vec<PPOBatchItem<'a>> {
        let total_samples = self.exps.len();
        let mut indices: Vec<usize> = (0..total_samples).collect();
        indices.shuffle(&mut rand::rng());

        indices
            .chunks(batch_size)
            .map(|chunk| {
                let mut batch_exps = Vec::with_capacity(chunk.len());
                let mut advs = Vec::with_capacity(chunk.len());
                let mut rets = Vec::with_capacity(chunk.len());

                for &i in chunk {
                    batch_exps.push(&self.exps[i]); // 参照を入れるだけ
                    advs.push(self.advantages[i]);
                    rets.push(self.returns[i]);
                }

                PPOBatchItem {
                    exps: batch_exps,
                    advantages: advs,
                    returns: rets,
                }
            })
            .collect()
    }
}

pub struct PPOBatchItem<'a> {
    pub exps: Vec<&'a PPOExperience>, 
    pub advantages: Vec<f32>,
    pub returns: Vec<f32>,
}
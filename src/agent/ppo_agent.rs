use crate::agent::agent::{Agent, AgentResult};
use crate::common::{INPUT_STATE_DIM, TRAIN_AGENT_ID};
use crate::env::RawState;
use crate::ppo::ppo_nn::PPONet;
use crate::ppo::ppo_processor::PPOProcessor;
use crate::ppo::rolloutbuffer::PPOBatchItem;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use std::cell::RefCell;

pub struct PPOAgent {
    pub device: Device,
    pub varmap: VarMap,
    pub net: PPONet,
    optimizer: RefCell<AdamW>,
    pub processor: PPOProcessor,
    // PPO Hyperparameters
    pub clip_eps: f32, // 例: 0.2
    pub c1: f32,       // Value Loss の係数 (例: 0.5)
    pub c2: f32,       // Entropy Bonus の係数 (例: 0.01)
}

impl PPOAgent {
    pub fn new() -> Self {
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // state_dim: INPUT_STATE_DIM, hidden_dim: 512, action_dim: 53
        let net = PPONet::new(INPUT_STATE_DIM, 512, 53, vb.pp("ppo_net")).unwrap();

        let my_params = ParamsAdamW {
            lr: 1e-4, 
            weight_decay: 0.0,
            ..ParamsAdamW::default()
        };

        let optimizer = AdamW::new(varmap.all_vars(), my_params).unwrap();
        let processor = PPOProcessor::new(512);

        Self {
            device,
            varmap,
            net,
            optimizer: RefCell::new(optimizer),
            processor,
            clip_eps: 0.2,
            c1: 0.5,
            c2: 0.01,
        }
    }

    /// 行動選択 (推論・サンプリング) と同時に、PPOに必要な log_prob と value を返す
    pub fn select_action_with_info(
        &self,
        state: &RawState,
        player_id: &usize,
        deterministic: bool,
    ) -> Result<(u8, f32, f32)> {
        let mut buf = self.processor.infer_buf.borrow_mut();
        buf.clear();

        // 状態の書き込み (num_players: 4 固定/環境に合わせて調整)
        self.processor.write_buf(&mut buf, state, *player_id, 4);

        assert_eq!(
            buf.len(),
            INPUT_STATE_DIM,
            "【致命的バグ防止】bufの要素数が一致しません！"
        );

        let state_tensor = Tensor::from_slice(&buf, (1, INPUT_STATE_DIM), &self.device)?;
        let mask_tensor = Tensor::from_slice(&state.legal_actions_mask, (1, 53), &self.device)?;

        // 順伝播 (masked_logits: [1, 53], value: [1])
        let (masked_logits, value_tensor) = self.net.forward(&state_tensor, &mask_tensor)?;
        let value = value_tensor.to_vec1::<f32>()?[0];

        // Softmax で行動確率分布を得る
        let probs_tensor = candle_nn::ops::softmax(&masked_logits, 1)?;
        let probs = probs_tensor.flatten_all()?.to_vec1::<f32>()?;

        // 行動の決定
        let action = if deterministic {
            probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as u8)
                .unwrap_or(52)
        } else {
            // Categorical Sampling (収集時)
            let mut rng = rand::rng();
            let dist = WeightedIndex::new(&probs).map_err(|e| {
                candle_core::Error::Msg(format!("WeightedIndex error: {:?}", e))
            })?;
            dist.sample(&mut rng) as u8
        };

        let selected_prob = probs[action as usize].max(1e-8);
        let log_prob = selected_prob.ln();

        Ok((action, log_prob, value))
    }

    pub fn update_batch(&mut self, batch: &PPOBatchItem) -> Result<(f32,f32,f32,f32)> {
        let batch_size = batch.exps.len();
        if batch_size == 0 {
            return Ok((0.0,0.0,0.0,0.0));
        }

        let (states_t, _, masks_t, _, actions_t, _, _) =
            self.processor.batch_to_tensors(&batch.exps, &self.device, TRAIN_AGENT_ID, 4)?;

        // exps から直接 old_log_probs を抽出
        let old_log_probs_raw: Vec<f32> = batch.exps.iter().map(|e| e.log_prob).collect();
        let old_log_probs_t = Tensor::from_slice(&old_log_probs_raw, batch_size, &self.device)?;
        let advantages_t = Tensor::from_slice(&batch.advantages, batch_size, &self.device)?;
        let returns_t = Tensor::from_slice(&batch.returns, batch_size, &self.device)?;
        let actions_t = actions_t.to_dtype(DType::U32)?;

        // 現在のネットワークでの推論
        let (logits, values) = self.net.forward(&states_t, &masks_t)?;

        // 1. Log softmax & Selected action log_prob
        let log_probs_all = candle_nn::ops::log_softmax(&logits, 1)?;
        let probs_all = candle_nn::ops::softmax(&logits, 1)?;

        let actions_unsq = actions_t.unsqueeze(1)?;
        let log_probs = log_probs_all.gather(&actions_unsq, 1)?.squeeze(1)?;

        // 2. Policy Loss (Clipped Surrogate)
        let ratio = (&log_probs - &old_log_probs_t)?.exp()?;
        let surr1 = (&ratio * &advantages_t)?;
        let clamped_ratio = ratio.clamp((1.0 - self.clip_eps) as f64, (1.0 + self.clip_eps) as f64)?;
        let surr2 = (&clamped_ratio * &advantages_t)?;

        // 最小値をとってマイナス（最大化のためのマイナス）
        let policy_loss = surr1.minimum(&surr2)?.neg()?.mean(0)?;

        // 3. Value Loss (MSE)
        let value_loss = (&values - &returns_t)?.sqr()?.mean(0)?;

        // 4. Entropy Bonus
        let entropy = (probs_all.neg()? * log_probs_all)?.sum(1)?.mean(0)?;

        // Total Loss = Policy_Loss + c1 * Value_Loss - c2 * Entropy
        let total_loss = (&policy_loss
            + (&value_loss * self.c1 as f64)?
            - (&entropy * self.c2 as f64)?)?;

        let mut opt = self.optimizer.borrow_mut();
        opt.backward_step(&total_loss)?;

        let total_loss_val =total_loss.to_scalar::<f32>()?;
        let policy_loss_val =policy_loss.to_scalar::<f32>()?;
        let value_loss_val =value_loss.to_scalar::<f32>()?;
        let entropy_val =entropy.to_scalar::<f32>()?;


        Ok((total_loss_val,policy_loss_val,value_loss_val,entropy_val))
    }

    pub fn set_learning_rate(&mut self, lr: f64) {
        self.optimizer.borrow_mut().set_learning_rate(lr);
    }

    // 重み保存 / 読み込み（MainAgentと同等の設計方針）
    pub fn save(&self, path: &str) -> Result<()> {
        self.varmap.save(path)?;
        Ok(())
    }

    pub fn load(&mut self, path: &str) -> Result<()> {
        self.varmap.load(path)?;
        println!("PPO Model loaded from {}", path);
        Ok(())
    }

    pub fn copy_weights_to(&self, other: &mut PPOAgent) -> Result<()> {
        let updates = {
            let src_vars = self
                .varmap
                .data()
                .lock()
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let mut data = Vec::new();
            for (name, var) in src_vars.iter() {
                data.push((name.clone(), var.as_tensor().copy()?));
            }
            data
        };

        {
            let dst_vars = other
                .varmap
                .data()
                .lock()
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            for (name, tensor) in updates {
                if let Some(dst_var) = dst_vars.get(&name) {
                    dst_var.set(&tensor)?;
                }
            }
        }
        Ok(())
    }
}

// 既存の Agent トレイトの実装
impl Agent for PPOAgent {
    fn select_action(&self, state: &RawState, player_id: &usize) -> AgentResult<u8> {
        self.select_action_with_info(state, player_id, false)
            .map(|(action, _, _)| action)
            .map_err(|e| e.to_string())
    }
}
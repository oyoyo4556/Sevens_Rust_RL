use crate::env::{PASS_ACTION, RawState};
use std::cell::RefCell;
use candle_core::{Device,Result,Tensor};
use candle_nn::{AdamW,Optimizer,ParamsAdamW,VarBuilder,VarMap};
use std::collections::VecDeque;
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use crate::rnet::{DuelingQNet,RNet};
use crate::buffer::ReplayBuffer;
use crate::processor::Processor;
use crate::common::{Experience, TRAIN_AGENT_ID,INPUT_STATE_DIM};
use crate::agent::agent::{Agent,AgentResult};


pub struct DRNAgent {
    device:Device,
    pub varmap:VarMap,
    pub policy_net:DuelingQNet,
    pub target_net:DuelingQNet,
    pub regret_net:RNet,
    pub target_regret_net:RNet,
    q_optimizer:RefCell<AdamW>,
    reg_optimizer:RefCell<AdamW>,
    pub buffer:ReplayBuffer,
    gamma:f32,
    n_step_buffer:VecDeque<(RawState,u8,f32,RawState,bool)>,
    n_step: usize,
    lambda:f64,
    temp:f64,
    processor:Processor,
    action_buffer:RefCell<Vec<u8>>,
    weights_buffer:RefCell<Vec<f32>>,
}

impl DRNAgent {
    pub fn new(capacity:usize,n_step:usize) -> Self{
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap,
        candle_core::DType::F32,&device);
        let policy_net = DuelingQNet::new(INPUT_STATE_DIM,512,53,vb.pp("policy")).unwrap();
        let target_net = DuelingQNet::new(INPUT_STATE_DIM,512,53,vb.pp("target")).unwrap();
        let regret_net = RNet::new(INPUT_STATE_DIM,512,53,vb.pp("regret")).unwrap();
        let target_regret_net = RNet::new(INPUT_STATE_DIM,512,53,vb.pp("target_regret")).unwrap();

        let (q_vars,reg_vars) = {
            let all_vars = varmap.data().lock().map_err(|e|candle_core::Error::Msg(e.to_string())).unwrap();
            let mut q = Vec::new();
            let mut r = Vec::new();

            for (name,var) in all_vars.iter() {
                if name.starts_with("policy.") {
                    q.push(var.clone());
                } else if name.starts_with("regret.") {
                    r.push(var.clone());
                }
            }

            (q,r)
            //ここでdrop
        };

        let my_q_params = ParamsAdamW{
            lr:5e-5,
            weight_decay:0.0,
            ..ParamsAdamW::default()
        };

        let my_reg_params = ParamsAdamW{
            lr:5e-5,
            weight_decay:0.0,
            ..ParamsAdamW::default()
        };

        let q_optimizer = AdamW::new(q_vars,my_q_params).unwrap();
        let reg_optimizer = AdamW::new(reg_vars,my_reg_params).unwrap();
        let processor = Processor::new(256);

        Self { 
            device,
            varmap,
            policy_net,
            target_net,
            regret_net,
            target_regret_net,
            q_optimizer:RefCell::new(q_optimizer),
            reg_optimizer:RefCell::new(reg_optimizer),
            buffer: ReplayBuffer::new(capacity),
            gamma: 0.99,
            n_step_buffer:VecDeque::with_capacity(n_step),
            n_step, 
            lambda:0.0,
            temp:1.0,
            processor,
            action_buffer:RefCell::new(Vec::with_capacity(53)),
            weights_buffer:RefCell::new(Vec::with_capacity(53)),
        }
    }

    pub fn infer_q(&self, state: &RawState, player_id: &usize) -> Result<u8> {
        let mut rng = rand::rng();

        // 1. 値の準備
        let mut buf = self.processor.infer_buf.borrow_mut();
        buf.clear();
        self.processor.write_buf(&mut buf, state, *player_id, 4); // player_id:0, num_players:4

        let state_tensor = Tensor::from_slice(&buf, (1, INPUT_STATE_DIM), &self.device)?;
        let mask_tensor = Tensor::from_slice(&state.legal_actions_mask, (1, 53), &self.device)?;
        
        let q_values = self.policy_net.forward(&state_tensor, &mask_tensor)?;
        let reg_values = self.regret_net.forward(&state_tensor)?;

        // 2.(1 - λ) * Q - λ * R の計算 (凸結合)
        let lambda_f = self.lambda as f32;
        let scaled_q = q_values.affine((1.0 - lambda_f) as f64, 0.0)?;
        let scaled_r = reg_values.affine(lambda_f as f64, 0.0)?;
        let combined_values = scaled_q.sub(&scaled_r)?;

        // 3. 合法手以外をソフトマックスから除外する
        let neg_inf_t = mask_tensor.affine(-1.0, 1.0)?.affine(-1e9f64, 0.0)?;
        let masked_combined = combined_values.add(&neg_inf_t)?;

        // 4. 温度付きsoftmax。ここでは1.0
        let temperature = self.temp;
        let scaled_for_softmax = if temperature != 1.0 {
            masked_combined.affine(1.0 / temperature, 0.0)?
        } else {
            masked_combined
        };

        // 5. ソフトマックス関数で確率分布に変換
        let probs_tensor = candle_nn::ops::softmax(&scaled_for_softmax, 1)?;
        let probs_vec = probs_tensor.flatten_all()?.to_vec1::<f32>()?;

        // 6. 確率分布（重み）に基づいてランダムサンプリング
        // 合法手かつ確率が微小に存在するインデックスと重みを集める
        let mut valid_actions = self.action_buffer.borrow_mut();
        let mut weights = self.weights_buffer.borrow_mut();

        valid_actions.clear();
        weights.clear();

        for (i, (&p, &m)) in probs_vec.iter().zip(state.legal_actions_mask.iter()).enumerate() {
            if m == 1.0 && p > 0.0 {
                valid_actions.push(i as u8);
                weights.push(p);
            }
        }

        if valid_actions.is_empty() {
            return Err(candle_core::Error::Msg("No valid legal actions with non-zero probability".to_string()));
        }

        // 重み付きサンプリングの実行
        let dist = WeightedIndex::new(weights.as_slice())
            .map_err(|e| candle_core::Error::Msg(format!("WeightedIndex error: {}", e)))?;
        let chosen_idx = dist.sample(&mut rng);

        Ok(valid_actions[chosen_idx])
    }

    pub fn add_experience(&mut self,state:RawState,action:u8,reward:f32,next_state:RawState,done:bool) {
        self.n_step_buffer.push_back(
            (state,action,reward,next_state,done)
        );
        if done || self.n_step_buffer.len() >= self.n_step {
            while ! self.n_step_buffer.is_empty(){
                
                let (s_start,a_start,_,_,_) = &self.n_step_buffer[0];
                let mut discount_reward = 0.0;
                for (i,(_,_,r,_,_)) in self.n_step_buffer.iter().enumerate(){
                    discount_reward += r * self.gamma.powi(i as i32);
                }
                let next_gamma = self.gamma.powi(self.n_step_buffer.len() as i32);

                let (_,_,_,last_next_state,last_done) = self.n_step_buffer.back().expect(
                    "Failed to get Last element from n_step_buffer "
                );
                let exp = Experience {
                    state:s_start.clone(),
                    action:*a_start,
                    reward:discount_reward,
                    next_state:last_next_state.clone(),
                    done:*last_done,
                    next_gamma,
                };

                self.buffer.add(exp);

                self.n_step_buffer.pop_front();

                if !done {break;}
            }
        }
    }

    pub fn update(&mut self,batch_size:usize) -> Result<(f32,f32)> {
        if self.buffer.len() < batch_size{
            return Ok((0.0,0.0));
        }

        let batch = self.buffer.sample(batch_size) ;
        let (states_t,next_states_t,masks_t,next_masks_t,actions_t,rewards_t,dones_t,next_gammas_t) 
        = self.processor.batch_to_tensors(&batch, &self.device, TRAIN_AGENT_ID, 4)?;//player_id:0,num_players:4

        let actions_t =actions_t.to_dtype(candle_core::DType::U32)?; //gatherするため
        let not_done = (dones_t.ones_like()? - &dones_t)?;
        //=============================================
        // DQN / DuelingQNetの更新
        //=============================================
        let q_values = self.policy_net.forward(&states_t,&masks_t)?;
        let current_q = q_values.gather(&actions_t.unsqueeze(1)?,1)?.squeeze(1)?;

        let next_q_policy = self.policy_net.forward(&next_states_t,&next_masks_t)?;

        let neg_inf_t = next_masks_t.affine(-1.0,1.0)?.affine(-1e9f64,0.0)?;    
        let masked_next_q = next_q_policy.add(&neg_inf_t)?;
        let next_actions = masked_next_q.argmax(1)?;

        let next_q_values = self.target_net.forward(&next_states_t,&next_masks_t)?;
        let max_next_q = next_q_values.gather(&next_actions.unsqueeze(1)?,1)?.squeeze(1)?;
        let max_next_q = max_next_q.detach();

        let target_q = max_next_q.broadcast_mul(&next_gammas_t)?.broadcast_mul(&not_done)?.broadcast_add(&rewards_t)?;
        let q_loss = candle_nn::loss::huber(&current_q,&target_q,1.0)?;

        let mut q_opt = self.q_optimizer.borrow_mut();

        q_opt.backward_step(&q_loss)?;

        //=============================================
        // DRN / RNetの更新
        //=============================================

        let mut r_loss_val = 0.0;

        if self.lambda > 0.0 {

            //(A) 減税の予測後悔値R(s,a)の取得
            let r_values = self.regret_net.forward(&states_t)?;
            let current_r = r_values.gather(&actions_t.unsqueeze(1)?,1)?.squeeze(1)?;

            //(B)即時後悔の計算
            let current_neg_inf = masks_t.affine(-1.0,1.0)?.affine(-1e9f64,0.0)?;
            let masked_current_q = q_values.detach().add(&current_neg_inf)?;
            let max_current_q = masked_current_q.max_keepdim(1)?.squeeze(1)?;
            let immediate_regret = max_current_q.sub(&current_q.detach())?;

            //(C)未来の最小後悔値の計算
            let next_r_target = self.target_regret_net.forward(&next_states_t)?;

            let pos_inf_t = next_masks_t.affine(-1.0,1.0)?.affine(1e9f64,0.0)?;
            let masked_next_r = next_r_target.add(&pos_inf_t)?;
            let min_next_r = masked_next_r.min_keepdim(1)?.squeeze(1)?;
            let min_next_r = min_next_r.detach();

            //(D)Rのtargetの計算
            let target_r = min_next_r.broadcast_mul(&next_gammas_t)?.broadcast_mul(&not_done)?.broadcast_add(&immediate_regret)?;
            let r_loss = candle_nn::loss::huber(&current_r,&target_r,1.0)?;

            let mut reg_opt = self.reg_optimizer.borrow_mut();
            reg_opt.backward_step(&r_loss)?;

            r_loss_val = r_loss.to_scalar::<f32>()?;

        }
        

        Ok((q_loss.to_scalar::<f32>()?,r_loss_val))



    }

    pub fn update_target_network(&mut self,tau:f32) -> Result<()> {

        //効率が悪い書き方だが、並列化をするときに、デッドロックの原因にしないために、全ての更新を一時的にVecに溜めて、
        //ロックを明示的にdropしてから更新する仕様とした。安全性重視にしたが、更新頻度を上げる場合は、
        //ロックを取得するたびに更新する方式にした方が効率は良い。ただしその場合はシングルスレッドを推奨する。
        let all_vars = self.varmap.data().lock().map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let mut updates = Vec::new();

        for (name,var) in all_vars.iter(){
            if name.starts_with("policy."){
                let target_name = name.replace("policy.","target.");
                if let Some(target_var) = all_vars.get(&target_name){
                    let p_tensor = var.as_tensor();
                    let t_tensor = target_var.as_tensor();
                    let updated = if tau >= 1.0 {
                        p_tensor.copy()?
                    } else {
                        let t = tau as f64;
                        ((p_tensor * t)? + (t_tensor *(1.0 - t))?)?
                    };

                    updates.push((target_var.clone(),updated));
                    
                }
            }
        }

        drop(all_vars);
        for (var,tensor) in updates {
            var.set(&tensor)?;
        }

        Ok(())
    }

    pub fn update_target_regret_network(&mut self,tau:f32) -> Result<()> {

        let all_vars = self.varmap.data().lock().map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let mut updates = Vec::new();

        for (name,var) in all_vars.iter(){
            if name.starts_with("regret."){
                let target_name = name.replace("regret.","target_regret.");
                if let Some(target_var) = all_vars.get(&target_name){
                    let p_tensor = var.as_tensor();
                    let t_tensor = target_var.as_tensor();
                    let updated = if tau >= 1.0 {
                        p_tensor.copy()?
                    } else {
                        let t = tau as f64;
                        ((p_tensor * t)? + (t_tensor *(1.0 - t))?)?
                    };

                    updates.push((target_var.clone(),updated));
                    
                }
            }
        }

        drop(all_vars);
        for (var,tensor) in updates {
            var.set(&tensor)?;
        }

        Ok(())
    }

    pub fn save(&self,path: &str) -> Result<()>{
        self.varmap.save(path)?;

        Ok(())
    }

    pub fn load(&mut self,path:&str) -> Result<()>{
        self.varmap.load(path)?;
        self.update_target_network(1.0)?;
        self.update_target_regret_network(1.0)?;


        println!("Model loaded from {}",path);

        Ok(())
    }

    pub fn set_qnet_learning_rate(&mut self,lr:f64) {
        self.q_optimizer.borrow_mut().set_learning_rate(lr);
    }

    pub fn set_rnet_learning_rate(&mut self,lr:f64) {
        self.reg_optimizer.borrow_mut().set_learning_rate(lr);
    }

    pub fn copy_weights_to(&self,other:&mut DRNAgent) -> Result<()> {
        //ここでも同様に、全ての更新を一時的にVecに溜めて、ロックを明示的にdropしてから更新する仕様とした。
        let updates = {
            let src_vars = self.varmap.data().lock().map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let mut data = Vec::new();
            for (name,var) in src_vars.iter(){
                data.push((name.clone(),var.as_tensor().copy()?));
            }
            data
        };

        {
            let dst_vars = other.varmap.data().lock().map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            for (name,tensor) in updates {
                if let Some(dst_var) = dst_vars.get(&name) {
                    dst_var.set(&tensor)?;
                }
            }
        }
        other.update_target_network(1.0)?;
        other.update_target_regret_network(1.0)?;
        Ok(())
    }

    pub fn set_lambda(&mut self,new_lambda:f64) {
        self.lambda = new_lambda.min(1.0)
    }

    pub fn debug_print_values(&self,state:&RawState,player_id:&usize) -> Result<()> {
        let mut buf = self.processor.infer_buf.borrow_mut();
        buf.clear();

        self.processor.write_buf(&mut buf,state,*player_id,4); //player_id:0,num_players:4

        let state_t = Tensor::from_slice(&buf,(1,INPUT_STATE_DIM),&self.device)?;
        let mask_t = Tensor::from_slice(&state.legal_actions_mask,(1,53),&self.device)?;

        let q_values = self.policy_net.forward(&state_t,&mask_t)?;
        let reg_values = self.regret_net.forward(&state_t)?;

        let q_vec = q_values.squeeze(0)?.to_vec1::<f32>()?;
        let r_vec = reg_values.squeeze(0)?.to_vec1::<f32>()?;
        let mask_vec = mask_t.squeeze(0)?.to_vec1::<f32>()?;

        println!("\n  [ 🧠 DRN 脳内評価値一覧 (lambda: {:.2}) ]", self.lambda);
        println!("  ---------------------------------------------------------------------");
        println!("    行動     |  合法  |   Q値 (報酬期待)  |  R値 (後悔/詰み) |  統合価値 ((1-λ)Q - λR)");
        println!("  ---------------------------------------------------------------------");

        for i in 0..53 {
            let is_legal = mask_vec[i] > 0.0;
            let q_val = q_vec[i];
            let r_val = r_vec[i];
            let lambda_f = self.lambda as f32;
            let combined = (1.0 - lambda_f) * q_val - lambda_f * r_val;

            // パスかカードかで名前を変える
            let action_name = if i == PASS_ACTION as usize {
                "PASS      ".to_string()
            } else {
                format!("{:<10}", crate::card::Card(i as u8).to_string())
            };

            // 合法手、あるいは数値が動いている（関心がある）手だけを表示
            // (全部出すと53行になって見づらいので、合法手、またはR値が反応している手だけに絞る)
            if is_legal  {
                let legal_marker = if is_legal { "✅ YES" } else { "❌ NO " };
                
                // 実際に argmax で選ばれる最善手候補には強調マークをつける
                let highlight = if is_legal && i == self.calc_best_action_id(&q_vec, &r_vec, &mask_vec) {
                    "★ 最善手"
                } else {
                    ""
                };

                println!(
                    "    {} |  {} |    {:>14.4} |    {:>13.4} |    {:>16.4}  {}",
                    action_name, legal_marker, q_val, r_val, combined, highlight
                );
            }
        }
        println!("  ---------------------------------------------------------------------");
        Ok(())
    }

    // 最善手判定
    fn calc_best_action_id(&self, q: &[f32], r: &[f32], mask: &[f32]) -> usize {
        let mut max_val = f32::NEG_INFINITY;
        let mut best_idx = 52;
        let lambda_f = self.lambda as f32; // 追加
        for i in 0..53 {
            if mask[i] > 0.0 {
                // 修正後
                let val = (1.0 - lambda_f) * q[i] - lambda_f * r[i];
                if val > max_val {
                    max_val = val;
                    best_idx = i;
                }
            }
        }
        best_idx
    }

}

impl Agent for DRNAgent {
    fn select_action(&self,state:&RawState,player_id:&usize) -> AgentResult<u8> {
        self.infer_q(state,&player_id).map_err(|e| e.to_string())
    }
}
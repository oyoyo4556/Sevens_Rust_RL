
use crate::agent::per_agent::PERDQNAgent;
use crate::agent::drn_agent::DRNAgent;
use crate::agent::drn_eps_agent::DRNEPSAgent;
use crate::env::{RawState};
use rand::seq::IndexedRandom;
use std::cell::RefCell;
use candle_core::{Device,Result,Tensor};
use candle_nn::{AdamW,Optimizer,ParamsAdamW,VarBuilder,VarMap};
use std::collections::VecDeque;
use crate::qnet::DuelingQNet;
use crate::buffer::ReplayBuffer;
use crate::processor::Processor;
use crate::common::{Experience, TRAIN_AGENT_ID,INPUT_STATE_DIM};


pub struct RandomAgent {
    act_buf:RefCell<Vec<u8>>,
}
pub struct MainAgent {
    device:Device,
    pub varmap:VarMap,
    pub policy_net:DuelingQNet,
    pub target_net:DuelingQNet,
    optimizer:RefCell<AdamW>,
    pub buffer:ReplayBuffer,
    gamma:f32,
    pub epsilon:f64,
    epsilon_min:f64,
    epsilon_decay:f64,
    n_step_buffer:VecDeque<(RawState,u8,f32,RawState,bool)>,
    n_step: usize,
    processor:Processor,
}

pub enum Opponent {
    Random(RandomAgent),
    Main(MainAgent),
    PER(PERDQNAgent),
    DRN(DRNAgent),
    DRNEPS(DRNEPSAgent),
}

pub type AgentResult<T> = std::result::Result<T,String>;
pub trait Agent {
    fn select_action(&self,state:&RawState,player_id:&usize) -> AgentResult<u8>;
}

impl Agent for Opponent {
    fn select_action(&self,state:&RawState,player_id:&usize) -> AgentResult<u8> {
        match self{
            Opponent::Random(a) => a.select_action(state,player_id),
            Opponent::Main(a) => a.select_action(state,player_id),
            Opponent::PER(a) => a.select_action(state,player_id),
            Opponent::DRN(a) => a.select_action(state,player_id),
            Opponent::DRNEPS(a) => a.select_action(state,player_id),
        }
    }
}

impl RandomAgent {
    pub fn new() -> Self {
        Self{
            act_buf:RefCell::new(Vec::with_capacity(53)),
        }
    }
}

impl Agent for RandomAgent {
    fn select_action(&self,_state:&RawState,_player_id:&usize) -> AgentResult<u8> {
        self.infer_q(_state).map_err(|e| e.to_string())
    }
}

impl RandomAgent {
    pub fn infer_q(&self, _state:&RawState) -> Result<u8> {
        let mut buf = self.act_buf.borrow_mut();
        buf.clear();
        for (i,&val) in _state.legal_actions_mask.iter().enumerate() {
            if val == 1.0 {
                buf.push(i as u8);
            }
        }
        if buf.is_empty() {
            panic!{"No legal actions for active player."};
        }
        
        let mut rng = rand::rng();
        let choice = buf.choose(&mut rng).ok_or(candle_core::Error::msg("No legal actions".to_string()))?;
        Ok(*choice)
    }
}

impl MainAgent {
    pub fn new(capacity:usize,n_step:usize) -> Self{
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap,
        candle_core::DType::F32,&device);
        let policy_net = DuelingQNet::new(INPUT_STATE_DIM,512,53,vb.pp("policy")).unwrap();
        let target_net = DuelingQNet::new(INPUT_STATE_DIM,512,53,vb.pp("target")).unwrap();

        let my_params = ParamsAdamW{
            lr:5e-5,
            weight_decay:0.0,
            ..ParamsAdamW::default()
        };

        let optimizer = AdamW::new(varmap.all_vars(),my_params).unwrap();
        let processor = Processor::new(256);

        Self { 
            device,
            varmap,
            policy_net,
            target_net,
            optimizer:RefCell::new(optimizer),
            buffer: ReplayBuffer::new(capacity),
            gamma: 0.99,
            epsilon: 0.8,
            epsilon_min: 0.01,
            epsilon_decay: 0.99995,
            n_step_buffer:VecDeque::with_capacity(n_step),
            n_step, 
            processor,
        }
    }

    pub fn infer_q(&self,state:&RawState,player_id:&usize) -> Result<u8> {
        let mut rng = rand::rng();
        if rand::Rng::random_bool(&mut rng,self.epsilon) {
            let mut legals = Vec::new();
            for (i,&m) in state.legal_actions_mask.iter().enumerate() {
                if m == 1.0 {legals.push(i as u8)}
            }
            return Ok(
                *legals.choose(&mut rng).ok_or(candle_core::Error::Msg("No legal actions".to_string()))?
            );
        }

        let mut buf = self.processor.infer_buf.borrow_mut();
        buf.clear();

        self.processor.write_buf(&mut buf,state,*player_id,4); //player_id:0,num_players:4

        let required_elements = INPUT_STATE_DIM;

        assert_eq!(
            buf.len(), required_elements, "【致命的バグ防止】bufの要素数 ({}) が、要求されたTensorのサイズ ({}) と一致しません！", 
            buf.len(), 
            required_elements
        );

        let state_tensor = Tensor::from_slice(&buf,(1,INPUT_STATE_DIM),&self.device)?;
        let mask_tensor = Tensor::from_slice(&state.legal_actions_mask,(1,53),&self.device)?;
        let q_values = self.policy_net.forward(&state_tensor,&mask_tensor)?;

        let q_vec = q_values.flatten_all()?.to_vec1::<f32>()?;
        let mut max_q = f32::NEG_INFINITY;
        let mut best_action = None;

        for (i,(&q,&m)) in q_vec.iter().zip(state.legal_actions_mask.iter()).enumerate() {
            if m == 1.0 {
                if q > max_q || best_action.is_none() {
                    max_q = q;
                    best_action = Some(i as u8);
                }
            }
        } 

        best_action.ok_or(candle_core::Error::Msg("No legal actions".to_string()))
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

    pub fn update(&mut self,batch_size:usize) -> Result<f32> {
        if self.buffer.len() < batch_size{
            return Ok(0.0);
        }

        let batch = self.buffer.sample(batch_size) ;
        let (states_t,next_states_t,masks_t,next_masks_t,actions_t,rewards_t,dones_t,next_gammas_t) 
        = self.processor.batch_to_tensors(&batch, &self.device, TRAIN_AGENT_ID, 4)?;//player_id:0,num_players:4

        let actions_t =actions_t.to_dtype(candle_core::DType::U32)?; //gatherするため
        let q_values = self.policy_net.forward(&states_t,&masks_t)?;
        let current_q = q_values.gather(&actions_t.unsqueeze(1)?,1)?.squeeze(1)?;

        let next_q_policy = self.policy_net.forward(&next_states_t,&next_masks_t)?;

        let neg_inf_t = next_masks_t.affine(-1.0,1.0)?.affine(-1e9f64,0.0)?;    
        let masked_next_q = next_q_policy.add(&neg_inf_t)?;
        let next_actions = masked_next_q.argmax(1)?;

        let next_q_values = self.target_net.forward(&next_states_t,&next_masks_t)?;
        let max_next_q = next_q_values.gather(&next_actions.unsqueeze(1)?,1)?.squeeze(1)?;
        let max_next_q = max_next_q.detach();

        let not_done = (dones_t.ones_like()? - &dones_t)?;
        let target_q = max_next_q.broadcast_mul(&next_gammas_t)?.broadcast_mul(&not_done)?.broadcast_add(&rewards_t)?;
        let loss = candle_nn::loss::huber(&current_q,&target_q,1.0)?;

        let mut opt = self.optimizer.borrow_mut();

        opt.backward_step(&loss)?;

        //select_actionで減衰させると、自己対戦の都合上、select_actionが呼ばれるたびに減衰する可能性があるため。
        //一般的にはselect_actionで減衰させるが、探索は学習進度に合わせたいので、ここで減衰させる方式にした。
        //やっていることは1stepごとにupdateする場合は同じである。
        //update頻度をtrainerがupdate_freqで管理する場合は、ここで減衰させた方が管理しやすいと思われる。
        if self.epsilon > self.epsilon_min {
            self.epsilon *= self.epsilon_decay;
        }
        

        Ok(loss.to_scalar::<f32>()?)



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

    pub fn save(&self,path: &str) -> Result<()>{
        self.varmap.save(path)?;

        Ok(())
    }

    pub fn load(&mut self,path:&str) -> Result<()>{
        self.varmap.load(path)?;
        self.update_target_network(1.0)?;


        println!("Model loaded from {}",path);

        Ok(())
    }

    pub fn set_learning_rate(&mut self,lr:f64) {
        self.optimizer.borrow_mut().set_learning_rate(lr);
    }

    pub fn copy_weights_to(&self,other:&mut MainAgent) -> Result<()> {
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
        Ok(())
    }

}

impl Agent for MainAgent {
    fn select_action(&self,state:&RawState,player_id:&usize) -> AgentResult<u8> {
        self.infer_q(state,&player_id).map_err(|e| e.to_string())
    }
}
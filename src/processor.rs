use crate::env::{RawState};
use std::cell::RefCell;
use candle_core::{Device,Result,Tensor};
use crate::common::{Experience,INPUT_STATE_DIM};
//processorはstateをagentが理解できる形にするのでagentの一部と考えています。modelが変わればprocessorも変わる仕様とします
pub struct Processor{
    states_buf:RefCell<Vec<f32>>,
    next_states_buf:RefCell<Vec<f32>>,
    masks_buf:RefCell<Vec<f32>>,
    next_masks_buf:RefCell<Vec<f32>>,
    pub infer_buf:RefCell<Vec<f32>>,//agentのselect_actionで使用
}

impl Processor {

    pub fn new(max_batch_size:usize) -> Self {
        Self {
            states_buf:RefCell::new(Vec::with_capacity(max_batch_size*INPUT_STATE_DIM)),
            next_states_buf:RefCell::new(Vec::with_capacity(max_batch_size*INPUT_STATE_DIM)),
            masks_buf:RefCell::new(Vec::with_capacity(max_batch_size*53)),
            next_masks_buf:RefCell::new(Vec::with_capacity(max_batch_size*53)),
            infer_buf:RefCell::new(Vec::with_capacity(INPUT_STATE_DIM)),
        }
    }

    pub fn batch_to_tensors(&self,exps:&[&Experience],device:&Device,player_id:usize,num_players:usize) 
    -> Result<(Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor)> {

        let batch_size = exps.len();

        let mut s_buf = self.states_buf.borrow_mut();
        let mut ns_buf = self.next_states_buf.borrow_mut();
        let mut m_buf = self.masks_buf.borrow_mut();
        let mut nm_buf = self.next_masks_buf.borrow_mut();
        let mut actions_raw:Vec<u8> = Vec::with_capacity(batch_size);
        let mut rewards_raw:Vec<f32> = Vec::with_capacity(batch_size);
        let mut dones_raw:Vec<f32> = Vec::with_capacity(batch_size);
        let mut next_gammas_raw:Vec<f32> = Vec::with_capacity(batch_size);

        s_buf.clear();
        ns_buf.clear();
        m_buf.clear();
        nm_buf.clear();



        for exp in exps {
            self.write_buf(&mut s_buf,&exp.state,player_id,num_players);
            self.write_buf(&mut ns_buf,&exp.next_state,player_id,num_players);
            m_buf.extend_from_slice(&exp.state.legal_actions_mask);
            nm_buf.extend_from_slice(&exp.next_state.legal_actions_mask);
            actions_raw.push(exp.action as u8);
            rewards_raw.push(exp.reward);
            dones_raw.push(if exp.done{1.0f32} else{0.0f32});
            next_gammas_raw.push(exp.next_gamma);
        }

        let required_elements = batch_size * INPUT_STATE_DIM;
        let required_mask_elements = batch_size * 53; 

        assert_eq!(
            s_buf.len(), 
            required_elements, 
            "【致命的バグ防止】s_bufの要素数 ({}) が、要求されたTensorのサイズ ({}) と一致しません！", 
            s_buf.len(), 
            required_elements
        );

        assert_eq!(
            ns_buf.len(), 
            required_elements, 
            "【致命的バグ防止】ns_bufの要素数({})が、要求されたTensorのサイズ({})と一致しません!",
            ns_buf.len(),
            required_elements
        );

        assert_eq!(m_buf.len(), required_mask_elements, "【致命的バグ防止】m_bufの要素数が一致しません!");
        assert_eq!(nm_buf.len(), required_mask_elements, "【致命的バグ防止】nm_bufの要素数が一致しません!");

        let states = Tensor::from_slice(&s_buf,(batch_size,INPUT_STATE_DIM),device)?;
        let next_states = Tensor::from_slice(&ns_buf,(batch_size,INPUT_STATE_DIM),device)?;
        let masks = Tensor::from_slice(&m_buf,(batch_size,53),device)?;
        let next_masks = Tensor::from_slice(&nm_buf,(batch_size,53),device)?;
        let actions = Tensor::from_vec(actions_raw,batch_size,device)?;
        let rewards = Tensor::from_vec(rewards_raw,batch_size,device)?;
        let dones = Tensor::from_vec(dones_raw,batch_size,device)?;
        let next_gammas = Tensor::from_vec(next_gammas_raw,batch_size,device)?;
        Ok((states,next_states,masks,next_masks,actions,rewards,dones,next_gammas))

    }

    pub fn is_weight_to_tensors(&self,is_weight:Vec<f32>,device:&Device) -> Result<Tensor> {
        let batch_size = is_weight.len();
        let is_weight_t = Tensor::from_vec(is_weight,batch_size,&device)?;

        Ok(is_weight_t)
    }


    pub fn write_buf(&self,obs:&mut Vec<f32>,state:&RawState,player_id:usize,num_players:usize) {

        //場[52]
        for &f in &state.field {
            obs.push(if f {1.0} else {0.0});
        }

        //手札[52]
        let mut my_hand_flags = [0.0f32;52];
        for &card_id in &state.hands[player_id] {
            my_hand_flags[card_id as usize] = 1.0;
        }

        obs.extend_from_slice(&my_hand_flags);

        //ドボン者の手札[52]
        let mut virtual_flags = [0.0f32;52];
        for &card_id in &state.virtual_hand{
            virtual_flags[card_id as usize] = 1.0;
        }

        obs.extend_from_slice(&virtual_flags);

        //rank距離[52]
        let mut rd = [0.0f32;52];
        let mut suit_stops = [0.0f32;4];
        for suit in 0..4 {
            let mut rank_min = 6;
            let mut rank_max = 6;

            for rank in 0..13 {
                let card_id  = suit * 13 + rank;
                if state.field[card_id as usize] {
                    if rank < rank_min {
                        rank_min = rank;
                    } 
                    if  rank > rank_max {
                        rank_max = rank;
                    }
                }
            }

            let mut stops =0;
            for rank in 0..13 {
                let card_id =(suit * 13 + rank) as u8;

                if !state.field[card_id as usize] && !state.virtual_hand.contains(&card_id) && state.hands[player_id].contains(&card_id) {
                    if (rank < rank_min && rank == rank_min - 1) || (rank > rank_max && rank == rank_max +1 ) {
                        stops += 1;
                    }
                }
                let val = if rank < 6 && rank < rank_min {
                    (rank_min - rank) as f32
                } else if rank > 6 && rank > rank_max {
                    (rank - rank_max) as f32
                } else {0.0};
                rd[suit * 13 + rank] = val/6.0;
            }

            suit_stops[suit as usize] = (stops as f32) / 2.0;
        }

        obs.extend_from_slice(&rd);
        obs.extend_from_slice(&suit_stops);//suit_stops[4]


        for i in 0..num_players {
            let p_idx = (player_id + i) % num_players;
            obs.push((4.0 - state.pass_counts[p_idx] as f32) / 4.0);//パス残り回数[1*4=4]
            obs.push( if state.pass_counts[p_idx] >= 3 {1.0} else {0.0} );//パス使いきったフラグ[1*4=4]
            obs.push(state.hands[p_idx].len() as f32/13.0);//handの長さ[1*4=4]
            obs.push(state.action_log[p_idx]);//action_log[1*4=4]


        }

        let finished = state.finished_order.len() as f32 / 4.0;
        let eliminated = state.eliminated.len() as f32 /4.0;
        obs.push(finished);//finished_orderの人数[1]
        obs.push(eliminated);//eliminatedの人数[1]

        let remain_num =(num_players as f32 - state.finished_order.len() as f32 - state.eliminated.len() as f32) / num_players as f32;
        obs.push(remain_num);//残り人数[1]

        let field_on_count = state.field.iter().filter(|&&f| f).count() as f32 / 52.0;
        obs.push(field_on_count);//終盤判定[1]

        let legal_count = state.legal_actions_mask.iter().filter(|&&m| m == 1.0).count() as f32 / 13.0;
        obs.push(legal_count);//合法手の数[1]

        


    }
}

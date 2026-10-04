use crate::env::{RawState};
use std::cell::RefCell;
use candle_core::{Device,Result,Tensor};
use crate::common::{PPOExperience,INPUT_STATE_DIM};
//processorはstateをagentが理解できる形にするのでagentの一部と考えています。modelが変わればprocessorも変わる仕様とします


pub struct PPOProcessor {
    pub max_batch_size: usize,
    pub infer_buf: RefCell<Vec<f32>>,
    states_buf: RefCell<Vec<f32>>,
    next_states_buf: RefCell<Vec<f32>>,
    masks_buf: RefCell<Vec<f32>>,
    next_masks_buf: RefCell<Vec<f32>>,
    actions_buf: RefCell<Vec<u32>>,
    rewards_buf: RefCell<Vec<f32>>,
    dones_buf: RefCell<Vec<f32>>,
}

impl PPOProcessor {
    pub fn new(max_batch_size: usize) -> Self {
        Self {
            max_batch_size,
            infer_buf: RefCell::new(Vec::with_capacity(INPUT_STATE_DIM)),
            states_buf: RefCell::new(Vec::with_capacity(max_batch_size * INPUT_STATE_DIM)),
            next_states_buf: RefCell::new(Vec::with_capacity(max_batch_size * INPUT_STATE_DIM)),
            masks_buf: RefCell::new(Vec::with_capacity(max_batch_size * 53)),
            next_masks_buf: RefCell::new(Vec::with_capacity(max_batch_size * 53)),
            actions_buf: RefCell::new(Vec::with_capacity(max_batch_size)),
            rewards_buf: RefCell::new(Vec::with_capacity(max_batch_size)),
            dones_buf: RefCell::new(Vec::with_capacity(max_batch_size)),
        }
    }

    pub fn batch_to_tensors(
        &self,
        batch: &[&PPOExperience],
        device: &Device,
        train_player_id: usize,
        num_players: usize,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let batch_size = batch.len();

        // 内部バッファの借用
        let mut states = self.states_buf.borrow_mut();
        let mut next_states = self.next_states_buf.borrow_mut();
        let mut masks = self.masks_buf.borrow_mut();
        let mut next_masks = self.next_masks_buf.borrow_mut();
        let mut actions = self.actions_buf.borrow_mut();
        let mut rewards = self.rewards_buf.borrow_mut();
        let mut dones = self.dones_buf.borrow_mut();

        // バッファのリセット（アロケーションは維持されたまま要素数のみ0になる）
        states.clear();
        next_states.clear();
        masks.clear();
        next_masks.clear();
        actions.clear();
        rewards.clear();
        dones.clear();

        // データの詰め込み
        for exp in batch {
            self.write_buf(&mut states, &exp.state, train_player_id, num_players);
            masks.extend_from_slice(&exp.state.legal_actions_mask);

            self.write_buf(&mut next_states, &exp.next_state, train_player_id, num_players);
            next_masks.extend_from_slice(&exp.next_state.legal_actions_mask);

            actions.push(exp.action as u32);
            rewards.push(exp.reward);
            dones.push(if exp.done { 1.0f32 } else { 0.0f32 });
        }

        // Tensor化
        let states_t = Tensor::from_slice(&states, (batch_size, INPUT_STATE_DIM), device)?;
        let next_states_t = Tensor::from_slice(&next_states, (batch_size, INPUT_STATE_DIM), device)?;
        let masks_t = Tensor::from_slice(&masks, (batch_size, 53), device)?;
        let next_masks_t = Tensor::from_slice(&next_masks, (batch_size, 53), device)?;

        let actions_t = Tensor::from_slice(&actions, batch_size, device)?;
        let rewards_t = Tensor::from_slice(&rewards, batch_size, device)?;
        let dones_t = Tensor::from_slice(&dones, batch_size, device)?;

        Ok((
            states_t,
            next_states_t,
            masks_t,
            next_masks_t,
            actions_t,
            rewards_t,
            dones_t,
        ))
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
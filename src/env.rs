use crate::card::Card;
use crate::agent::{Opponent,Agent};
use rand::seq::SliceRandom;
use serde::{Serialize,Deserialize};

pub const PASS_ACTION:u8 =52;
pub const NUM_CARDS:usize = 52;

#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct RawState {
    pub hands:Vec<Vec<u8>>,
    pub field:Vec<bool>,
    pub virtual_hand:Vec<u8>,
    pub pass_counts:Vec<u8>,
    pub current_player:usize,
    pub finished_order:Vec<usize>,
    pub eliminated:Vec<usize>,
    pub action_log:Vec<f32>,
    pub legal_actions_mask:Vec<f32>,
}

pub struct SevensEnv {
    pub num_players:usize,
    pub agent_id:usize,
    pub opponent:Opponent,
    pub work_buf:Vec<u8>, //process_virtualで使用
    pub state:RawState,
}

impl SevensEnv {
    pub fn new(num_players:usize,agent_id:usize,opponent:Opponent) -> Self {
        Self {
            num_players,
            agent_id,
            opponent,
            work_buf:Vec::with_capacity(52),
            state:RawState {
                hands:vec![vec![];num_players],
                field:vec![false;NUM_CARDS],
                virtual_hand:vec![],
                pass_counts:vec![0;num_players],
                current_player:0,
                finished_order:Vec::new(),
                eliminated:Vec::new(),
                action_log:vec![0.0;num_players],
                legal_actions_mask:vec![0.0;53],
            },
        }
    }

    pub fn reset(&mut self) -> RawState{
        self.init_game();
        self.get_raw_state()
    }

    pub fn get_raw_state(&self) -> RawState{
        self.state.clone()
    }

    pub fn init_game(&mut self) {
        let mut deck: Vec<u8> = (0..52).collect();
        let mut rng = rand::rng();
        deck.shuffle(&mut rng);

        self.state.field = vec![false;NUM_CARDS];
        self.state.pass_counts = vec![0;self.num_players];
        self.state.finished_order.clear();
        self.state.eliminated.clear();
        self.state.virtual_hand.clear();
        self.state.action_log = vec![0.0;self.num_players];

        let chunk = 52 / self.num_players;
        for i in 0..self.num_players {
            self.state.hands[i] = deck[i*chunk..(i+1)*chunk].to_vec();
        }

        let diamond_seven = 6;
        for p in 0..self.num_players {
            if self.state.hands[p].contains(&diamond_seven){
                self.state.current_player = p;
            }

            self.state.hands[p].retain(|&card|{
                let is_seven = card % 13 == 6;
                if is_seven {
                    self.state.field[card as usize] = true;
                }
                !is_seven
            });
        }

        self.state.legal_actions_mask = self.get_legal_action_mask();

        while self.state.current_player != self.agent_id {
            self.opponent_turn().expect("Opponent failed during step");
        }

    }

    pub fn step(&mut self,action:u8) -> (RawState,f32,bool) {
        let player = self.state.current_player;
        let mut reward = 0.0;

        if self.state.legal_actions_mask[action as usize] == 0.0 {
            panic!("Illegal action:{}",action);
        }

        let event = self.apply_action(player,action);

        if event.dobon {reward -= 0.1;}
        
        let done = self.check_done();
        if done {
            let rewards = self.compute_reward_vector();
            reward += rewards[player];
            let mut final_state = self.get_raw_state();
            final_state.legal_actions_mask.fill(0.0);
            return (final_state,reward,true);
        }

        self.advance_player();

        while self.state.current_player != self.agent_id{

            if self.check_done() {
                let rewards = self.compute_reward_vector();
                reward += rewards[self.agent_id];
                let mut final_state = self.get_raw_state();
                final_state.legal_actions_mask.fill(0.0);
                return (final_state,reward,true);
            }
            reward += self.opponent_turn().expect("Opponent failed during step");
        }

        

        return (self.get_raw_state(),reward,false);

    }

    pub fn apply_action(&mut self,player:usize,action:u8) -> ActionEvent {
        let mut event = ActionEvent::default();

        if action != PASS_ACTION {
            if let Some(pos) = self.state.hands[player].iter().position(|&c| c == action) {
                self.state.hands[player].remove(pos);
                self.state.field[action as usize] = true;
                self.state.action_log[player] = (action as f32 + 1.0)/53.0;
            }
        } else {
            self.state.pass_counts[player] += 1;
            self.state.action_log[player] = (action as f32 + 1.0) /53.0;

            if self.state.pass_counts[player] >= 4 {
                if player == self.agent_id {
                event.dobon = true;
                } else {
                    event.make_opp_dobon = true;
                }
                self.dobon(player);
            }
        }

        self.process_virtual();

        if self.state.hands[player].is_empty() && 
        !self.state.finished_order.contains(&player) &&
        !self.state.eliminated.contains(&player) {
            self.state.finished_order.push(player);
        }

        event
    }

    pub fn dobon(&mut self,player:usize) {
        self.state.virtual_hand.append(&mut self.state.hands[player]);
        if !self.state.eliminated.contains(&player) {
            self.state.eliminated.push(player)
        }
        self.state.action_log[player] = 0.0;
    }

    pub fn advance_player(&mut self) {
        if self.check_done(){return;}
        loop {
            self.state.current_player = (self.state.current_player + 1 ) % self.num_players;
            if !self.state.finished_order.contains(&self.state.current_player) && 
            !self.state.eliminated.contains(&self.state.current_player) {break;}
        }
        self.state.legal_actions_mask = self.get_legal_action_mask();
    } 

    pub fn can_play(&self,card:Card) -> bool {
        let rank = card.rank();
        if rank == 6 {return true;}
        let target = if rank < 6 {
            card.neighbor(1)
        } else {
            card.neighbor(-1)
        };

        target.map_or(false,|t| self.state.field[t.0 as usize])
    }

    pub fn get_legal_action_mask(&self) -> Vec<f32> {
        let mut mask = vec![0.0;53];
        let p = self.state.current_player;

        if self.state.finished_order.contains(&p) || 
        self.state.eliminated.contains(&p) {
            return mask;
        }

        let mut playable_count = 0 ;
        for &card_id in &self.state.hands[p] {
            if self.can_play(Card(card_id)) {
                mask[card_id as usize] = 1.0;
                playable_count += 1;
            }
        }

        if self.state.pass_counts[p] < 4 {
            if playable_count >= 1 && self.state.pass_counts[p] == 3 {
                mask[PASS_ACTION as usize] = 0.0;
            } else {
                mask[PASS_ACTION as usize] = 1.0;
            }
        }

        mask

    }

    pub fn opponent_turn(&mut self) -> Result<f32,String> {
        let mut reward = 0.0;
        let player = self.state.current_player;

        let action = self.opponent.select_action(&self.state,&player)?;

        let event = self.apply_action(player, action);
        if event.make_opp_dobon {reward += 0.05;}
        self.advance_player();

        

        Ok(reward)
        
    }

    pub fn process_virtual(&mut self) {
        let mut changed = true;
        while changed {
            changed = false;
            self.work_buf.clear();
            for &card_id in &self.state.virtual_hand {
                if self.can_play(Card(card_id)) {
                    self.work_buf.push(card_id)
                }
            }

            if !self.work_buf.is_empty() {
                changed =true;
                for &card_id in &self.work_buf {
                    self.state.field[card_id as usize] = true;
                }
                let work = &self.work_buf;

                self.state.virtual_hand.retain(|c| ! work.contains(c))
            }
            
        }
    }

    pub fn check_done(&self) -> bool {
        self.state.finished_order.len() + self.state.eliminated.len() == self.num_players
    }

    fn compute_reward_vector(&self) -> Vec<f32> {
        let mut  rewards = vec![0.0;self.num_players];

        if self.state.finished_order.is_empty() {
            return rewards;
        }
        
        let values = [1.0,0.3,-0.3,-1.0];
        let mut full_order = self.state.finished_order.clone();
        let mut el = self.state.eliminated.clone();

        el.reverse();
        full_order.extend(el);

        for (rank,&p_idx) in full_order.iter().enumerate() {
            if rank < values.len() {
                rewards[p_idx] = values[rank];
            }
        }
        rewards
    }





}

#[derive(Default)]
pub struct ActionEvent {
    dobon:bool,
    make_opp_dobon:bool,
}
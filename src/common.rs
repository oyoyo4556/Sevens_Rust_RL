use crate::env::RawState;

#[derive(Clone)]
pub struct Experience {
    pub state:RawState,
    pub action:u8,
    pub reward:f32,
    pub next_state:RawState,
    pub done:bool,
    pub next_gamma: f32,
}

pub const TRAIN_AGENT_ID:usize = 0;
pub const INPUT_STATE_DIM:usize = 233;
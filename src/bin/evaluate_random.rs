use std::fs;
use std::path::Path;
use sevens::agent::per_agent::PERDQNAgent;
use sevens::env::{SevensEnv};
use sevens::agent::agent::{RandomAgent,Opponent};
use sevens::trainer::Trainer;

fn main(){
    let save_dir ="checkpoints".to_string();
    if !Path::new(&save_dir).exists() {
        fs::create_dir_all(&save_dir).expect("Failed to create save directory.");
        println!("Created directory: {}",save_dir);
    }

    let eta_max = 1e-4;
    let eta_min = 1e-6;
    let t_0 = 10000;
    let t_mult = 2;

    let batch_size = 64;
    let tau = 1.0;
    let save_interval = 3000;
    let num_episodes = 10000;
    let agent_name = "perdqn_v1.0.0".to_string();

    let opponent = Opponent::Random(RandomAgent::new());
    let mut env = SevensEnv::new(4,0,opponent);
    let mut agent = PERDQNAgent::new(100_000,3);
    let mut trainer = Trainer::new(
        eta_max,
        eta_min,
        t_0,
        t_mult,
        batch_size,
        tau,
        save_dir,
        save_interval,
        agent_name,
    );

    agent.load("checkpoints/perdqn_v1.0.0_ep10000.safetensors").expect("Failed to load model.check the path!");
    agent.epsilon = 0.0;

    trainer.per_vs_random(&mut agent,&mut env,num_episodes).unwrap();
}


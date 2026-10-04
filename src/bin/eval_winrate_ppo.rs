use std::fs;
use std::path::Path;
use sevens::agent::ppo_agent::PPOAgent;
use sevens::env::SevensEnv;
use sevens::agent::agent::{MainAgent,RandomAgent,Opponent};
use sevens::ppo::ppo_trainer::PPOTrainer;

fn main(){
    let save_dir ="checkpoints".to_string();
    if !Path::new(&save_dir).exists() {
        fs::create_dir_all(&save_dir).expect("Failed to create save directory.");
        println!("Created directory: {}",save_dir);
    }

    let eta_max = 1e-4;
    let eta_min = 1e-5;
    let t_0 = 10000;
    let t_mult = 2;

    let batch_size = 64;
    let opponent_update_interval = 10000;
    let num_envs = 1;
    let steps_per_env = 50;
    let save_interval = 3000;
    let num_episodes = 10000;
    let agent_name = "ppo_v1.0.0".to_string();

    let mut agent = PPOAgent::new();
    agent.load("checkpoints/ppo_v1.0.0_it12000.safetensors").expect("Failed to load model.check the path!");
    
    //let opponent = Opponent::Random(RandomAgent::new());
    let mut opponent = MainAgent::new(100,1);
    opponent.load("checkpoints/dqn_v1.4.1_ep200000.safetensors").expect("failed copy_weight to opponent!");
    opponent.epsilon = 0.0;
    let opponent = Opponent::Main(opponent);
    let mut env = SevensEnv::new(4,0,opponent);
    if let Opponent::Main(ref mut opp_agent) = env.opponent  {
        println!("Opponet is DQNAgent with epsilon: {}",opp_agent.epsilon);
    } else {
        println!("Opponent is RandomAgent");
    } 

    let mut trainer = PPOTrainer::new(
        agent_name,
        num_envs, 
        steps_per_env, 
        batch_size,
        opponent_update_interval,
        eta_max,
        eta_min,
        t_0,
        t_mult,
        save_dir,
        save_interval,
    );

    trainer.ppo_vs(&mut agent,&mut env,num_episodes).unwrap();
}


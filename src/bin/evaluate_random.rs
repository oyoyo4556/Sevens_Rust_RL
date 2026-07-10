use std::fs;
use std::path::Path;
use sevens::env::{SevensEnv};
use sevens::agent::drn_agent::DRNAgent;
use sevens::agent::agent::{RandomAgent,Opponent};
use sevens::trainer::DRNTrainer;

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
    let tau = 1.0;
    let save_interval = 3000;
    let num_episodes = 10000;
    let agent_name = "drn_v1.1.2".to_string();

    let mut agent = DRNAgent::new(100_000,3);
    agent.load("checkpoints/drn_v1.1.2_ep100000.safetensors").expect("Failed to load model.check the path!");
    agent.set_lambda(1.0);

    let opponent = Opponent::Random(RandomAgent::new());
    //let mut opponent = DRNAgent::new(100,1);
    //agent.copy_weights_to(&mut opponent).expect("failed copy_weight to opponent!");
    //opponent.set_lambda(0.0);
    //let opponent = Opponent::DRN(opponent);
    let mut env = SevensEnv::new(4,0,opponent);
    println!("Agent lambda :{}",agent.lambda);
    if let Opponent::DRN(ref mut opp_agent) = env.opponent  {
        println!("Opponet is DRNAgent with lambda :{}",opp_agent.lambda);
    } else {
        println!("Opponent is RandomAgent");
    }
    let mut trainer = DRNTrainer::new(
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

    trainer.drn_vs_random(&mut agent,&mut env,num_episodes).unwrap();
}


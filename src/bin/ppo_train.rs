use std::fs;
use std::path::Path;
use sevens::ppo::ppo_trainer::PPOTrainer;

fn main(){
    let save_dir ="checkpoints".to_string();
    if !Path::new(&save_dir).exists() {
        fs::create_dir_all(&save_dir).expect("Failed to create save directory.");
        println!("Created directory: {}",save_dir);
    }

    let cpus = std::thread::available_parallelism().unwrap().get();
    println!("Available parallelism: {}", cpus);
    println!("But only using one core.");

    let num_envs = 1;
    println!("use num_envs: {}",num_envs);
    let steps_per_env = 1024;
    let opponent_update_interval = 100;

    let eta_max = 1e-4;
    let eta_min = 1e-5;
    let t_0 = 1000;
    let t_mult = 1;

    let batch_size = 512;
    let save_interval = 2000;
    let num_iteration = 50_000;
    let agent_name = "ppo_v1.0.0".to_string();

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


    println!("========================================================");
    println!("Starting training for {} iteration",num_iteration);
    println!("Save_Interval:every {} iteration",save_interval);
    println!("Agent Name:{}",&trainer.agent_name);
    println!("=========================================================");

    trainer.train(num_iteration).unwrap();

    let final_model_path = format!("{}/final_model.safetensors",trainer.save_dir);
    trainer.main_agent.save(&final_model_path).unwrap();
    println!("========================================================");
    println!("Training completed. Final model Savedto :{}",final_model_path);
    println!("========================================================");
}
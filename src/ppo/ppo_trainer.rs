use crate::agent::ppo_agent::PPOAgent;
use crate::env::SevensEnv;
use crate::ppo::rolloutbuffer::RolloutBuffer;
use crate::agent::agent::{Agent,RandomAgent,Opponent};
use crate::common::{PPOExperience,TRAIN_AGENT_ID};
use crate::lr_scheduler::CosineAnnealingWarmRestarts;

use rayon::prelude::*;
use candle_core::{Result};
use rand::seq::IndexedRandom;


pub struct PPOTrainer {
    pub main_agent: PPOAgent,
    pub worker_agents: Vec<PPOAgent>,
    pub envs: Vec<SevensEnv>,
    pub buffer: RolloutBuffer,
    pub scheduler: CosineAnnealingWarmRestarts,
    pub steps_per_env: usize,
    pub num_envs: usize,
    pub gamma: f32,
    pub gae_lambda: f32,
    pub ppo_epochs: usize,
    pub batch_size: usize,
    pub opponent_update_interval: usize, 
    pub opponent_pool: Vec<PPOAgent>, 
    pub max_pool_size: usize,
    pub agent_name:String,
    pub save_dir:String,
    pub save_interval:usize,
}

impl PPOTrainer {
    pub fn new(agent_name:String,num_envs: usize, steps_per_env: usize, batch_size: usize,opponent_update_interval: usize,eta_max:f64,eta_min:f64,t_0:usize,t_mult:usize,save_dir:String,save_interval:usize) -> Self {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(num_envs)
            .build_global().is_ok(); 
        let capacity = num_envs * steps_per_env;
        let main_agent = PPOAgent::new();

        let worker_agents = (0..num_envs).map(|_| PPOAgent::new()).collect();
        let envs = (0..num_envs)
            .map(|_| SevensEnv::new(4, 0, Opponent::Random(RandomAgent::new()))) // 既存の対戦相手設定
            .collect();

        Self {
            
            main_agent,
            worker_agents,
            envs,
            buffer: RolloutBuffer::new(capacity),
            scheduler: CosineAnnealingWarmRestarts::new(eta_max, eta_min, t_0, t_mult),
            steps_per_env,
            num_envs,
            gamma: 0.99,
            gae_lambda: 0.95,
            ppo_epochs: 4,
            batch_size,
            opponent_update_interval,
            opponent_pool: Vec::new(),
            max_pool_size:5,
            agent_name,
            save_dir,
            save_interval,
        }
    }

    /// 1 イテレーションの学習ループ
    pub fn train_iteration(&mut self) -> Result<(f32, f32,f32,f32,f32)> {

        self.scheduler.step();
        let current_lr = self.scheduler.get_lr();
        self.main_agent.set_learning_rate(current_lr);

        for worker in self.worker_agents.iter_mut() {
        self.main_agent.copy_weights_to(worker)?;
        }

        let steps_per_env = self.steps_per_env;
        let rollouts: Vec<Vec<PPOExperience>> = self.envs
            .par_iter_mut()
            .zip(self.worker_agents.par_iter_mut())
            .map(|(env, agent)| {
                let mut local_exps = Vec::with_capacity(steps_per_env);

                let mut state = env.reset();
                for _ in 0..steps_per_env {
                    let player_id = state.current_player;

                    // 行動選択 (action, log_prob, value)
                    let (action, log_prob, value) = agent
                        .select_action_with_info(&state, &player_id, false)
                        .unwrap();

                    let (next_state, reward, done) = env.step(action);

                    local_exps.push(PPOExperience {
                        state,
                        action,
                        log_prob,
                        value,
                        reward,
                        done,
                        next_state: next_state.clone(),
                    });

                    if done {
                        state = env.reset();
                    } else {
                        state = next_state;
                    }
                }
                local_exps
            })
            .collect();

        // 3. RolloutBuffer にデータを流し込んで GAE 計算
        self.buffer.clear();
        for rollout in rollouts {
            self.buffer.push_batch(rollout);
        }
        self.buffer.compute_gae(self.gamma, self.gae_lambda);

        // 4. PPO Epoch (学習)
        let mut total_loss = 0.0;
        let mut total_policy_loss = 0.0;
        let mut total_value_loss = 0.0;
        let mut total_entropy = 0.0;
        let mut update_count = 0;

        for _epoch in 0..self.ppo_epochs {
            // Buffer から参照渡しの PPOBatchItem 群を取得
            let batches = self.buffer.get_batches(self.batch_size);

            for batch in batches {
                // Trainerは batch をそのまま PPOAgent に渡す
                let (loss,policy_loss,value_loss,entropy) = self.main_agent.update_batch(&batch)?;

                total_loss += loss;
                total_policy_loss += policy_loss;
                total_value_loss += value_loss;
                total_entropy += entropy;
                update_count += 1;
            }
        }

        let (avg_loss,avg_policy_loss,avg_value_loss,avg_entropy) = if update_count > 0 {
            let update_countf32 = update_count as f32;
            (
                total_loss/update_countf32,
                total_policy_loss/update_countf32,
                total_value_loss/update_countf32 ,
                total_entropy/update_countf32,
            )
        } else {
            (0.0,0.0,0.0,0.0)
        };
        let avg_reward =
            self.buffer.exps.iter().map(|e| e.reward).sum::<f32>() / self.buffer.exps.len() as f32;
        

        Ok((avg_loss,avg_policy_loss,avg_value_loss,avg_entropy, avg_reward))
    }

    pub fn train(&mut self,num_iteration:usize) -> Result<()> {
        let mut reward_history = Vec::new();
        let mut total_loss_sum = 0.0;
        let mut total_policy_loss_sum = 0.0;
        let mut total_value_loss_sum = 0.0;
        let mut total_entropy_sum = 0.0;
        for iter in 1..num_iteration+1 { 
            let (avg_loss,avg_policy_loss,avg_value_loss,avg_entropy,avg_reward) = self.train_iteration()?;

            reward_history.push(avg_reward);
            total_loss_sum += avg_loss;
            total_policy_loss_sum += avg_policy_loss;
            total_value_loss_sum += avg_value_loss;
            total_entropy_sum += avg_entropy;

            if iter % 10 == 0 {
                let avg_reward:f32 = reward_history.iter().rev().take(10).sum::<f32>()/10.0;
                let avg_loss:f32 = total_loss_sum/10.0;
                let avg_policy_loss:f32 = total_policy_loss_sum/10.0;
                let avg_value_loss:f32 = total_value_loss_sum/10.0;
                let avg_entropy:f32 = total_entropy_sum/10.0;
                let current_lr = self.scheduler.get_lr();
                println!("Iteration :{:>5},Ave_Reward:{:>7.2e}, Ave_Total_Loss:{:>8.4}, Ave_Po_Loss:{:>8.4}, Ave_Va_Loss:{:>8.4}, Ave_Ent:{:>8.4}, lr:{:>8.2e}"
                ,iter,avg_reward,avg_loss,avg_policy_loss,avg_value_loss,avg_entropy,current_lr);
                total_loss_sum =0.0;
                total_policy_loss_sum =0.0;
                total_value_loss_sum =0.0;
                total_entropy_sum =0.0;


                if reward_history.len() > 1000 {
                    reward_history.drain(0..reward_history.len()-500);
                }

            }

            if iter % self.opponent_update_interval == 0 {
                self.snapshot_to_opponent_pool()?;
                self.refresh_env_opponents()?;
            }

            if iter % self.save_interval == 0 {
                let path = format!("{}/{}_it{}.safetensors",self.save_dir,self.agent_name,iter);
                self.main_agent.save(&path)?;
                println!("Model saved on iteration {}",iter);
            }
        }
        Ok(())
    }

    fn snapshot_to_opponent_pool(&mut self) -> Result<()> {
        let mut snapshot = PPOAgent::new();
        self.main_agent.copy_weights_to(&mut snapshot)?;

        self.opponent_pool.push(snapshot);

        // プールサイズが上限を超えたら古いものを捨てる
        if self.opponent_pool.len() > self.max_pool_size {
            self.opponent_pool.remove(0);
        }

        println!(
            "[Self-Play] Added snapshot to pool. Current pool size: {}",
            self.opponent_pool.len()
        );
        Ok(())
    }

    /// 各環境の対戦相手（Opponent）をプール内からランダムに再割り当て
    fn refresh_env_opponents(&mut self) -> Result<()> {
        if self.opponent_pool.is_empty() {
            return Ok(());
        }

        let mut rng = rand::rng();

        for env in self.envs.iter_mut() {
            // プールの中からランダムに「過去の自分」を選択
            if let Some(past_agent_src) = self.opponent_pool.choose(&mut rng) {
                let mut opp_agent = PPOAgent::new();
                past_agent_src.copy_weights_to(&mut opp_agent)?;

                // 対戦相手としてセット
                env.opponent = Opponent::PPO(opp_agent);
            }
        }
        println!("Update opponent!");

        Ok(())
    }

    pub fn winrate_check(&mut self,agent:&mut PPOAgent,num_episodes:usize) -> Result<()> {
        let opponent = Opponent::Random(RandomAgent::new());
        let mut env = SevensEnv::new(4,0,opponent);
        let mut agent_ranks = Vec::new();
        let mut rank_counts = vec![0;4];

        for _episode in 1..=num_episodes {
            let mut state = env.reset();
            let mut done = false;
            while !done {
                let action = agent.select_action(&state,&TRAIN_AGENT_ID).map_err(candle_core::Error::msg)?;
                let (next_state,_,is_done) = env.step(action);
                state = next_state;
                done = is_done;
            }
            
            let mut final_ranks = env.state.finished_order.clone();
            let mut eliminated = env.state.eliminated.clone();
            eliminated.reverse();
            final_ranks.extend(eliminated);
            let agent_rank = final_ranks.iter().position(|&p| p == 0).expect("Failed to find agent's rank");
            agent_ranks.push(agent_rank);
            rank_counts[agent_rank] += 1;
        }

        let r1_rate = rank_counts[0] as f32/num_episodes as f32 *100.0;
        let r2_rate = rank_counts[1] as f32/num_episodes as f32 *100.0;
        let r3_rate = rank_counts[2] as f32/num_episodes as f32 *100.0;
        let r4_rate = rank_counts[3] as f32/num_episodes as f32 *100.0;
        let ave_rank = agent_ranks.iter().sum::<usize>() as f32 / agent_ranks.len() as f32 + 1.0;
        println!("========================================================");
        println!("Result {} Games vs RandomAgent",num_episodes);
        println!("Mainagent Win Rate (1st place): {:.2}%",rank_counts[0] as f32/num_episodes as f32 *100.0);
        println!("Rank Rate: 1st:{:.2}% | 2nd:{:.2}% | 3rd:{:.2}% | 4th:{:.2}%",r1_rate,r2_rate,r3_rate,r4_rate);
        println!("Ave_Rank : {:.4}",ave_rank);
        println!("========================================================");
        Ok(())
    }

    pub fn ppo_vs(&mut self,agent:&mut PPOAgent,env:&mut SevensEnv,num_episodes:usize) -> Result<()> {
        let mut agent_ranks = Vec::new();
        let mut rank_counts = vec![0;4];
        println!("=============================================================");
        println!("Starting evaluation PPO vs Opponent for {} episodes",num_episodes);
        println!("=============================================================");
        for episode in 1..num_episodes {
            let mut state = env.reset();
            let mut done = false;
            while !done {
                
                let action = agent.select_action_with_info(&state,&TRAIN_AGENT_ID, false)
                                  .map(|(action, _, _)| action)
                                  .map_err(candle_core::Error::msg)?;

                let (next_state,_,is_done) = env.step(action);
                state = next_state;
                done = is_done;
            }
            let mut final_ranks = env.state.finished_order.clone();
            let mut eliminated = env.state.eliminated.clone();
            eliminated.reverse();
            final_ranks.extend(eliminated);
            let agent_rank = final_ranks.iter().position(|&p| p == 0).expect("Failed to find agent's rank");
            agent_ranks.push(agent_rank);
            rank_counts[agent_rank] += 1;

            if episode % 1000 == 0 {
                let r1_rate = rank_counts[0] as f32/episode as f32 *100.0;
                let r2_rate = rank_counts[1] as f32/episode as f32 *100.0;
                let r3_rate = rank_counts[2] as f32/episode as f32 *100.0;
                let r4_rate = rank_counts[3] as f32/episode as f32 *100.0;
                let ave_rank = agent_ranks.iter().sum::<usize>() as f32 / agent_ranks.len() as f32 + 1.0;
                println!("Games:{:>5} | 1st:{:>5.2}% | 2nd:{:>5.2}% | 3rd:{:>5.2}% | 4th:{:>5.2}% | Ave_Rank:{:>5.2}",
                episode,r1_rate,r2_rate,r3_rate,r4_rate,ave_rank);
            }


            

        }
        let r1_rate = rank_counts[0] as f32/num_episodes as f32 *100.0;
            let r2_rate = rank_counts[1] as f32/num_episodes as f32 *100.0;
            let r3_rate = rank_counts[2] as f32/num_episodes as f32 *100.0;
            let r4_rate = rank_counts[3] as f32/num_episodes as f32 *100.0;
            let ave_rank = agent_ranks.iter().sum::<usize>() as f32 / agent_ranks.len() as f32 + 1.0;
            println!("========================================================");
            println!("Final Result {} Games vs Opponent",num_episodes);
            println!("Mainagent Win Rate (1st place): {:.2}%",rank_counts[0] as f32/num_episodes as f32 *100.0);
            println!("Rank Rate: 1st:{:.2}% | 2nd:{:.2}% | 3rd:{:.2}% | 4th:{:.2}%",r1_rate,r2_rate,r3_rate,r4_rate);
            println!("Ave_Rank : {:.4}",ave_rank);
            println!("========================================================");
        Ok(())
    }
}
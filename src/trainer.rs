use crate::agent::per_agent::PERDQNAgent;
use crate::agent::drn_agent::DRNAgent;
use crate::agent::drn_eps_agent::DRNEPSAgent;
use crate::lr_scheduler::CosineAnnealingWarmRestarts;
use crate::agent::agent::{Agent,MainAgent,Opponent,RandomAgent};
use crate::env::SevensEnv;
use candle_core::Result;
use crate::common::TRAIN_AGENT_ID;

pub struct Trainer {
    scheduler:CosineAnnealingWarmRestarts,
    batch_size:usize,
    tau:f32,
    pub save_dir:String,
    pub save_interval:usize,
    pub agent_name:String,
}

impl Trainer {
    pub fn new(eta_max:f64,eta_min:f64,t_0:usize,t_mult:usize,batch_size:usize,tau:f32,save_dir:String,save_interval:usize,agent_name:String) -> Self {
        Self { 
            scheduler: CosineAnnealingWarmRestarts::new(eta_max, eta_min, t_0, t_mult),
            batch_size,
            tau,
            save_dir,
            save_interval,
            agent_name,
        }
    }

    pub fn train_step(&mut self,agent:&mut MainAgent) -> Result<f32> {
        self.scheduler.step();
        let current_lr = self.scheduler.get_lr();
        agent.set_learning_rate(current_lr);

        if self.scheduler.is_at_vally() {
            let cycle =self.scheduler.get_cycle_index();
            agent.save(&format!("{}/{}_cycle{}.safetensors",self.save_dir,self.agent_name,cycle))?;
            println!("Model saved at cycle {}",cycle);
        }

        let loss = agent.update(self.batch_size)?;
        agent.update_target_network(self.tau)?;
        Ok(loss)
    }

    pub fn run_episode(&mut self,agent:&mut MainAgent,env:&mut SevensEnv) -> Result<(f32,f32)> {
        let mut state = env.reset();
        let mut total_loss = 0.0;
        let mut steps = 0;
        let mut done = false;
        let mut total_reward = 0.0;

        while !done {
            let action = agent.select_action(&state,&TRAIN_AGENT_ID).map_err(candle_core::Error::msg)?;
            let (next_state,reward,is_done) = env.step(action);
            
            agent.add_experience(
                state.clone(),
                action,
                reward,
                next_state.clone(),
                is_done,
            );

            state = next_state;
            
            done = is_done;
            total_reward += reward;
        }

        if agent.buffer.len() >= self.batch_size {
                let loss = self.train_step(agent)?;
                total_loss += loss;
                steps += 1;
        }

        Ok((if steps >0 {total_loss/steps as f32} else {0.0},total_reward))
    }

    pub fn train(&mut self,agent:&mut MainAgent,env:&mut SevensEnv,num_episodes:usize) -> Result<()> {

        let mut reward_history = Vec::new();
        let mut total_loss_sum = 0.0; 

        for episode in 1..num_episodes {
            let (loss,episode_reward) = self.run_episode(agent,env)?;
            reward_history.push(episode_reward);
            total_loss_sum += loss;
            
            if episode % 100 == 0 {
                let avg_reward:f32 = reward_history.iter().rev().take(100).sum::<f32>()/100.0;
                let avg_loss:f32 = total_loss_sum/100.0;
                let current_lr = self.scheduler.get_lr();
                println!("Episode :{:>5},Ave_Reward:{:>7.2}, Ave_Loss:{:>8.4}, lr:{:>8.2e},Epsilon:{:>4.2}"
                ,episode,avg_reward,avg_loss,current_lr,agent.epsilon);
                total_loss_sum =0.0;
                if reward_history.len() > 1000 {
                    reward_history.drain(0..reward_history.len()-500);
                }

            }
            if episode %self.save_interval == 0 {
                let path = format!("{}/{}_ep{}.safetensors",self.save_dir,self.agent_name,episode);
                agent.save(&path)?;
                println!("Model saved on episode {}",episode);
            }

            //対戦相手の更新
            if episode % 3000 == 0 {
                if let Opponent::Main(ref mut opp_agent) = env.opponent {
                    agent.copy_weights_to(opp_agent)?;
                    opp_agent.epsilon = 0.0;
                    println!("Opponent updated at episode {}",episode);
                } else {
                    let mut new_opp_agent = MainAgent::new(100,1);
                    agent.copy_weights_to(&mut new_opp_agent)?;
                    new_opp_agent.epsilon = 0.0;
                    env.opponent = Opponent::Main(new_opp_agent);
                    println!("Opponent switched to new agent");
                }
            }

            if episode % 20000 == 0 {
                if let Err(e) = self.winrate_check(agent, 10000) {
                    eprintln!("Error during winrate check: {}", e);
                } else {
                    println!("Winrate check completed successfully at episode {}", episode);
                }
            }
        }
        
        Ok(())

    }

    pub fn train_random(&mut self,agent:&mut MainAgent,env:&mut SevensEnv,num_episodes:usize) -> Result<()> {

        let mut reward_history = Vec::new();
        let mut total_loss_sum = 0.0; 

        for episode in 1..num_episodes {
            let (loss,episode_reward) = self.run_episode(agent,env)?;
            reward_history.push(episode_reward);
            total_loss_sum += loss;
            
            if episode % 100 == 0 {
                let avg_reward:f32 = reward_history.iter().rev().take(100).sum::<f32>()/100.0;
                let avg_loss:f32 = total_loss_sum/100.0;
                let current_lr = self.scheduler.get_lr();
                println!("Episode :{:>5},Ave_Reward:{:>7.2}, Ave_Loss:{:>8.4}, lr:{:>8.2e},Epsilon:{:>4.2}"
                ,episode,avg_reward,avg_loss,current_lr,agent.epsilon);
                total_loss_sum =0.0;
                if reward_history.len() > 1000 {
                    reward_history.drain(0..reward_history.len()-500);
                }

            }
            if episode %self.save_interval == 0 {
                let path = format!("{}/{}_ep{}.safetensors",self.save_dir,self.agent_name,episode);
                agent.save(&path)?;
                println!("Model saved on episode {}",episode);
            }

            
        }
        Ok(())

    }

    pub fn vs_random(&mut self,agent:&mut MainAgent,env:&mut SevensEnv,num_episodes:usize) -> Result<()> {
        let mut agent_ranks = Vec::new();
        let mut rank_counts = vec![0;4];
        env.opponent = Opponent::Random(RandomAgent::new());
        println!("=============================================================");
        println!("Starting evaluation vs RandomAgent for {} episodes",num_episodes);
        println!("=============================================================");
        for episode in 1..num_episodes {
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
            println!("Final Result {} Games vs RandomAgent",num_episodes);
            println!("Mainagent Win Rate (1st place): {:.2}%",rank_counts[0] as f32/num_episodes as f32 *100.0);
            println!("Rank Rate: 1st:{:.2}% | 2nd:{:.2}% | 3rd:{:.2}% | 4th:{:.2}%",r1_rate,r2_rate,r3_rate,r4_rate);
            println!("Ave_Rank : {:.4}",ave_rank);
            println!("========================================================");
        Ok(())
    }

    pub fn winrate_check(&mut self,agent:&mut MainAgent,num_episodes:usize) -> Result<()> {
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

    pub fn per_train_step(&mut self,agent:&mut PERDQNAgent) -> Result<f32> {
        self.scheduler.step();
        let current_lr = self.scheduler.get_lr();
        agent.set_learning_rate(current_lr);

        if self.scheduler.is_at_vally() {
            let cycle =self.scheduler.get_cycle_index();
            agent.save(&format!("{}/{}_cycle{}.safetensors",self.save_dir,self.agent_name,cycle))?;
            println!("Model saved at cycle {}",cycle);
        }

        let loss = agent.update(self.batch_size)?;
        agent.update_target_network(self.tau)?;
        Ok(loss)
    }

    pub fn per_run_episode(&mut self,agent:&mut PERDQNAgent,env:&mut SevensEnv) -> Result<(f32,f32)> {
        let mut state = env.reset();
        let mut total_loss = 0.0;
        let mut steps = 0;
        let mut done = false;
        let mut total_reward = 0.0;

        while !done {
            let action = agent.select_action(&state,&TRAIN_AGENT_ID).map_err(candle_core::Error::msg)?;
            let (next_state,reward,is_done) = env.step(action);
            
            agent.add_experience(
                state.clone(),
                action,
                reward,
                next_state.clone(),
                is_done,
            );

            if agent.buffer.size() >= self.batch_size {
                let loss = self.per_train_step(agent)?;
                total_loss += loss;
                steps += 1;
            }

            state = next_state;
            
            done = is_done;
            total_reward += reward;
        }
        Ok((if steps >0 {total_loss/steps as f32} else {0.0},total_reward))
    }

    pub fn per_train(&mut self,agent:&mut PERDQNAgent,env:&mut SevensEnv,num_episodes:usize) -> Result<()> {

        let mut reward_history = Vec::new();
        let mut total_loss_sum = 0.0; 

        for episode in 1..num_episodes {
            let (loss,episode_reward) = self.per_run_episode(agent,env)?;
            reward_history.push(episode_reward);
            total_loss_sum += loss;
            
            if episode % 100 == 0 {
                let avg_reward:f32 = reward_history.iter().rev().take(100).sum::<f32>()/100.0;
                let avg_loss:f32 = total_loss_sum/100.0;
                let current_lr = self.scheduler.get_lr();
                println!("Episode :{:>5},Ave_Reward:{:>7.2}, Ave_Loss:{:>8.4}, lr:{:>8.2e},Epsilon:{:>4.2},Beta:{:>4.2}"
                ,episode,avg_reward,avg_loss,current_lr,agent.epsilon,agent.buffer.beta);
                total_loss_sum =0.0;
                if reward_history.len() > 1000 {
                    reward_history.drain(0..reward_history.len()-500);
                }

            }
            if episode %self.save_interval == 0 {
                let path = format!("{}/{}_ep{}.safetensors",self.save_dir,self.agent_name,episode);
                agent.save(&path)?;
                println!("Model saved on episode {}",episode);
            }

            //対戦相手の更新
            if episode % 3000 == 0 {
                if let Opponent::PER(ref mut opp_agent) = env.opponent {
                    agent.copy_weights_to(opp_agent)?;
                    opp_agent.epsilon = 0.0;
                    println!("Opponent updated at episode {}",episode);
                } else {
                    let mut new_opp_agent = PERDQNAgent::new(100,1);
                    agent.copy_weights_to(&mut new_opp_agent)?;
                    new_opp_agent.epsilon = 0.0;
                    env.opponent = Opponent::PER(new_opp_agent);
                    println!("Opponent switched to new agent");
                }
            }
        }
        Ok(())

    }

    pub fn per_vs_random(&mut self,agent:&mut PERDQNAgent,env:&mut SevensEnv,num_episodes:usize) -> Result<()> {
        let mut agent_ranks = Vec::new();
        let mut rank_counts = vec![0;4];
        env.opponent = Opponent::Random(RandomAgent::new());
        println!("=============================================================");
        println!("Starting evaluation vs RandomAgent for {} episodes",num_episodes);
        println!("=============================================================");
        for episode in 1..num_episodes {
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
            println!("Final Result {} Games vs RandomAgent",num_episodes);
            println!("Mainagent Win Rate (1st place): {:.2}%",rank_counts[0] as f32/num_episodes as f32 *100.0);
            println!("Rank Rate: 1st:{:.2}% | 2nd:{:.2}% | 3rd:{:.2}% | 4th:{:.2}%",r1_rate,r2_rate,r3_rate,r4_rate);
            println!("Ave_Rank : {:.4}",ave_rank);
            println!("========================================================");
        Ok(())
    }
} 

pub struct DRNTrainer {
    scheduler:CosineAnnealingWarmRestarts,
    batch_size:usize,
    tau:f32,
    pub save_dir:String,
    pub save_interval:usize,
    pub agent_name:String,
}

impl DRNTrainer{

    pub fn new(eta_max:f64,eta_min:f64,t_0:usize,t_mult:usize,batch_size:usize,tau:f32,save_dir:String,save_interval:usize,agent_name:String) -> Self {
        Self { 
            scheduler: CosineAnnealingWarmRestarts::new(eta_max, eta_min, t_0, t_mult),
            batch_size,
            tau,
            save_dir,
            save_interval,
            agent_name,
        }
    }

    pub fn train_step(&mut self,agent:&mut DRNAgent) -> Result<(f32,f32,f32,f32)> {
        self.scheduler.step();
        let current_lr = self.scheduler.get_lr();
        agent.set_qnet_learning_rate(current_lr);
        agent.set_rnet_learning_rate(current_lr);

        if self.scheduler.is_at_vally() {
            let cycle =self.scheduler.get_cycle_index();
            agent.save(&format!("{}/{}_cycle{}.safetensors",self.save_dir,self.agent_name,cycle))?;
            println!("Model saved at cycle {}",cycle);
            let new_lambda = (cycle + 1) as f64 * 0.1;
            agent.set_lambda(new_lambda); // lambdaの更新は簡易的に谷で切り替えることにする。set_lambdaでmin(1.0)にしているため1は超えない
            println!("Set new lambda to {:.4}", &agent.lambda);
        }

        let (q_loss,r_loss,kl_loss,w_loss) = agent.update(self.batch_size)?;
        agent.update_target_network(self.tau)?;
        agent.update_target_regret_network(self.tau)?;
        Ok((q_loss,r_loss,kl_loss,w_loss))
    }

    pub fn run_episode(&mut self,agent:&mut DRNAgent,env:&mut SevensEnv) -> Result<(f32,f32,f32,f32,f32)> {
        let mut state = env.reset();
        let mut total_q_loss = 0.0;
        let mut total_r_loss = 0.0;
        let mut total_kl_loss = 0.0;
        let mut total_w_loss = 0.0;
        let mut steps = 0;
        let mut done = false;
        let mut total_reward = 0.0;

        while !done {
            let action = agent.select_action(&state,&TRAIN_AGENT_ID).map_err(candle_core::Error::msg)?;
            let (next_state,reward,is_done) = env.step(action);
            
            agent.add_experience(
                state.clone(),
                action,
                reward,
                next_state.clone(),
                is_done,
            );

            state = next_state;        
            done = is_done;
            total_reward += reward;
        }

        if agent.buffer.len() >= self.batch_size {
            let (q_loss,r_loss,kl_loss,w_loss) = self.train_step(agent)?;
            total_q_loss += q_loss;
            total_r_loss += r_loss;
            total_kl_loss += kl_loss;
            total_w_loss += w_loss;
            steps += 1;
        }



        Ok(if steps >0 {(total_q_loss/steps as f32,total_r_loss/steps as f32,total_kl_loss/steps as f32,total_w_loss/steps as f32,total_reward)} else {(0.0,0.0,0.0,0.0,total_reward)})
    }

    pub fn train(&mut self,agent:&mut DRNAgent,env:&mut SevensEnv,num_episodes:usize) -> Result<()> {

        let mut reward_history = Vec::new();
        let mut total_q_loss_sum = 0.0; 
        let mut total_r_loss_sum = 0.0;
        let mut total_kl_loss_sum = 0.0;
        let mut total_w_loss_sum = 0.0;

        for episode in 1..=num_episodes {
            let (q_loss,r_loss,kl_loss,w_loss,episode_reward) = self.run_episode(agent,env)?;
            reward_history.push(episode_reward);
            total_q_loss_sum += q_loss;
            total_r_loss_sum += r_loss;
            total_kl_loss_sum += kl_loss;
            total_w_loss_sum += w_loss;

            if episode % 100 == 0 {
                let avg_reward:f32 = reward_history.iter().rev().take(100).sum::<f32>()/100.0;
                let avg_q_loss:f32 = total_q_loss_sum/100.0;
                let avg_r_loss:f32 = total_r_loss_sum/100.0;
                let avg_kl_loss:f32 = total_kl_loss_sum/100.0;
                let avg_w_loss:f32 = total_w_loss_sum/100.0;
                let current_lr = self.scheduler.get_lr();
                println!("Episode :{:>5},Ave_Reward:{:>7.2}, Ave_Q_Loss:{:>8.4},Ave_R_Loss:{:>8.4}, Ave_KL_Loss:{:>8.4}, Ave_W_Loss:{:>8.4}, lr:{:>8.2e}"
                ,episode,avg_reward,avg_q_loss,avg_r_loss,avg_kl_loss,avg_w_loss,current_lr);
                total_q_loss_sum = 0.0;
                total_r_loss_sum = 0.0;
                total_kl_loss_sum = 0.0;
                total_w_loss_sum = 0.0;
                if reward_history.len() > 1000 {
                    reward_history.drain(0..reward_history.len()-500);
                }

            }
            if episode %self.save_interval == 0 {
                let path = format!("{}/{}_ep{}.safetensors",self.save_dir,self.agent_name,episode);
                agent.save(&path)?;
                println!("Model saved on episode {}",episode);
            }

            //対戦相手の更新
            if episode % 3000 == 0 && episode >= 9000 {
                if let Opponent::DRN(ref mut opp_agent) = env.opponent {
                    agent.copy_weights_to(opp_agent)?;
                    
                    opp_agent.set_lambda(0.0);
                    opp_agent.epsilon = 0.0;
                    println!("Opponent updated at episode {}",episode);
                } else {
                    let mut new_opp_agent = DRNAgent::new(100,1);
                    agent.copy_weights_to(&mut new_opp_agent)?;
                    
                    new_opp_agent.set_lambda(0.0);
                    new_opp_agent.epsilon = 0.0;
                    env.opponent = Opponent::DRN(new_opp_agent);
                    println!("Opponent switched to new agent");
                }
            }

            if episode % 40000 == 0 {
                if let Err(e) = self.winrate_check(agent, 10000) {
                    eprintln!("Error during winrate check: {}", e);
                } else {
                    println!("Winrate check completed successfully at episode {}", episode);
                }
            }

            if episode as f32 == 0.2 * num_episodes as f32 {
                agent.set_beta(0.02);
            } else if episode as f32 == 0.4 * num_episodes as f32 {
                agent.set_beta(0.04);
            } else if episode as f32 == 0.6 * num_episodes as f32 {
                agent.set_beta(0.06);
            } else if episode as f32 == 0.8 * num_episodes as f32 {
                agent.set_beta(0.1);
            }
        }

        let final_model_path = format!("{}/{}_final_reg.safetensors",self.save_dir,self.agent_name);
        agent.save_target_regret(&final_model_path)?;

        Ok(())

    }

    pub fn drn_vs_random(&mut self,agent:&mut DRNAgent,env:&mut SevensEnv,num_episodes:usize) -> Result<()> {
        let mut agent_ranks = Vec::new();
        let mut rank_counts = vec![0;4];
        println!("=============================================================");
        println!("Starting evaluation vs RandomAgent for {} episodes",num_episodes);
        println!("=============================================================");
        for episode in 1..num_episodes {
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
            println!("Final Result {} Games vs RandomAgent",num_episodes);
            println!("Mainagent Win Rate (1st place): {:.2}%",rank_counts[0] as f32/num_episodes as f32 *100.0);
            println!("Rank Rate: 1st:{:.2}% | 2nd:{:.2}% | 3rd:{:.2}% | 4th:{:.2}%",r1_rate,r2_rate,r3_rate,r4_rate);
            println!("Ave_Rank : {:.4}",ave_rank);
            println!("========================================================");
        Ok(())
    }

    pub fn winrate_check(&mut self,agent:&mut DRNAgent,num_episodes:usize) -> Result<()> {
        let opponent = Opponent::Random(RandomAgent::new());
        let mut env = SevensEnv::new(4,0,opponent);
        agent.eta = 1e5;
        println!("Agent lambda :{}",agent.lambda);
        println!("Agent eta: {}",agent.eta);
        println!("Agent temp: {}",agent.temp);
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

        agent.eta = 1.0;
        println!("Agent eta reset to {}",agent.eta);
        Ok(())
    }
}

pub struct DRNEPSTrainer {
    scheduler:CosineAnnealingWarmRestarts,
    batch_size:usize,
    tau:f32,
    pub save_dir:String,
    pub save_interval:usize,
    pub agent_name:String,
}

impl DRNEPSTrainer{

    pub fn new(eta_max:f64,eta_min:f64,t_0:usize,t_mult:usize,batch_size:usize,tau:f32,save_dir:String,save_interval:usize,agent_name:String) -> Self {
        Self { 
            scheduler: CosineAnnealingWarmRestarts::new(eta_max, eta_min, t_0, t_mult),
            batch_size,
            tau,
            save_dir,
            save_interval,
            agent_name,
        }
    }

    pub fn train_step(&mut self,agent:&mut DRNEPSAgent) -> Result<(f32,f32,f32)> {
        self.scheduler.step();
        let current_lr = self.scheduler.get_lr();
        agent.set_qnet_learning_rate(current_lr);
        agent.set_rnet_learning_rate(current_lr);

        if self.scheduler.is_at_vally() {
            let cycle =self.scheduler.get_cycle_index();
            agent.save(&format!("{}/{}_cycle{}.safetensors",self.save_dir,self.agent_name,cycle))?;
            println!("Model saved at cycle {}",cycle);
            let new_lambda = (cycle + 1) as f64 * 0.2;
            agent.set_lambda(new_lambda); // lambdaの更新は簡易的に谷で切り替えることにする。set_lambdaでmin(1.0)にしているため1は超えない
            println!("Set new lambda to {:.4}", &agent.lambda);
        }

        let (q_loss,r_loss,kl_loss) = agent.update(self.batch_size)?;
        agent.update_target_network(self.tau)?;
        agent.update_target_regret_network(self.tau)?;
        Ok((q_loss,r_loss,kl_loss))
    }

    pub fn run_episode(&mut self,agent:&mut DRNEPSAgent,env:&mut SevensEnv) -> Result<(f32,f32,f32,f32)> {
        let mut state = env.reset();
        let mut total_q_loss = 0.0;
        let mut total_r_loss = 0.0;
        let mut total_kl_loss = 0.0;
        let mut steps = 0;
        let mut done = false;
        let mut total_reward = 0.0;

        while !done {
            let action = agent.select_action(&state,&TRAIN_AGENT_ID).map_err(candle_core::Error::msg)?;
            let (next_state,reward,is_done) = env.step(action);
            
            agent.add_experience(
                state.clone(),
                action,
                reward,
                next_state.clone(),
                is_done,
            );

            state = next_state;        
            done = is_done;
            total_reward += reward;
        }

        if agent.buffer.len() >= self.batch_size {
            let (q_loss,r_loss,kl_loss) = self.train_step(agent)?;
            total_q_loss += q_loss;
            total_r_loss += r_loss;
            total_kl_loss += kl_loss;
            steps += 1;
        }



        Ok(if steps >0 {(total_q_loss/steps as f32,total_r_loss/steps as f32,total_kl_loss/steps as f32,total_reward)} else {(0.0,0.0,0.0,total_reward)})
    }

    pub fn train(&mut self,agent:&mut DRNEPSAgent,env:&mut SevensEnv,num_episodes:usize) -> Result<()> {

        let mut reward_history = Vec::new();
        let mut total_q_loss_sum = 0.0; 
        let mut total_r_loss_sum = 0.0;
        let mut total_kl_loss_sum = 0.0;

        for episode in 1..=num_episodes {
            let (q_loss,r_loss,kl_loss,episode_reward) = self.run_episode(agent,env)?;
            reward_history.push(episode_reward);
            total_q_loss_sum += q_loss;
            total_r_loss_sum += r_loss;
            total_kl_loss_sum += kl_loss;
            

            if episode % 100 == 0 {
                let avg_reward:f32 = reward_history.iter().rev().take(100).sum::<f32>()/100.0;
                let avg_q_loss:f32 = total_q_loss_sum/100.0;
                let avg_r_loss:f32 = total_r_loss_sum/100.0;
                let avg_kl_loss:f32 = total_kl_loss_sum/100.0;
                let current_lr = self.scheduler.get_lr();
                println!("Episode :{:>5},Ave_Reward:{:>7.2}, Ave_Q_Loss:{:>8.4},Ave_R_Loss:{:>8.4}, Ave_KL_Loss:{:>8.4}, lr:{:>8.2e}"
                ,episode,avg_reward,avg_q_loss,avg_r_loss,avg_kl_loss,current_lr);
                total_q_loss_sum = 0.0;
                total_r_loss_sum = 0.0;
                total_kl_loss_sum = 0.0;
                if reward_history.len() > 1000 {
                    reward_history.drain(0..reward_history.len()-500);
                }

            }
            if episode %self.save_interval == 0 {
                let path = format!("{}/{}_ep{}.safetensors",self.save_dir,self.agent_name,episode);
                agent.save(&path)?;
                println!("Model saved on episode {}",episode);
            }

            //対戦相手の更新
            if episode % 3000 == 0 && episode >= 6000 {
                if let Opponent::DRNEPS(ref mut opp_agent) = env.opponent {
                    agent.copy_weights_to(opp_agent)?;
                    
                    opp_agent.set_lambda(0.0);
                    opp_agent.epsilon = 0.0;
                    println!("Opponent updated at episode {}",episode);
                } else {
                    let mut new_opp_agent = DRNEPSAgent::new(100,1);
                    agent.copy_weights_to(&mut new_opp_agent)?;
                    
                    new_opp_agent.set_lambda(0.0);
                    new_opp_agent.epsilon = 0.0;
                    env.opponent = Opponent::DRNEPS(new_opp_agent);
                    println!("Opponent switched to new agent");
                }
            }

            if episode % 20000 == 0 {
                if let Err(e) = self.winrate_check(agent, 10000) {
                    eprintln!("Error during winrate check: {}", e);
                } else {
                    println!("Winrate check completed successfully at episode {}", episode);
                }
            }

            if episode as f32 == 0.2 * num_episodes as f32 {
                agent.set_beta(0.02);
            } else if episode as f32 == 0.4 * num_episodes as f32 {
                agent.set_beta(0.04);
            } else if episode as f32 == 0.6 * num_episodes as f32 {
                agent.set_beta(0.06);
            } else if episode as f32 == 0.8 * num_episodes as f32 {
                agent.set_beta(0.1);
            }
        }

        let final_model_path = format!("{}/{}_final_reg.safetensors",self.save_dir,self.agent_name);
        agent.save_target_regret(&final_model_path)?;

        Ok(())

    }

    pub fn drneps_vs(&mut self,agent:&mut DRNEPSAgent,env:&mut SevensEnv,num_episodes:usize) -> Result<()> {
        let mut agent_ranks = Vec::new();
        let mut rank_counts = vec![0;4];
        println!("=============================================================");
        println!("Starting evaluation vs RandomAgent for {} episodes",num_episodes);
        println!("=============================================================");
        for episode in 1..num_episodes {
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
            println!("Final Result {} Games vs RandomAgent",num_episodes);
            println!("Mainagent Win Rate (1st place): {:.2}%",rank_counts[0] as f32/num_episodes as f32 *100.0);
            println!("Rank Rate: 1st:{:.2}% | 2nd:{:.2}% | 3rd:{:.2}% | 4th:{:.2}%",r1_rate,r2_rate,r3_rate,r4_rate);
            println!("Ave_Rank : {:.4}",ave_rank);
            println!("========================================================");
        Ok(())
    }

    pub fn winrate_check(&mut self,agent:&mut DRNEPSAgent,num_episodes:usize) -> Result<()> {
        let opponent = Opponent::Random(RandomAgent::new());
        let mut env = SevensEnv::new(4,0,opponent);
        println!("Agent lambda :{}",agent.lambda);
        println!("Agent delta: {}",agent.delta);
        println!("Agent temp: {}",agent.temp);
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
}
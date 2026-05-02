use sevens::card::Card;
use sevens::env::{SevensEnv,PASS_ACTION};
use sevens::agent::{MainAgent, RandomAgent, Opponent};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --- 初期設定 ---
    let num_players = 4;
    let agent_id = 0;
    //let main_agent = RandomAgent::new();//デバッグ用
    let mut main_agent = MainAgent::new(100, 1);
    main_agent.load("checkpoints/dqn_v1.2.1_cycle3.safetensors").expect("Failed to load model.check the path!");
    main_agent.epsilon = 0.0; // 決定論的な行動を選択

    let opponent = Opponent::Random(RandomAgent::new());
    let mut env = SevensEnv::new(num_players, agent_id, opponent);

    println!("\n==================================================");
    println!("   🃏 七並べ AI推論デバッグログ 🃏");
    println!("==================================================");

    let mut state = env.reset();
    let mut step_count = 0;

    loop {
        step_count += 1;
        let current_p = env.state.current_player;

        println!("\n--- [ STEP: {:03} | Player {} の番 ] ---", step_count, current_p);

        // 1. 場の状況を表示（スートごとに1行）
        display_visual_field(&env.state.field);

        // 2. 全プレイヤーの状態（手札・パス・ステータス）を表示
        display_all_hands(&env);

        println!("Pass Counts:{:?}", &env.state.pass_counts);
        println!("Finished Order: {:?}",&env.state.finished_order);
        println!("Eliminated: {:?}",&env.state.eliminated);
        

        // 3. 行動の選択と表示
        let mask = env.get_legal_action_mask();

        display_legal_actions(&mask);
        
        // 推論（自分か相手かに関わらず、ロジック上の選択を明示）
        let action = main_agent.infer_q(&state,&env.agent_id)?;
        

        println!(">> 行動選択: 【 {} 】", format_action_visual(action));

        // 4. 環境の更新
        let (next_state, reward, done) = env.step(action);

        if reward != 0.0 {
            println!("✨ 報酬獲得: {:.2}", reward);
        }

        state = next_state;

        if done {
            println!("========================================================");
            println!("🏁 ゲーム終了！");
            println!("最終順位: {:?}", env.state.finished_order);
            if !env.state.eliminated.is_empty() {
                println!("脱落者: {:?}", env.state.eliminated);
            }
            println!("Reward: {:.2}",reward);
            println!("========================================================");
            break;
        }

        if step_count > 200 { break; }
    }

    Ok(())
}

/// 場の状況をトランプらしく表示
fn display_visual_field(field: &[bool]) {
    let suits = ["♦", "♥", "♣", "♠"];
    println!("【 現在の場 】");
    for (s_idx, suit) in suits.iter().enumerate() {
        print!("  {} | ", suit);
        for rank in 0..13 {
            let id = s_idx * 13 + rank;
            if field[id] {
                print!("{:>2} ", format_rank(rank as u8));
            } else {
                print!("-- "); // まだ置かれていない場所
            }
        }
        println!();
    }
}

/// 全プレイヤーの手札を表示
fn display_all_hands(env: &SevensEnv) {
    for p in 0..env.num_players {
        let prefix = if p == env.agent_id { "★" } else { "  " };
        let status = if env.state.finished_order.contains(&p) { " (GOAL!)" }
                    else if env.state.eliminated.contains(&p) { " (DOBON)" }
                    else { "" };

        let mut hand = env.state.hands[p].clone();
        hand.sort();
        let cards_str: Vec<String> = hand.iter().map(|&id| Card(id).to_string()).collect();
        
        println!("{}[P{}] パス:{}/4 {} | {}", 
            prefix, p, env.state.pass_counts[p], status, cards_str.join(" "));
    }
}

/// アクションを[♥ 7]のような形式で表示
fn format_action_visual(action: u8) -> String {
    if action == PASS_ACTION {
        "PASS (パス)".to_string()
    } else {
        format!("{}", Card(action))
    }
}

/// ランクの数字を A, J, Q, K に変換
fn format_rank(rank: u8) -> &'static str {
    match rank {
        0 => "A",
        9 => "10",
        10 => "J",
        11 => "Q",
        12 => "K",
        _ => ["2","3","4","5","6","7","8","9"][(rank-1) as usize],
    }
}

fn display_legal_actions(mask: &[f32]) {
    let mut legals = Vec::new();
    let mut can_pass = false;

    for (i,&m) in mask.iter().enumerate() {
        if m > 0.0 {
            if i == PASS_ACTION as usize {
                can_pass = true;
            } else {
                legals.push(Card(i as u8).to_string());
            }
        }
    }

    print!("[合法手]:");
    if legals.is_empty() {
        print!("なし");
    } else {
        print!("{}",legals.join(","));
    }

    if can_pass {
        print!(" + PASS");
    }
    println!();
}

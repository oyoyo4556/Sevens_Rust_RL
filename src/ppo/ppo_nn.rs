use candle_core::{Result,Tensor};
use candle_nn::{linear,Linear,Module,VarBuilder,LayerNorm,LayerNormConfig};

pub struct ResidualBlock{
    fc1:Linear,
    ln1:LayerNorm,
    fc2:Linear,
}

impl ResidualBlock {
    pub fn new(dim:usize,vb:VarBuilder) -> Result<Self> {
        let fc1 = candle_nn::linear(dim,2*dim,vb.pp("fc1"))?;
        let ln1 = candle_nn::layer_norm(dim,LayerNormConfig::default(),vb.pp("ln1"))?;
        let fc2 = candle_nn::linear(dim*2,dim,vb.pp("fc2"))?;
        Ok(Self{fc1,ln1,fc2})
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
    // 入力を正規化 (Pre-LN)
    let h = self.ln1.forward(x)?;
    // 特徴量の抽出 (dim -> 2*dim -> dim)
    let h = self.fc1.forward(&h)?.relu()?;
    let h = self.fc2.forward(&h)?;
    // 残差結合
    x.add(&h)
    }

}

pub struct PPONet {
    input_layer:Linear,
    res:ResidualBlock,
    buffer_layer:Linear,
    ln:LayerNorm,
    fc_ac:Linear,
    fc_cri:Linear,
    actor_head: Linear,  // 53次元 (各カード + PASS)
    critic_head: Linear,
}

impl PPONet {
    pub fn new(state_dim:usize,hidden_dim:usize,action_dim:usize,vb: VarBuilder) -> Result<Self> {
        
        let hidden2_dim = if hidden_dim % 2 == 0 {hidden_dim / 2} else {(hidden_dim + 1 )/2};

        let input_layer = linear(state_dim,hidden_dim,vb.pp("input_layer"))?;
        let res = ResidualBlock::new(hidden_dim,vb.pp("res"))?;
        let ln = candle_nn::layer_norm(hidden_dim,candle_nn::LayerNormConfig::default(),vb.pp("ln"))?;
        let buffer_layer = linear(hidden_dim,hidden2_dim,vb.pp("buffer_layer"))?;
        let fc_ac = linear(hidden2_dim,hidden2_dim,vb.pp("fc_ac"))?;
        let fc_cri = linear(hidden2_dim,hidden2_dim,vb.pp("fc_cri"))?;
        let critic_head = linear(hidden2_dim,1,vb.pp("critic"))?;
        let actor_head = linear(hidden2_dim,action_dim,vb.pp("actor"))?;

        Ok(Self {input_layer,res,buffer_layer,ln,fc_ac,fc_cri,critic_head,actor_head})
    }

    pub fn forward(&self,x:&Tensor,mask:&Tensor) -> Result<(Tensor,Tensor)> {
        let h = self.input_layer.forward(&x)?;
        let h = self.res.forward(&h)?;
        let h = self.ln.forward(&h)?;
        let h = self.buffer_layer.forward(&h)?.relu()?;

        let logits = self.fc_ac.forward(&h)?.relu()?;
        let logits = self.actor_head.forward(&logits)?;
        let value = self.fc_cri.forward(&h)?.relu()?;
        let value = self.critic_head.forward(&value)?.squeeze(1)?; 

        // ★ 非合法手に大きな負の数 (-1e9) を加算して Masking
        // mask: 1.0 (合法), 0.0 (非合法)
        let mask_offset = mask.affine(-1.0,1.0)?.affine(-1e9f64,0.0)?;
        let masked_logits = logits.add(&mask_offset)?;

        Ok((masked_logits, value))
    }
}
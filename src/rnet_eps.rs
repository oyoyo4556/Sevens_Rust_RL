use candle_core::{Result,Tensor};
use candle_nn::{linear,Linear,Module,VarBuilder,LayerNorm,LayerNormConfig,Init};
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

pub struct DuelingQNet{
    input_layer:Linear,
    res:ResidualBlock,
    final_ln:LayerNorm,
    buffer_layer:Linear,
    value:Linear,
    advantage:Linear,
}

impl DuelingQNet{
    pub fn new(state_dim:usize,hidden_dim:usize,action_dim:usize,vb: VarBuilder) -> Result<Self> {
        
        let hidden2_dim = if hidden_dim % 2 == 0 {hidden_dim / 2} else {(hidden_dim + 1 )/2};

        let input_layer = linear(state_dim,hidden_dim,vb.pp("input_layer"))?;
        let res = ResidualBlock::new(hidden_dim,vb.pp("res"))?;
        let final_ln = candle_nn::layer_norm(hidden_dim,candle_nn::LayerNormConfig::default(),vb.pp("final_ln"))?;
        let buffer_layer = linear(hidden_dim,hidden2_dim,vb.pp("buffer_layer"))?;
        let value = linear(hidden2_dim,1,vb.pp("value"))?;
        let advantage = linear(hidden2_dim,action_dim,vb.pp("advantage"))?;

        Ok(Self {input_layer,res,final_ln,buffer_layer,value,advantage})
    }

    pub fn forward(&self,x:&Tensor,mask:&Tensor) -> Result<Tensor> {
        let mut x = self.input_layer.forward(x)?;
        x = x.relu()?;
        x = self.res.forward(&x)?;
        x = self.final_ln.forward(&x)?;
        x = self.buffer_layer.forward(&x)?;
        x = x.relu()?;
        let v = self.value.forward(&x)?;
        let a = self.advantage.forward(&x)?;

        let masked_a = a.broadcast_mul(mask)?;
        let legal_counts = mask.sum_keepdim(1)?.affine(1.0,1e-8)?;
        
        let a_mean = masked_a.sum_keepdim(1)?.broadcast_div(&legal_counts)?;

        let advantage_centered = a.broadcast_sub(&a_mean)?.broadcast_mul(mask)?;
        let q = advantage_centered.broadcast_add(&v)?;

        Ok(q)
    }
}

pub fn customlinear(in_dim: usize, out_dim: usize,bias:f64, vb: VarBuilder) -> Result<Linear> {
    let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints((out_dim, in_dim), "weight", init_ws)?;

    let bs = vb.get_with_hints(
        out_dim,
        "bias",
        Init::Const(bias)
    )?;

    Ok(Linear::new(ws, Some(bs)))
}

pub fn smooth_relu(x: &Tensor, a: f64) -> Result<Tensor> {
    let x_sq = x.sqr()?;
    let inside_sqrt = x_sq.affine(1.0, a)?; // x^2 + a
    let sqrt_part = inside_sqrt.sqrt()?;
    let sum = x.add(&sqrt_part)?;
    sum.affine(0.5, 0.0) 
}

pub fn stable_softplus(x: &Tensor) -> Result<Tensor> {
    let max_x_0 = x.relu()?;
    let abs_x = x.abs()?;
    let neg_abs_x = abs_x.neg()?;
    let exp_neg_abs = neg_abs_x.exp()?;
    let plus_one = exp_neg_abs.affine(1.0, 1.0)?;
    let log_term = plus_one.log()?;

    max_x_0.add(&log_term)
}

pub struct RNetEPS {
    input_layer:Linear,
    res1:ResidualBlock,
    final_ln:LayerNorm,
    buffer_layer:Linear,
    regret:Linear,
    fcr:Linear,
    fce:Linear,
    eps:Linear,
}

impl RNetEPS{
    pub fn new(state_dim:usize,hidden_dim:usize,action_dim:usize,vb: VarBuilder) -> Result<Self> {
        
        let hidden2_dim = if hidden_dim % 2 == 0 {hidden_dim / 2} else {(hidden_dim + 1 )/2};

        let input_layer = linear(state_dim,hidden_dim,vb.pp("input_layer"))?;
        let res1 = ResidualBlock::new(hidden_dim,vb.pp("res1"))?;
        let final_ln = candle_nn::layer_norm(hidden_dim,candle_nn::LayerNormConfig::default(),vb.pp("final_ln"))?;
        let buffer_layer = linear(hidden_dim,hidden2_dim,vb.pp("buffer_layer"))?;
        let regret = customlinear(hidden2_dim, action_dim, 3.0, vb.pp("regret"))?;
        let fcr = linear(hidden2_dim, hidden2_dim, vb.pp("fcr"))?;
        let fce = linear(hidden2_dim, hidden2_dim, vb.pp("fce"))?;
        let eps = linear(hidden2_dim, action_dim, vb.pp("eps"))?;


        Ok(Self {input_layer,res1,final_ln,buffer_layer,regret,fcr,fce,eps})
    }

    pub fn forward(&self,x:&Tensor) -> Result<(Tensor,Tensor)> {
        let mut x = self.input_layer.forward(x)?;
        x = x.relu()?;
        x = self.res1.forward(&x)?;
        x = self.final_ln.forward(&x)?;
        x = self.buffer_layer.forward(&x)?;
        x = x.relu()?;

        let r = self.fcr.forward(&x)?.relu()?;
        let r = smooth_relu(&self.regret.forward(&r)?, 0.0001)?; 

        let e = self.fce.forward(&x)?.relu()?;
        let e = stable_softplus(&self.eps.forward(&e)?)?;
        Ok((r,e))
    }
}
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

    pub fn forward(&self,x:&Tensor) -> Result<Tensor> {
        let residual = x;

        let mut out = self.ln1.forward(x)?;
        out = self.fc1.forward(&out)?;
        out = out.relu()?;
        out = self.fc2.forward(&out)?;

        out.add(residual)
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

pub struct RNet {
    input_layer:Linear,
    res:ResidualBlock,
    final_ln:LayerNorm,
    buffer_layer:Linear,
    regret:Linear,
}

impl RNet{
    pub fn new(state_dim:usize,hidden_dim:usize,action_dim:usize,vb: VarBuilder) -> Result<Self> {
        
        let hidden2_dim = if hidden_dim % 2 == 0 {hidden_dim / 2} else {(hidden_dim + 1 )/2};

        let input_layer = linear(state_dim,hidden_dim,vb.pp("input_layer"))?;
        let res = ResidualBlock::new(hidden_dim,vb.pp("res"))?;
        let final_ln = candle_nn::layer_norm(hidden_dim,candle_nn::LayerNormConfig::default(),vb.pp("final_ln"))?;
        let buffer_layer = linear(hidden_dim,hidden2_dim,vb.pp("buffer_layer"))?;
        let regret = customlinear(hidden2_dim, action_dim, 3.0, vb.pp("regret"))?;


        Ok(Self {input_layer,res,final_ln,buffer_layer,regret})
    }

    pub fn forward(&self,x:&Tensor) -> Result<Tensor> {
        let mut x = self.input_layer.forward(x)?;
        x = x.relu()?;
        x = self.res.forward(&x)?;
        x = self.final_ln.forward(&x)?;
        x = self.buffer_layer.forward(&x)?;
        x = x.relu()?;
        x = self.regret.forward(&x)?;
        x = x.gelu()?;

        Ok(x)
    }
}
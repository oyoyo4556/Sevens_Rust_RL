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
    res1:ResidualBlock,
    res2:ResidualBlock,
    final_ln:LayerNorm,
    value:Linear,
    advantage:Linear,
}

impl DuelingQNet{
    pub fn new(state_dim:usize,hidden_dim:usize,action_dim:usize,vb: VarBuilder) -> Result<Self> {
        let input_layer = linear(state_dim,hidden_dim,vb.pp("input_layer"))?;
        let res1 = ResidualBlock::new(hidden_dim,vb.pp("res1"))?;
        let res2 = ResidualBlock::new(hidden_dim,vb.pp("res2"))?;
        let final_ln = candle_nn::layer_norm(hidden_dim,candle_nn::LayerNormConfig::default(),vb.pp("final_ln"))?;
        let value = linear(hidden_dim,1,vb.pp("value"))?;
        let advantage = linear(hidden_dim,action_dim,vb.pp("advantage"))?;

        Ok(Self {input_layer,res1,res2,final_ln,value,advantage})
    }

    pub fn forward(&self,x:&Tensor) -> Result<Tensor> {
        let mut x = self.input_layer.forward(x)?;
        x = x.relu()?;
        x = self.res1.forward(&x)?;
        x = self.res2.forward(&x)?;
        x = self.final_ln.forward(&x)?;
        let v = self.value.forward(&x)?;
        let a = self.advantage.forward(&x)?;
        
        let a_mean = a.mean_keepdim(1)?;
        let q = a.broadcast_sub(&a_mean)?.broadcast_add(&v)?;

        Ok(q)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_core::DType;

    #[test]
    fn test_network_dimensions() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device); // 重みは全部0でOK
        
        let state_dim = 233; // 今回増やした次元
        let hidden_dim = 256;
        let action_dim = 53;
        
        let net = DuelingQNet::new(state_dim, hidden_dim, action_dim, vb).unwrap();
        
        // バッチサイズ1のダミーデータを作成
        let dummy_input = Tensor::zeros(&[1, state_dim], DType::F32, &device).unwrap();
        
        // 実行してエラーが出ないか、次元が [1, 53] かを確認
        let output = net.forward(&dummy_input).expect("Dimension Mismatch!");
        assert_eq!(output.dims(), &[1, 53]);
        
        println!("Network Dimension Check: Passed!");
    }
}


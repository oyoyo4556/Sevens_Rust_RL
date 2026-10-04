use candle_core::{Result, Tensor};

pub fn qhoe_loss(
    pred: &Tensor,
    target: &Tensor,
    epsilon: &Tensor,
    t: f64,
    eps_min: f64,
    eps_max: f64,
) -> Result<Tensor> {
    // 形状の一致確認
    if pred.dims() != target.dims() || pred.dims() != epsilon.dims() {
        return Err(candle_core::Error::Msg(format!(
            "pred, target, and epsilon must have the same shape, got pred: {:?}, target: {:?}, epsilon: {:?}",
            pred.dims(),
            target.dims(),
            epsilon.dims()
        )));
    }

    // eps_eff = eps_min + eps_max * tanh(epsilon / eps_max)
    let eps_div_max = (epsilon / eps_max)?;
    let eps_eff = ((eps_div_max.tanh()? * eps_max)? + eps_min)?;

    // x = eps_eff / (2 * T)
    let x = (&eps_eff / (2.0 * t))?;
    
    // 勾配・数値安定のための安全装置：x -> clamp(min=1e-6)
    let x_clamped = x.clamp(1e-6, f64::MAX)?;

    // diff = pred - target
    let diff = (pred - target)?;
    let diff_sq = (&diff * &diff)?;

    // term1 = eps_eff * (pred - target)^2
    let term1 = (&eps_eff * &diff_sq)?;

    // term2 = T * log(2 * sinh(x))
    // 2 * sinh(x) = exp(x) - exp(-x)
    let exp_x = x_clamped.exp()?;
    let exp_neg_x = x_clamped.neg()?.exp()?;
    let two_sinh = (&exp_x - &exp_neg_x)?;
    let term2 = (two_sinh.log()? * t)?;

    // term3 = 0.5 * eps_eff
    let term3 = (&eps_eff * 0.5)?;

    // loss = term1 - term2 + term3
    let loss = ((&term1 - &term2)? + &term3)?;

    // バッチ全体での平均値（全平均）
    loss.mean_all()
}
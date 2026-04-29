pub struct CosineAnnealingWarmRestarts {
    eta_max:f64,
    eta_min:f64,
    t_0:f64,
    t_mult:f64,
    last_step:f64,
}

impl CosineAnnealingWarmRestarts {
    pub fn new(eta_max:f64,eta_min:f64,t_0:usize,t_mult:usize) ->Self {
        Self { 
            eta_max,
            eta_min,
            t_0:t_0 as f64,
            t_mult:t_mult as f64,
            last_step:0.0,
        }
    }

    pub fn get_lr(&self) -> f64 {
        let t_cur = self.last_step;
        let (t_i,t_start_of_cycle) = if self.t_mult > 1.0 {
            let i =(1.0 + t_cur*(self.t_mult - 1.0)/self.t_0).log(self.t_mult).floor();
            let t_i = self.t_0 * self.t_mult.powf(i);
            let t_start_of_cycle = self.t_0 * (1.0 -self.t_mult.powf(i))/(1.0 -self.t_mult);
            (t_i,t_start_of_cycle)
        } else {
            let t_i =self.t_0;
            let t_start_of_cycle = (t_cur/self.t_0).floor()*self.t_0;
        (t_i,t_start_of_cycle)        
        };

        let t_relative = t_cur - t_start_of_cycle;

        let pi = std::f64::consts::PI;
        self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (1.0 + (pi * t_relative / t_i).cos())
    }

    pub fn step(&mut self) {
        self.last_step += 1.0;
    }

    pub fn get_cycle_index(&self) -> usize {
        self.calculate_cycle_index(self.last_step)
    }

    pub fn is_at_vally(&self) -> bool {
        let current_idx = self.calculate_cycle_index(self.last_step);
        let next_idx = self.calculate_cycle_index(self.last_step + 1.0);
        next_idx > current_idx 
    }

    fn calculate_cycle_index(&self,t:f64) -> usize{
        if self.t_mult > 1.0 {
            ((1.0 + t*(self.t_mult - 1.0)/self.t_0).log(self.t_mult)).floor() as usize
        } else {
            (t/self.t_0).floor() as usize
        }
    }
}
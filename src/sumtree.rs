
pub struct SumTree {
    pub capacity:usize,
    pub sum_tree:Vec<f32>,
    pub min_tree:Vec<f32>,
}

impl SumTree {
    pub fn new(capacity:usize) -> Self {
        Self { 
            capacity, 
            sum_tree: vec![0.0; 2 * capacity - 1],
            min_tree: vec![f32::INFINITY; 2 * capacity - 1],
        } 
    }

    pub fn update(&mut self, data_idx:usize,priority:f32) {
        let mut tree_idx = data_idx + self.capacity -1 ;

        self.sum_tree[tree_idx] = priority;
        self.min_tree[tree_idx] = priority;

        while tree_idx > 0 {
            tree_idx = (tree_idx - 1)/2;
            let left = 2 * tree_idx + 1;
            let right = left + 1;
            //sum_treeは足し算で更新
            self.sum_tree[tree_idx] = self.sum_tree[left] + self.sum_tree[right];

            //min_treeは小さい方を採用
            let l_min = self.min_tree[left];
            let r_min = self.min_tree[right];
            self.min_tree[tree_idx] = l_min.min(r_min);
        }
    }

    pub fn get_leaf(&self,mut value:f32) -> usize {
        let mut parent_idx = 0;

        while parent_idx < self.capacity - 1 {
            let left_child_idx = 2 * parent_idx + 1;
            let right_child_idx = left_child_idx + 1;

            if value <= self.sum_tree[left_child_idx] {
                parent_idx = left_child_idx;
            } else {
                value -= self.sum_tree[left_child_idx];
                parent_idx = right_child_idx;
            }
        }
        parent_idx - (self.capacity - 1)
    }

    pub fn total_priority(&self) -> f32 {
        self.sum_tree[0]
    }

    pub fn get_priority(&self,data_idx:usize) -> f32 {
        self.sum_tree[data_idx + self.capacity - 1]
    }

    pub fn min_priority(&self) -> f32 {
        let min_val = self.min_tree[0];
        if min_val == f32::INFINITY {
            return 1.0; //ここには来ないはずだが安全策
        }

        min_val.max(1e-8)
    }
}

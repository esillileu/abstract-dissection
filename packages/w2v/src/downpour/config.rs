use crate::config::{Real, Status};

pub const DEFAULT_PS_SHARDS: usize = 2;
pub const DEFAULT_MINI_BATCH_TARGETS: usize = 100;
pub const DEFAULT_ADAGRAD_GAMMA: Real = 0.05;
pub const DEFAULT_ADAGRAD_EPSILON: Real = 1e-6;
pub const DEFAULT_QUEUE_CAPACITY: usize = 32;

#[derive(Clone, Debug, PartialEq)]
pub struct DownpourConfig {
    pub parameter_server_shards: usize,
    pub mini_batch_targets: usize,
    pub adagrad_gamma: Real,
    pub adagrad_epsilon: Real,
    pub fetch_interval: usize,
    pub push_interval: usize,
    pub queue_capacity: usize,
}

impl Default for DownpourConfig {
    fn default() -> Self {
        Self {
            parameter_server_shards: DEFAULT_PS_SHARDS,
            mini_batch_targets: DEFAULT_MINI_BATCH_TARGETS,
            adagrad_gamma: DEFAULT_ADAGRAD_GAMMA,
            adagrad_epsilon: DEFAULT_ADAGRAD_EPSILON,
            fetch_interval: 1,
            push_interval: 1,
            queue_capacity: DEFAULT_QUEUE_CAPACITY,
        }
    }
}

impl DownpourConfig {
    pub fn validate(&self) -> Status {
        if self.parameter_server_shards == 0
            || self.mini_batch_targets == 0
            || !self.adagrad_gamma.is_finite()
            || self.adagrad_gamma <= 0.0
            || !self.adagrad_epsilon.is_finite()
            || self.adagrad_epsilon <= 0.0
            || self.fetch_interval != 1
            || self.push_interval != 1
            || self.queue_capacity == 0
        {
            Status::InvalidArgument
        } else {
            Status::Ok
        }
    }
}

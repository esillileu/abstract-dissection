use crate::{ObjectiveKind, RngAlgorithm, Status, config::Real};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HiddenActivation {
    Tanh,
    Sigmoid,
}

/// Scientific choices are required explicitly; this type has no paper defaults.
#[derive(Clone, Debug, PartialEq)]
pub struct NnlmTrainingConfig {
    pub embedding_dimension: usize,
    pub history_length: usize,
    pub hidden_dimension: usize,
    pub hidden_activation: HiddenActivation,
    pub epochs: usize,
    pub thread_count: usize,
    pub objective_kind: ObjectiveKind,
    pub initial_learning_rate: Real,
    pub observation_interval: usize,
    pub root_seed: u64,
    pub rng_algorithm: RngAlgorithm,
}

impl NnlmTrainingConfig {
    pub fn digest(&self) -> String {
        let mut hash = crate::identity::StableDigest::new();
        for value in [
            self.embedding_dimension as u64,
            self.history_length as u64,
            self.hidden_dimension as u64,
            self.hidden_activation as u64,
            self.epochs as u64,
            self.thread_count as u64,
            self.objective_kind as u64,
            self.initial_learning_rate.to_bits() as u64,
            self.observation_interval as u64,
            self.root_seed,
            self.rng_algorithm as u64,
        ] {
            hash.value(value);
        }
        hash.finish()
    }
    pub fn validate(&self) -> Status {
        if self.embedding_dimension == 0
            || self.history_length == 0
            || self.hidden_dimension == 0
            || self.epochs == 0
            || self.thread_count == 0
            || self.objective_kind != ObjectiveKind::HierarchicalSoftmax
            || !self.initial_learning_rate.is_finite()
            || self.initial_learning_rate <= 0.0
            || self
                .embedding_dimension
                .checked_mul(self.history_length)
                .and_then(|n| n.checked_mul(self.hidden_dimension))
                .is_none_or(|n| n > isize::MAX as usize / size_of::<Real>())
        {
            Status::InvalidArgument
        } else {
            Status::Ok
        }
    }
}

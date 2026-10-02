use super::config::DownpourConfig;
use super::replica::ReplicaState;
use crate::{
    config::{Real, TrainingConfig},
    identity::StableDigest,
    vocab::VocabularyState,
};

pub const DOWNPOUR_TRAINING_STATE_SCHEMA_VERSION: u32 = 2;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DownpourStateDescriptor {
    pub schema_version: u32,
    pub config_digest: String,
    pub vocabulary_digest: String,
    pub corpus_digest: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DownpourTrainingState {
    pub descriptor: DownpourStateDescriptor,
    pub completed_epochs: usize,
    pub processed_tokens: u64,
    pub vocabulary: VocabularyState,
    pub input_embeddings: Vec<Real>,
    pub output_embeddings: Vec<Real>,
    pub input_adagrad: Vec<Real>,
    pub output_adagrad: Vec<Real>,
    pub replicas: Vec<ReplicaState>,
}

pub fn downpour_config_digest(training: &TrainingConfig, downpour: &DownpourConfig) -> String {
    let mut hash = StableDigest::new();
    for value in [
        training.model_kind as u64,
        training.objective_kind as u64,
        training.embedding_dimension as u64,
        training.window_radius as u64,
        training.context_policy as u64,
        training.epochs as u64,
        training.thread_count as u64,
        training.learning_rate_update_interval as u64,
        training.observation_interval as u64,
        training.initial_learning_rate.to_bits() as u64,
        training.subsampling_threshold.to_bits() as u64,
        training.negative_sample_count as u64,
        training.root_seed,
        training.rng_algorithm as u64,
        training.negative_table_size as u64,
        training.sigmoid_table_size as u64,
        training.sigmoid_max.to_bits() as u64,
        training.hs_out_of_range_policy as u64,
        training.update_strategy as u64,
        // Downpour config extension:
        downpour.parameter_server_shards as u64,
        downpour.mini_batch_targets as u64,
        downpour.adagrad_gamma.to_bits() as u64,
        downpour.adagrad_epsilon.to_bits() as u64,
        downpour.fetch_interval as u64,
        downpour.push_interval as u64,
        downpour.queue_capacity as u64,
    ] {
        hash.value(value);
    }
    hash.finish()
}

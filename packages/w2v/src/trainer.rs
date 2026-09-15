use crate::{
    config::TrainingConfig,
    corpus::Corpus,
    model::{Model, SigmoidTable},
    vocab::{NegativeSampler, Vocabulary},
};
use std::sync::atomic::AtomicU64;

/// Borrows corpus, vocabulary, and model as the C trainer does.
pub struct Trainer<'a> {
    pub corpus: &'a Corpus,
    pub vocab: &'a Vocabulary,
    pub model: &'a Model,
    pub negative_sampler: NegativeSampler,
    pub sigmoid_table: SigmoidTable,
    pub config: TrainingConfig,
    pub processed_tokens: AtomicU64,
}

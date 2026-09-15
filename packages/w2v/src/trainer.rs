use crate::{
    config::{ObjectiveKind, Status, TrainingConfig},
    corpus::Corpus,
    model::{Model, SigmoidTable},
    training::worker::Worker,
    vocab::{NegativeSampler, Vocabulary},
};
use std::{
    sync::atomic::{AtomicU64, Ordering},
    thread,
};

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

impl<'a> Trainer<'a> {
    pub fn create(
        corpus: &'a Corpus,
        vocab: &'a Vocabulary,
        model: &'a Model,
        config: &TrainingConfig,
    ) -> Result<Self, Status> {
        if config.validate() != Status::Ok
            || model.vocab_size != vocab.entries.len()
            || model.embedding_dimension != config.embedding_dimension
            || (config.objective_kind == ObjectiveKind::NegativeSampling && vocab.entries.len() < 2)
        {
            return Err(Status::InvalidArgument);
        }
        let sigmoid_table =
            SigmoidTable::initialize(config.sigmoid_table_size, config.sigmoid_max)?;
        let negative_sampler = if config.objective_kind == ObjectiveKind::NegativeSampling {
            NegativeSampler::initialize(vocab, config.negative_table_size)?
        } else {
            NegativeSampler::default()
        };
        Ok(Self {
            corpus,
            vocab,
            model,
            negative_sampler,
            sigmoid_table,
            config: config.clone(),
            processed_tokens: AtomicU64::new(0),
        })
    }

    pub fn processed_tokens(&self) -> u64 {
        self.processed_tokens.load(Ordering::Relaxed)
    }

    pub fn train(&self) -> Result<(), Status> {
        if self.config.thread_count == 0 {
            return Err(Status::InvalidArgument);
        }
        self.processed_tokens.store(0, Ordering::Relaxed);
        if self.config.thread_count == 1 {
            return Worker::initialize(self, 0)?.run(self);
        }
        thread::scope(|scope| {
            let mut handles = Vec::new();
            handles
                .try_reserve_exact(self.config.thread_count)
                .map_err(|_| Status::OutOfMemory)?;
            for worker_id in 0..self.config.thread_count {
                match thread::Builder::new().spawn_scoped(scope, move || {
                    Worker::initialize(self, worker_id)?.run(self)
                }) {
                    Ok(handle) => handles.push(handle),
                    Err(_) => {
                        for handle in handles {
                            let _ = handle.join();
                        }
                        return Err(Status::ThreadError);
                    }
                }
            }
            let mut result = Ok(());
            for handle in handles {
                match handle.join() {
                    Ok(Ok(())) => {}
                    Ok(Err(status)) if result.is_ok() => result = Err(status),
                    Err(_) if result.is_ok() => result = Err(Status::ThreadError),
                    _ => {}
                }
            }
            result
        })
    }
}

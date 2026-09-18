use crate::{
    config::{ObjectiveKind, Real, Status, TrainingConfig},
    corpus::Corpus,
    identity::StableDigest,
    model::{EmbeddingKind, Model, SigmoidTable},
    training::worker::{Worker, WorkerState},
    vocab::{NegativeSampler, Vocabulary, VocabularyState},
};
use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    thread,
    time::Instant,
};

pub const TRAINING_STATE_SCHEMA_VERSION: u32 = 2;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateDescriptor {
    pub schema_version: u32,
    pub config_digest: String,
    pub vocabulary_digest: String,
    pub corpus_digest: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrainingState {
    pub descriptor: StateDescriptor,
    pub completed_epochs: usize,
    pub processed_tokens: u64,
    pub vocabulary: VocabularyState,
    pub input_embeddings: Vec<Real>,
    pub output_embeddings: Vec<Real>,
    pub workers: Vec<WorkerState>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EpochReport {
    pub epoch: usize,
    pub epoch_tokens: u64,
    pub processed_tokens: u64,
    pub learning_rate: Real,
    pub elapsed_seconds: f64,
    pub tokens_per_second: f64,
    pub objective_loss_sum: f64,
    pub objective_loss_count: u64,
    pub observations: Vec<Observation>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Observation {
    pub epoch: usize,
    pub processed_tokens: u64,
    pub learning_rate: Real,
    pub objective_loss_sum: f64,
    pub objective_loss_count: u64,
    pub elapsed_seconds: f64,
    pub tokens_per_second: f64,
}

pub struct Trainer {
    pub corpus: Arc<Corpus>,
    pub corpus_digest: String,
    pub vocab: Arc<Vocabulary>,
    pub model: Arc<Model>,
    pub negative_sampler: NegativeSampler,
    pub sigmoid_table: SigmoidTable,
    pub config: TrainingConfig,
    pub processed_tokens: AtomicU64,
}

pub struct TrainingSession {
    trainer: Arc<Trainer>,
    workers: Vec<Worker>,
    completed_epochs: usize,
    running: bool,
}

struct TrainingGuard<'a>(&'a Model);

impl Drop for TrainingGuard<'_> {
    fn drop(&mut self) {
        self.0.end_training();
    }
}

fn config_digest(config: &TrainingConfig) -> String {
    let mut hash = StableDigest::new();
    for value in [
        config.model_kind as u64,
        config.objective_kind as u64,
        config.embedding_dimension as u64,
        config.window_radius as u64,
        config.context_policy as u64,
        config.epochs as u64,
        config.thread_count as u64,
        config.learning_rate_update_interval as u64,
        config.observation_interval as u64,
        config.initial_learning_rate.to_bits() as u64,
        config.subsampling_threshold.to_bits() as u64,
        config.negative_sample_count as u64,
        config.root_seed,
        config.rng_algorithm as u64,
        config.negative_table_size as u64,
        config.sigmoid_table_size as u64,
        config.sigmoid_max.to_bits() as u64,
        config.hs_out_of_range_policy as u64,
    ] {
        hash.value(value);
    }
    hash.finish()
}

impl Trainer {
    pub fn create(
        corpus: Arc<Corpus>,
        vocab: Arc<Vocabulary>,
        model: Arc<Model>,
        config: &TrainingConfig,
    ) -> Result<Self, Status> {
        Self::create_with_digest(corpus, vocab, model, config, None)
    }

    pub fn create_with_digest(
        corpus: Arc<Corpus>,
        vocab: Arc<Vocabulary>,
        model: Arc<Model>,
        config: &TrainingConfig,
        corpus_digest: Option<String>,
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
            NegativeSampler::initialize(&vocab, config.negative_table_size)?
        } else {
            NegativeSampler::default()
        };
        let corpus_digest = match corpus_digest {
            Some(digest) => digest,
            None => corpus.digest()?,
        };
        Ok(Self {
            corpus,
            corpus_digest,
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

    pub fn state_descriptor(&self) -> Result<StateDescriptor, Status> {
        Ok(StateDescriptor {
            schema_version: TRAINING_STATE_SCHEMA_VERSION,
            config_digest: config_digest(&self.config),
            vocabulary_digest: self.vocab.digest(),
            corpus_digest: self.corpus_digest.clone(),
        })
    }

    pub fn session(self: &Arc<Self>) -> Result<TrainingSession, Status> {
        TrainingSession::create(Arc::clone(self))
    }

    pub fn train(self: &Arc<Self>) -> Result<(), Status> {
        let mut session = self.session()?;
        while !session.is_complete() {
            session.train_epoch()?;
        }
        Ok(())
    }
}

impl TrainingSession {
    pub fn create(trainer: Arc<Trainer>) -> Result<Self, Status> {
        trainer.processed_tokens.store(0, Ordering::Relaxed);
        let mut workers = Vec::new();
        workers
            .try_reserve_exact(trainer.config.thread_count)
            .map_err(|_| Status::OutOfMemory)?;
        for worker_id in 0..trainer.config.thread_count {
            workers.push(Worker::initialize(&trainer, worker_id)?);
        }
        Ok(Self {
            trainer,
            workers,
            completed_epochs: 0,
            running: false,
        })
    }

    pub fn restore(trainer: Arc<Trainer>, state: &TrainingState) -> Result<Self, Status> {
        if state.descriptor.schema_version != TRAINING_STATE_SCHEMA_VERSION {
            return Err(Status::SchemaMismatch);
        }
        if state.descriptor != trainer.state_descriptor()? {
            return Err(Status::IdentityMismatch);
        }
        if state.completed_epochs > trainer.config.epochs
            || state.workers.len() != trainer.config.thread_count
            || Vocabulary::restore(&state.vocabulary)?.digest() != trainer.vocab.digest()
        {
            return Err(Status::InvalidState);
        }
        if trainer
            .model
            .restore_embeddings(&state.input_embeddings, &state.output_embeddings)
            != Status::Ok
        {
            return Err(Status::InvalidState);
        }
        let mut session = Self::create(Arc::clone(&trainer))?;
        session.completed_epochs = state.completed_epochs;
        trainer
            .processed_tokens
            .store(state.processed_tokens, Ordering::Relaxed);
        for (worker, worker_state) in session.workers.iter_mut().zip(&state.workers) {
            let status = worker.restore_state(worker_state);
            if status != Status::Ok {
                return Err(status);
            }
        }
        Ok(session)
    }

    pub const fn completed_epochs(&self) -> usize {
        self.completed_epochs
    }

    pub fn is_complete(&self) -> bool {
        self.completed_epochs >= self.trainer.config.epochs
    }

    pub fn train_epoch(&mut self) -> Result<EpochReport, Status> {
        if self.running || self.is_complete() {
            return Err(Status::InvalidState);
        }
        if self.trainer.model.begin_training() != Status::Ok {
            return Err(Status::InvalidState);
        }
        let _guard = TrainingGuard(&self.trainer.model);
        self.running = true;
        let started = Instant::now();
        let before = self.trainer.processed_tokens();
        let reset = self.completed_epochs > 0;
        let result = if self.workers.len() == 1 {
            self.workers[0].run_epoch(&self.trainer, reset, started)
        } else {
            thread::scope(|scope| {
                let mut handles = Vec::new();
                handles
                    .try_reserve_exact(self.workers.len())
                    .map_err(|_| Status::OutOfMemory)?;
                for worker in &mut self.workers {
                    let trainer = Arc::clone(&self.trainer);
                    handles.push(scope.spawn(move || worker.run_epoch(&trainer, reset, started)));
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
        };
        self.running = false;
        result?;
        self.completed_epochs += 1;
        let processed = self.trainer.processed_tokens();
        let epoch_tokens = processed.saturating_sub(before);
        let elapsed_seconds = started.elapsed().as_secs_f64();
        let learning_rate = self
            .workers
            .iter()
            .map(|worker| worker.learning_rate)
            .fold(self.trainer.config.initial_learning_rate, Real::min);
        if processed < before || !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(Status::InvalidState);
        }
        let mut observations: Vec<Observation> = self
            .workers
            .iter()
            .flat_map(|worker| {
                worker.observations.iter().map(|item| Observation {
                    epoch: self.completed_epochs,
                    processed_tokens: item.processed_tokens,
                    learning_rate: item.learning_rate,
                    objective_loss_sum: item.objective_loss_sum,
                    objective_loss_count: item.objective_loss_count,
                    elapsed_seconds: item.elapsed_seconds,
                    tokens_per_second: if item.elapsed_seconds > 0.0 {
                        item.processed_tokens.saturating_sub(before) as f64 / item.elapsed_seconds
                    } else {
                        0.0
                    },
                })
            })
            .collect();
        observations.sort_by_key(|item| item.processed_tokens);
        let objective_loss_sum = observations
            .iter()
            .map(|item| item.objective_loss_sum)
            .sum();
        let objective_loss_count = observations
            .iter()
            .map(|item| item.objective_loss_count)
            .sum();
        Ok(EpochReport {
            epoch: self.completed_epochs,
            epoch_tokens,
            processed_tokens: processed,
            learning_rate,
            elapsed_seconds,
            tokens_per_second: if elapsed_seconds > 0.0 {
                epoch_tokens as f64 / elapsed_seconds
            } else {
                0.0
            },
            objective_loss_sum,
            objective_loss_count,
            observations,
        })
    }

    pub fn export_state(&self) -> Result<TrainingState, Status> {
        if self.running {
            return Err(Status::InvalidState);
        }
        let count = self.trainer.model.vocab_size * self.trainer.model.embedding_dimension;
        let mut input_embeddings = vec![0.0; count];
        let mut output_embeddings = vec![0.0; count];
        if self
            .trainer
            .model
            .snapshot_into(EmbeddingKind::Input, &mut input_embeddings)
            != Status::Ok
            || self
                .trainer
                .model
                .snapshot_into(EmbeddingKind::Output, &mut output_embeddings)
                != Status::Ok
        {
            return Err(Status::InvalidState);
        }
        Ok(TrainingState {
            descriptor: self.trainer.state_descriptor()?,
            completed_epochs: self.completed_epochs,
            processed_tokens: self.trainer.processed_tokens(),
            vocabulary: self.trainer.vocab.export_state(),
            input_embeddings,
            output_embeddings,
            workers: self.workers.iter().map(Worker::export_state).collect(),
        })
    }
}

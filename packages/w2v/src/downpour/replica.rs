use super::config::DownpourConfig;
use super::gradient::{GradientBatch, GradientRowHeader, ParameterKind, shard_index};
use crate::{
    atomic_float,
    config::{MAX_SENTENCE_LENGTH, ModelKind, Real, Status},
    corpus::Tokenizer,
    model::Model,
    random::{Rng, RngPurpose, derive_seed},
    simd,
    trainer::Trainer,
    training::{
        ModelStep, ObjectiveLoss, ParameterBackend, cbow_train_with_backend,
        skip_gram_train_with_backend, worker::WorkerObservation,
    },
};
use std::{
    collections::HashMap,
    fs::File,
    io::BufReader,
    sync::{Arc, atomic::Ordering, mpsc::SyncSender},
    time::Instant,
};

#[derive(Clone, Debug, PartialEq)]
pub struct ReplicaState {
    pub replica_id: usize,
    pub local_token_count: u64,
    pub window_rng_state: u64,
    pub subsampling_rng_state: u64,
    pub negative_rng_state: u64,
    pub objective_count: u64,
}

pub struct ReplicaBackend {
    pub embedding_dimension: usize,
    pub model: Arc<Model>,
    pub cache_map: HashMap<(ParameterKind, usize), usize>,
    pub cache_arena: Vec<Real>,
    pub grad_map: HashMap<(ParameterKind, usize), usize>,
    pub grad_arena: Vec<Real>,
    pub grad_headers: Vec<GradientRowHeader>,
}

impl ReplicaBackend {
    pub fn new(embedding_dimension: usize, model: Arc<Model>) -> Self {
        Self {
            embedding_dimension,
            model,
            cache_map: HashMap::new(),
            cache_arena: Vec::new(),
            grad_map: HashMap::new(),
            grad_arena: Vec::new(),
            grad_headers: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        self.cache_map.clear();
        self.cache_arena.clear();
        self.grad_map.clear();
        self.grad_arena.clear();
        self.grad_headers.clear();
    }
}

impl ParameterBackend for ReplicaBackend {
    #[inline(always)]
    fn embedding_dimension(&self) -> usize {
        self.embedding_dimension
    }

    #[inline(always)]
    fn load_input_row(&mut self, token: usize, destination: &mut [Real]) {
        let dim = self.embedding_dimension;
        if let Some(&offset) = self.cache_map.get(&(ParameterKind::Input, token)) {
            destination.copy_from_slice(&self.cache_arena[offset..offset + dim]);
        } else {
            let offset = self.cache_arena.len();
            let start = token * dim;
            let row = &self.model.input_embeddings[start..start + dim];
            for (dst, src) in destination.iter_mut().zip(row) {
                *dst = atomic_float::load(src);
            }
            self.cache_arena.extend_from_slice(destination);
            self.cache_map.insert((ParameterKind::Input, token), offset);
        }
    }

    #[inline(always)]
    fn accumulate_input_row(&mut self, token: usize, destination: &mut [Real]) {
        let dim = self.embedding_dimension;
        if let Some(&offset) = self.cache_map.get(&(ParameterKind::Input, token)) {
            for (dst, src) in destination
                .iter_mut()
                .zip(&self.cache_arena[offset..offset + dim])
            {
                *dst += *src;
            }
        } else {
            let offset = self.cache_arena.len();
            let start = token * dim;
            let row = &self.model.input_embeddings[start..start + dim];
            self.cache_arena.reserve(dim);
            for src in row {
                let val = atomic_float::load(src);
                self.cache_arena.push(val);
            }
            for (dst, src) in destination
                .iter_mut()
                .zip(&self.cache_arena[offset..offset + dim])
            {
                *dst += *src;
            }
            self.cache_map.insert((ParameterKind::Input, token), offset);
        }
    }

    #[inline(always)]
    fn load_output_row(&mut self, token: usize, destination: &mut [Real]) {
        let dim = self.embedding_dimension;
        if let Some(&offset) = self.cache_map.get(&(ParameterKind::Output, token)) {
            destination.copy_from_slice(&self.cache_arena[offset..offset + dim]);
        } else {
            let offset = self.cache_arena.len();
            let start = token * dim;
            let row = &self.model.output_embeddings[start..start + dim];
            for (dst, src) in destination.iter_mut().zip(row) {
                *dst = atomic_float::load(src);
            }
            self.cache_arena.extend_from_slice(destination);
            self.cache_map
                .insert((ParameterKind::Output, token), offset);
        }
    }

    #[inline(always)]
    fn apply_output_gradient(
        &mut self,
        token: usize,
        hidden: &[Real],
        output_snapshot: &mut [Real],
        raw_error: Real,
        _learning_rate: Real,
        hidden_gradient: &mut [Real],
    ) {
        let dim = self.embedding_dimension;
        simd::scaled_accumulate(hidden_gradient, output_snapshot, raw_error);
        if let Some(&offset) = self.grad_map.get(&(ParameterKind::Output, token)) {
            simd::scaled_accumulate(
                &mut self.grad_arena[offset..offset + dim],
                hidden,
                raw_error,
            );
        } else {
            let offset = self.grad_arena.len();
            self.grad_arena.resize(offset + dim, 0.0);
            simd::scaled_accumulate(
                &mut self.grad_arena[offset..offset + dim],
                hidden,
                raw_error,
            );
            self.grad_headers.push(GradientRowHeader {
                kind: ParameterKind::Output,
                row_index: token,
            });
            self.grad_map.insert((ParameterKind::Output, token), offset);
        }
    }

    #[inline(always)]
    fn apply_input_gradient(
        &mut self,
        token: usize,
        hidden_gradient: &[Real],
        _learning_rate: Real,
    ) {
        let dim = self.embedding_dimension;
        if let Some(&offset) = self.grad_map.get(&(ParameterKind::Input, token)) {
            for (dst, src) in self.grad_arena[offset..offset + dim]
                .iter_mut()
                .zip(hidden_gradient)
            {
                *dst += *src;
            }
        } else {
            let offset = self.grad_arena.len();
            self.grad_arena.extend_from_slice(hidden_gradient);
            self.grad_headers.push(GradientRowHeader {
                kind: ParameterKind::Input,
                row_index: token,
            });
            self.grad_map.insert((ParameterKind::Input, token), offset);
        }
    }
}

pub struct Replica {
    pub replica_id: usize,
    pub shard_start: usize,
    pub sentence: Vec<usize>,
    pub hidden: Vec<Real>,
    pub hidden_gradient: Vec<Real>,
    pub output_snapshot: Vec<Real>,
    pub local_token_count: u64,
    pub epoch_token_count: u64,
    pub window_rng: Rng,
    pub subsampling_rng: Rng,
    pub negative_rng: Rng,
    pub objective_count: u64,
    pub observations: Vec<WorkerObservation>,
    pub backend: ReplicaBackend,
    pub targets_in_batch: usize,
    pub mini_batch_targets: usize,
    pub batch_id: u64,
    pub adagrad_gamma: Real,
    tokenizer: Tokenizer<BufReader<File>>,
}

impl Replica {
    pub fn initialize(
        trainer: &Trainer,
        downpour_config: &DownpourConfig,
        replica_id: usize,
    ) -> Result<Self, Status> {
        if replica_id >= trainer.config.thread_count {
            return Err(Status::InvalidArgument);
        }
        let shard_start = trainer.corpus.byte_size / trainer.config.thread_count * replica_id;
        let dimension = trainer.config.embedding_dimension;
        let mut sentence = Vec::new();
        sentence
            .try_reserve_exact(MAX_SENTENCE_LENGTH)
            .map_err(|_| Status::OutOfMemory)?;
        let mut hidden = Vec::new();
        let mut hidden_gradient = Vec::new();
        let mut output_snapshot = Vec::new();
        hidden
            .try_reserve_exact(dimension)
            .map_err(|_| Status::OutOfMemory)?;
        hidden_gradient
            .try_reserve_exact(dimension)
            .map_err(|_| Status::OutOfMemory)?;
        output_snapshot
            .try_reserve_exact(dimension)
            .map_err(|_| Status::OutOfMemory)?;
        hidden.resize(dimension, 0.0);
        hidden_gradient.resize(dimension, 0.0);
        output_snapshot.resize(dimension, 0.0);
        let rng = |purpose| {
            Rng::new(
                derive_seed(trainer.config.root_seed, replica_id, purpose),
                trainer.config.rng_algorithm,
            )
        };
        Ok(Self {
            replica_id,
            shard_start,
            sentence,
            hidden,
            hidden_gradient,
            output_snapshot,
            local_token_count: 0,
            epoch_token_count: 0,
            window_rng: rng(RngPurpose::Window),
            subsampling_rng: rng(RngPurpose::Subsample),
            negative_rng: rng(RngPurpose::Negative),
            objective_count: 0,
            observations: Vec::new(),
            backend: ReplicaBackend::new(dimension, Arc::clone(&trainer.model)),
            targets_in_batch: 0,
            mini_batch_targets: downpour_config.mini_batch_targets,
            batch_id: 0,
            adagrad_gamma: downpour_config.adagrad_gamma,
            tokenizer: trainer.corpus.tokenizer(shard_start)?,
        })
    }

    pub fn reset_epoch(&mut self, trainer: &Trainer) -> Result<(), Status> {
        self.epoch_token_count = 0;
        self.tokenizer = trainer.corpus.tokenizer(self.shard_start)?;
        self.backend.clear();
        self.targets_in_batch = 0;
        Ok(())
    }

    pub fn export_state(&self) -> ReplicaState {
        ReplicaState {
            replica_id: self.replica_id,
            local_token_count: self.local_token_count,
            window_rng_state: self.window_rng.state,
            subsampling_rng_state: self.subsampling_rng.state,
            negative_rng_state: self.negative_rng.state,
            objective_count: self.objective_count,
        }
    }

    pub fn restore_state(&mut self, state: &ReplicaState) -> Status {
        if state.replica_id != self.replica_id {
            return Status::InvalidState;
        }
        self.local_token_count = state.local_token_count;
        self.window_rng.state = state.window_rng_state;
        self.subsampling_rng.state = state.subsampling_rng_state;
        self.negative_rng.state = state.negative_rng_state;
        self.objective_count = state.objective_count;
        self.backend.clear();
        self.targets_in_batch = 0;
        Status::Ok
    }

    pub fn fill_sentence(&mut self, trainer: &Trainer) -> Result<bool, Status> {
        self.sentence.clear();
        let subsampling_threshold = trainer.config.subsampling_threshold;
        let retained_token_count = trainer.vocab.retained_token_count;
        let mut finished = false;
        while self.sentence.len() < MAX_SENTENCE_LENGTH {
            let read = self.tokenizer.read_token()?;
            if read.at_eof {
                finished = true;
                break;
            }
            let Some(token) = trainer.vocab.find(&read.token) else {
                continue;
            };
            self.local_token_count = self.local_token_count.wrapping_add(1);
            self.epoch_token_count = self.epoch_token_count.wrapping_add(1);
            if token == 0 {
                break;
            }
            if subsampling_threshold > 0.0 {
                let sample = subsampling_threshold;
                let count = trainer.vocab.entries[token].count;
                let keep_probability =
                    ((count as Real / (sample * retained_token_count as Real)).sqrt() + 1.0)
                        * (sample * retained_token_count as Real)
                        / count as Real;
                if keep_probability < self.subsampling_rng.uniform() {
                    continue;
                }
            }
            self.sentence.push(token);
        }
        Ok(finished)
    }

    pub fn push_gradient_batch(
        &mut self,
        ps_senders: &[SyncSender<GradientBatch>],
    ) -> Result<(), Status> {
        if self.backend.grad_headers.is_empty() {
            self.backend.clear();
            self.targets_in_batch = 0;
            return Ok(());
        }
        let shard_count = ps_senders.len();
        let dim = self.backend.embedding_dimension;
        for (shard_id, sender) in ps_senders.iter().enumerate() {
            let mut batch = GradientBatch::new(self.replica_id, self.batch_id);
            for header in &self.backend.grad_headers {
                if shard_index(header.kind, header.row_index, shard_count) == shard_id {
                    let offset = self.backend.grad_map[&(header.kind, header.row_index)];
                    batch.headers.push(header.clone());
                    batch
                        .gradients
                        .extend_from_slice(&self.backend.grad_arena[offset..offset + dim]);
                }
            }
            if !batch.is_empty() {
                sender.send(batch).map_err(|_| Status::ThreadError)?;
            }
        }
        self.batch_id = self.batch_id.wrapping_add(1);
        self.backend.clear();
        self.targets_in_batch = 0;
        Ok(())
    }

    pub fn train_sentence(
        &mut self,
        trainer: &Trainer,
        started: Instant,
        ps_senders: &[SyncSender<GradientBatch>],
    ) -> Result<(), Status> {
        for position in 0..self.sentence.len() {
            self.objective_count += 1;
            self.targets_in_batch += 1;
            let observe = trainer.config.observation_interval > 0
                && self
                    .objective_count
                    .is_multiple_of(trainer.config.observation_interval as u64);
            let mut step = ModelStep {
                trainer,
                target_token: self.sentence[position],
                learning_rate: self.adagrad_gamma,
                sentence: &self.sentence,
                sentence_position: position,
                hidden: &mut self.hidden,
                hidden_gradient: &mut self.hidden_gradient,
                output_snapshot: &mut self.output_snapshot,
                window_rng: &mut self.window_rng,
                negative_rng: &mut self.negative_rng,
                observe_objective: observe,
            };
            let loss: ObjectiveLoss = match trainer.config.model_kind {
                ModelKind::Cbow => cbow_train_with_backend(&mut step, &mut self.backend),
                ModelKind::SkipGram => skip_gram_train_with_backend(&mut step, &mut self.backend),
            }?;
            if observe && loss.count > 0 {
                self.observations.push(WorkerObservation {
                    processed_tokens: trainer.processed_tokens(),
                    learning_rate: self.adagrad_gamma,
                    objective_loss_sum: loss.sum,
                    objective_loss_count: loss.count,
                    elapsed_seconds: started.elapsed().as_secs_f64(),
                });
            }
            if self.targets_in_batch >= self.mini_batch_targets {
                self.push_gradient_batch(ps_senders)?;
            }
        }
        Ok(())
    }

    pub fn run_epoch(
        &mut self,
        trainer: &Trainer,
        reset: bool,
        started: Instant,
        ps_senders: &[SyncSender<GradientBatch>],
    ) -> Result<(), Status> {
        self.observations.clear();
        if reset {
            self.reset_epoch(trainer)?;
        }
        let before_tokens = self.local_token_count;
        let mut finished = false;
        while !finished {
            finished = self.fill_sentence(trainer)?;
            let limit = trainer.vocab.retained_token_count / trainer.config.thread_count as u64;
            if self.epoch_token_count > limit {
                finished = true;
            }
            if !finished {
                self.train_sentence(trainer, started, ps_senders)?;
            }
        }
        if self.targets_in_batch > 0 {
            self.push_gradient_batch(ps_senders)?;
        }
        let epoch_tokens = self.local_token_count.saturating_sub(before_tokens);
        trainer
            .processed_tokens
            .fetch_add(epoch_tokens, Ordering::Relaxed);
        Ok(())
    }
}

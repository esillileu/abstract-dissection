//! Sentence filling, subsampling, and worker-local rate from the C oracle.
use super::{ModelStep, cbow_train, skip_gram_train};
use crate::{
    config::{MAX_SENTENCE_LENGTH, ModelKind, Real, Status},
    corpus::Tokenizer,
    random::{Rng, RngPurpose, derive_seed},
    trainer::Trainer,
};
use std::{fs::File, sync::atomic::Ordering};

pub struct Worker {
    pub worker_id: usize,
    pub shard_start: usize,
    pub sentence: Vec<usize>,
    pub hidden: Vec<Real>,
    pub hidden_gradient: Vec<Real>,
    pub local_token_count: u64,
    pub epoch_token_count: u64,
    pub last_learning_rate_update_count: u64,
    pub learning_rate: Real,
    pub window_rng: Rng,
    pub subsampling_rng: Rng,
    pub negative_rng: Rng,
    tokenizer: Tokenizer<File>,
}

impl Worker {
    pub fn initialize(trainer: &Trainer<'_>, worker_id: usize) -> Result<Self, Status> {
        if worker_id >= trainer.config.thread_count {
            return Err(Status::InvalidArgument);
        }
        let shard_start = trainer.corpus.byte_size / trainer.config.thread_count * worker_id;
        let dimension = trainer.config.embedding_dimension;
        let mut sentence = Vec::new();
        sentence
            .try_reserve_exact(MAX_SENTENCE_LENGTH)
            .map_err(|_| Status::OutOfMemory)?;
        let mut hidden = Vec::new();
        let mut hidden_gradient = Vec::new();
        hidden
            .try_reserve_exact(dimension)
            .map_err(|_| Status::OutOfMemory)?;
        hidden_gradient
            .try_reserve_exact(dimension)
            .map_err(|_| Status::OutOfMemory)?;
        hidden.resize(dimension, 0.0);
        hidden_gradient.resize(dimension, 0.0);
        let rng = |purpose| {
            Rng::new(
                derive_seed(trainer.config.root_seed, worker_id, purpose),
                trainer.config.rng_algorithm,
            )
        };
        Ok(Self {
            worker_id,
            shard_start,
            sentence,
            hidden,
            hidden_gradient,
            local_token_count: 0,
            epoch_token_count: 0,
            last_learning_rate_update_count: 0,
            learning_rate: trainer.config.initial_learning_rate,
            window_rng: rng(RngPurpose::Window),
            subsampling_rng: rng(RngPurpose::Subsample),
            negative_rng: rng(RngPurpose::Negative),
            tokenizer: trainer.corpus.tokenizer(shard_start)?,
        })
    }

    pub fn reset_epoch(&mut self, trainer: &Trainer<'_>) -> Result<(), Status> {
        self.epoch_token_count = 0;
        self.tokenizer = trainer.corpus.tokenizer(self.shard_start)?;
        Ok(())
    }

    pub fn fill_sentence(&mut self, trainer: &Trainer<'_>) -> Result<bool, Status> {
        self.sentence.clear();
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
            trainer.processed_tokens.fetch_add(1, Ordering::Relaxed);
            if token == 0 {
                break;
            }
            if trainer.config.subsampling_threshold > 0.0 {
                let sample = trainer.config.subsampling_threshold;
                let count = trainer.vocab.entries[token].count;
                let train_words = trainer.vocab.retained_token_count;
                let keep_probability = ((count as Real / (sample * train_words as Real)).sqrt()
                    + 1.0)
                    * (sample * train_words as Real)
                    / count as Real;
                if keep_probability < self.subsampling_rng.uniform() {
                    continue;
                }
            }
            self.sentence.push(token);
        }
        Ok(finished)
    }

    pub fn update_learning_rate(&mut self, trainer: &Trainer<'_>) {
        let since_update = self.local_token_count - self.last_learning_rate_update_count;
        if since_update <= trainer.config.learning_rate_update_interval as u64 {
            return;
        }
        let processed = trainer.processed_tokens.load(Ordering::Relaxed);
        let total = trainer.vocab.retained_token_count as f64 * trainer.config.epochs as f64;
        let rate =
            trainer.config.initial_learning_rate * (1.0 - processed as f64 / (total + 1.0)) as Real;
        let minimum = trainer.config.initial_learning_rate * 0.0001;
        self.learning_rate = rate.max(minimum);
        self.last_learning_rate_update_count = self.local_token_count;
    }

    pub fn train_sentence(&mut self, trainer: &Trainer<'_>) {
        for position in 0..self.sentence.len() {
            self.update_learning_rate(trainer);
            let mut step = ModelStep {
                trainer,
                target_token: self.sentence[position],
                learning_rate: self.learning_rate,
                sentence: &self.sentence,
                sentence_position: position,
                hidden: &mut self.hidden,
                hidden_gradient: &mut self.hidden_gradient,
                window_rng: &mut self.window_rng,
                negative_rng: &mut self.negative_rng,
            };
            match trainer.config.model_kind {
                ModelKind::Cbow => cbow_train(&mut step),
                ModelKind::SkipGram => skip_gram_train(&mut step),
            }
        }
    }

    pub fn run(&mut self, trainer: &Trainer<'_>) -> Result<(), Status> {
        for epoch in 0..trainer.config.epochs {
            if epoch > 0 {
                self.reset_epoch(trainer)?;
            }
            let mut finished = false;
            while !finished {
                finished = self.fill_sentence(trainer)?;
                let limit = trainer.vocab.retained_token_count / trainer.config.thread_count as u64;
                if self.epoch_token_count > limit {
                    finished = true;
                }
                if !finished {
                    self.train_sentence(trainer);
                }
            }
        }
        Ok(())
    }
}

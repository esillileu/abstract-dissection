use super::{
    NnlmModel, NnlmTrainingConfig,
    training::{apply_sgd, forward_backward},
};
use crate::{
    Corpus, EpochReport, Observation, StateDescriptor, Status, Vocabulary, VocabularyState,
};
use std::{collections::VecDeque, sync::Arc, time::Instant};

#[derive(Clone, Debug, PartialEq)]
pub struct NnlmTrainingState {
    pub descriptor: StateDescriptor,
    pub vocabulary: VocabularyState,
    pub model: NnlmModel,
    pub completed_epochs: usize,
    pub processed_tokens: u64,
}

/// Single-thread SGD mathematical baseline. Canonical DistBelief training is a
/// separate subsystem; this session never silently ignores a thread request.
pub struct NnlmTrainingSession {
    pub corpus: Arc<Corpus>,
    pub vocabulary: Arc<Vocabulary>,
    pub config: NnlmTrainingConfig,
    pub state: NnlmTrainingState,
}

impl NnlmTrainingSession {
    pub fn create(
        corpus: Arc<Corpus>,
        vocabulary: Arc<Vocabulary>,
        config: NnlmTrainingConfig,
        corpus_digest: Option<String>,
    ) -> Result<Self, Status> {
        if config.thread_count != 1 {
            return Err(Status::InvalidArgument);
        }
        let model = NnlmModel::create(&vocabulary, &config)?;
        let descriptor = StateDescriptor {
            schema_version: 1,
            config_digest: config.digest(),
            vocabulary_digest: vocabulary.digest(),
            corpus_digest: corpus_digest.map_or_else(|| corpus.digest(), Ok)?,
        };
        Ok(Self {
            corpus,
            vocabulary: Arc::clone(&vocabulary),
            config,
            state: NnlmTrainingState {
                descriptor,
                vocabulary: vocabulary.export_state(),
                model,
                completed_epochs: 0,
                processed_tokens: 0,
            },
        })
    }

    pub fn is_complete(&self) -> bool {
        self.state.completed_epochs >= self.config.epochs
    }

    pub fn restore(
        corpus: Arc<Corpus>,
        vocabulary: Arc<Vocabulary>,
        config: NnlmTrainingConfig,
        state: &NnlmTrainingState,
        corpus_digest: Option<String>,
    ) -> Result<Self, Status> {
        if state.descriptor.schema_version != 1 {
            return Err(Status::SchemaMismatch);
        }
        if state.descriptor.config_digest != config.digest()
            || state.descriptor.vocabulary_digest != vocabulary.digest()
            || state.descriptor.corpus_digest
                != corpus_digest.map_or_else(|| corpus.digest(), Ok)?
            || Vocabulary::restore(&state.vocabulary)?.digest() != vocabulary.digest()
        {
            return Err(Status::IdentityMismatch);
        }
        if config.thread_count != 1
            || state.completed_epochs > config.epochs
            || !state.model.validate(&vocabulary, &config)
        {
            return Err(Status::InvalidState);
        }
        Ok(Self {
            corpus,
            vocabulary,
            config,
            state: state.clone(),
        })
    }

    pub fn train_epoch(&mut self) -> Result<EpochReport, Status> {
        if self.is_complete() {
            return Err(Status::InvalidState);
        }
        let start = Instant::now();
        let before = self.state.processed_tokens;
        let mut tokenizer = self.corpus.tokenizer(0)?;
        let mut history = VecDeque::with_capacity(self.config.history_length);
        let mut loss = 0.0;
        let mut targets = 0;
        let mut observations = Vec::new();
        loop {
            let read = tokenizer.read_token()?;
            // The tokenizer may return a final lexical token together with EOF.
            if !read.token.is_empty() {
                match self.vocabulary.find(&read.token) {
                    Some(0) | None => history.clear(),
                    Some(id) => {
                        self.state.processed_tokens += 1;
                        if history.len() == self.config.history_length {
                            let input: Vec<_> = history.iter().copied().collect();
                            let gradient = forward_backward(
                                &self.state.model,
                                &self.vocabulary,
                                self.config.hidden_activation,
                                &input,
                                id,
                            )?;
                            loss += gradient.loss;
                            targets += 1;
                            apply_sgd(
                                &mut self.state.model,
                                &gradient,
                                self.config.initial_learning_rate,
                            );
                            history.pop_front();
                            if self.config.observation_interval > 0
                                && targets % self.config.observation_interval as u64 == 0
                            {
                                let elapsed = start.elapsed().as_secs_f64();
                                observations.push(Observation {
                                    epoch: self.state.completed_epochs + 1,
                                    processed_tokens: self.state.processed_tokens,
                                    learning_rate: self.config.initial_learning_rate,
                                    objective_loss_sum: loss,
                                    objective_loss_count: targets,
                                    elapsed_seconds: elapsed,
                                    tokens_per_second: (self.state.processed_tokens - before)
                                        as f64
                                        / elapsed,
                                });
                            }
                        }
                        history.push_back(id);
                    }
                }
            }
            if read.at_eof {
                break;
            }
        }
        self.state.completed_epochs += 1;
        let elapsed = start.elapsed().as_secs_f64();
        Ok(EpochReport {
            epoch: self.state.completed_epochs,
            epoch_tokens: self.state.processed_tokens - before,
            processed_tokens: self.state.processed_tokens,
            learning_rate: self.config.initial_learning_rate,
            elapsed_seconds: elapsed,
            tokens_per_second: (self.state.processed_tokens - before) as f64 / elapsed,
            objective_loss_sum: loss,
            objective_loss_count: targets,
            observations,
        })
    }
}

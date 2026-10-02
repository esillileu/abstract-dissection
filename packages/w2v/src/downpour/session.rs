use super::{
    config::DownpourConfig,
    gradient::{GradientBatch, ParameterKind},
    parameter_server::ParameterServerShard,
    replica::Replica,
    state::{
        DOWNPOUR_TRAINING_STATE_SCHEMA_VERSION, DownpourStateDescriptor, DownpourTrainingState,
        downpour_config_digest,
    },
};
use crate::{
    config::Status,
    model::{EmbeddingKind, Model},
    trainer::{EpochReport, Observation, Trainer},
    vocab::Vocabulary,
};
use std::{
    sync::{
        Arc,
        atomic::Ordering,
        mpsc::{SyncSender, sync_channel},
    },
    thread,
    time::Instant,
};

struct TrainingGuard<'a>(&'a Model);

impl Drop for TrainingGuard<'_> {
    fn drop(&mut self) {
        self.0.end_training();
    }
}

pub struct DownpourTrainingSession {
    trainer: Arc<Trainer>,
    downpour_config: DownpourConfig,
    replicas: Vec<Replica>,
    ps_shards: Vec<ParameterServerShard>,
    completed_epochs: usize,
    running: bool,
}

impl DownpourTrainingSession {
    pub fn create(trainer: Arc<Trainer>, downpour_config: DownpourConfig) -> Result<Self, Status> {
        if downpour_config.validate() != Status::Ok {
            return Err(Status::InvalidArgument);
        }
        trainer.processed_tokens.store(0, Ordering::Relaxed);
        let thread_count = trainer.config.thread_count;
        let mut replicas = Vec::new();
        replicas
            .try_reserve_exact(thread_count)
            .map_err(|_| Status::OutOfMemory)?;
        for replica_id in 0..thread_count {
            replicas.push(Replica::initialize(&trainer, &downpour_config, replica_id)?);
        }
        let shard_count = downpour_config.parameter_server_shards;
        let mut ps_shards = Vec::new();
        ps_shards
            .try_reserve_exact(shard_count)
            .map_err(|_| Status::OutOfMemory)?;
        for shard_id in 0..shard_count {
            ps_shards.push(ParameterServerShard::initialize(
                shard_id,
                shard_count,
                trainer.model.vocab_size,
                trainer.model.embedding_dimension,
                downpour_config.adagrad_gamma,
                downpour_config.adagrad_epsilon,
                Arc::clone(&trainer.model),
            )?);
        }
        Ok(Self {
            trainer,
            downpour_config,
            replicas,
            ps_shards,
            completed_epochs: 0,
            running: false,
        })
    }

    pub fn restore(
        trainer: Arc<Trainer>,
        downpour_config: DownpourConfig,
        state: &DownpourTrainingState,
    ) -> Result<Self, Status> {
        if state.descriptor.schema_version != DOWNPOUR_TRAINING_STATE_SCHEMA_VERSION {
            return Err(Status::SchemaMismatch);
        }
        let expected_descriptor = DownpourStateDescriptor {
            schema_version: DOWNPOUR_TRAINING_STATE_SCHEMA_VERSION,
            config_digest: downpour_config_digest(&trainer.config, &downpour_config),
            vocabulary_digest: trainer.vocab.digest(),
            corpus_digest: trainer.corpus_digest.clone(),
        };
        if state.descriptor != expected_descriptor {
            return Err(Status::IdentityMismatch);
        }
        if state.completed_epochs > trainer.config.epochs
            || state.replicas.len() != trainer.config.thread_count
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
        let mut session = Self::create(Arc::clone(&trainer), downpour_config)?;
        session.completed_epochs = state.completed_epochs;
        trainer
            .processed_tokens
            .store(state.processed_tokens, Ordering::Relaxed);
        for (replica, replica_state) in session.replicas.iter_mut().zip(&state.replicas) {
            let status = replica.restore_state(replica_state);
            if status != Status::Ok {
                return Err(status);
            }
        }
        for shard in &mut session.ps_shards {
            shard.restore_adagrad(ParameterKind::Input, &state.input_adagrad);
            shard.restore_adagrad(ParameterKind::Output, &state.output_adagrad);
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
        let queue_capacity = self.downpour_config.queue_capacity;

        let mut senders: Vec<SyncSender<GradientBatch>> = Vec::new();
        let mut receivers = Vec::new();
        for _ in 0..self.ps_shards.len() {
            let (tx, rx) = sync_channel::<GradientBatch>(queue_capacity);
            senders.push(tx);
            receivers.push(rx);
        }

        let result = thread::scope(|scope| {
            // Spawn PS shard worker threads
            let mut ps_handles = Vec::new();
            for (shard, rx) in self.ps_shards.iter_mut().zip(receivers) {
                ps_handles.push(scope.spawn(move || {
                    while let Ok(batch) = rx.recv() {
                        shard.apply_batch(&batch);
                    }
                }));
            }

            // Spawn Replica worker threads
            let mut replica_handles = Vec::new();
            for replica in &mut self.replicas {
                let trainer = Arc::clone(&self.trainer);
                let replica_senders = senders.clone();
                replica_handles.push(
                    scope.spawn(move || {
                        replica.run_epoch(&trainer, reset, started, &replica_senders)
                    }),
                );
            }

            // Drop main thread's senders so PS shards know when all replicas have finished
            drop(senders);

            let mut replica_result = Ok(());
            for handle in replica_handles {
                match handle.join() {
                    Ok(Ok(())) => {}
                    Ok(Err(status)) if replica_result.is_ok() => replica_result = Err(status),
                    Err(_) if replica_result.is_ok() => replica_result = Err(Status::ThreadError),
                    _ => {}
                }
            }

            // Join PS threads
            for handle in ps_handles {
                if handle.join().is_err() && replica_result.is_ok() {
                    replica_result = Err(Status::ThreadError);
                }
            }

            replica_result
        });

        self.running = false;
        result?;
        self.completed_epochs += 1;
        let processed = self.trainer.processed_tokens();
        let epoch_tokens = processed.saturating_sub(before);
        let elapsed_seconds = started.elapsed().as_secs_f64();
        let learning_rate = self.downpour_config.adagrad_gamma;

        let mut observations: Vec<Observation> = self
            .replicas
            .iter()
            .flat_map(|replica| {
                replica.observations.iter().map(|item| Observation {
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

    pub fn export_state(&self) -> Result<DownpourTrainingState, Status> {
        if self.running {
            return Err(Status::InvalidState);
        }
        let vocab_size = self.trainer.model.vocab_size;
        let dim = self.trainer.model.embedding_dimension;
        let count = vocab_size * dim;
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
        let mut input_adagrad = vec![0.0; count];
        let mut output_adagrad = vec![0.0; count];
        for shard in &self.ps_shards {
            shard.export_adagrad(ParameterKind::Input, &mut input_adagrad);
            shard.export_adagrad(ParameterKind::Output, &mut output_adagrad);
        }
        Ok(DownpourTrainingState {
            descriptor: DownpourStateDescriptor {
                schema_version: DOWNPOUR_TRAINING_STATE_SCHEMA_VERSION,
                config_digest: downpour_config_digest(&self.trainer.config, &self.downpour_config),
                vocabulary_digest: self.trainer.vocab.digest(),
                corpus_digest: self.trainer.corpus_digest.clone(),
            },
            completed_epochs: self.completed_epochs,
            processed_tokens: self.trainer.processed_tokens(),
            vocabulary: self.trainer.vocab.export_state(),
            input_embeddings,
            output_embeddings,
            input_adagrad,
            output_adagrad,
            replicas: self.replicas.iter().map(Replica::export_state).collect(),
        })
    }
}

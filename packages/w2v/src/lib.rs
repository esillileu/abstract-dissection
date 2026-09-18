//! Shared types and public contracts for the modular word2vec port.
//!
//! Configuration is fixed in stage 1; stage 2 adds independent primitives.

pub mod atomic_float;
pub mod config;
pub mod corpus;
mod identity;
pub mod model;
pub mod random;
pub mod trainer;
pub mod training;
pub mod vocab;

#[cfg(feature = "python")]
mod python;

pub use config::{
    HsOutOfRangePolicy, ModelKind, ObjectiveKind, RngAlgorithm, Status, TrainingConfig,
    VocabularyConfig,
};
pub use corpus::Corpus;
pub use model::{EmbeddingKind, Model, SigmoidTable};
pub use trainer::{
    EpochReport, Observation, StateDescriptor, Trainer, TrainingSession, TrainingState,
};
pub use training::worker::WorkerState;
pub use vocab::{NegativeSampler, Vocabulary, VocabularyEntry, VocabularyState};

#[cfg(feature = "python")]
pub use python::*;

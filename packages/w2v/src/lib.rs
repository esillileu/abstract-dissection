//! Shared types and public contracts for the modular word2vec port.
//!
//! Configuration is fixed in stage 1; stage 2 adds independent primitives.

pub mod atomic_float;
pub mod config;
pub mod corpus;
pub mod model;
pub mod random;
pub mod trainer;
pub mod training;
pub mod vocab;

pub use config::{
    HsOutOfRangePolicy, ModelKind, ObjectiveKind, RngAlgorithm, Status, TrainingConfig,
    VocabularyConfig,
};
pub use corpus::Corpus;
pub use model::{EmbeddingKind, Model, SigmoidTable};
pub use trainer::Trainer;
pub use vocab::{NegativeSampler, Vocabulary, VocabularyEntry};

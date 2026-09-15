//! Shared types and public contracts for the modular word2vec port.
//!
//! This first stage fixes ownership and configuration semantics. Constructors
//! and algorithm implementations are added in later stages.

pub mod config;
pub mod corpus;
pub mod model;
pub mod trainer;
pub mod vocab;

pub use config::{
    HsOutOfRangePolicy, ModelKind, ObjectiveKind, RngAlgorithm, Status, TrainingConfig,
    VocabularyConfig,
};
pub use corpus::Corpus;
pub use model::{EmbeddingKind, Model, SigmoidTable};
pub use trainer::Trainer;
pub use vocab::{NegativeSampler, Vocabulary, VocabularyEntry};

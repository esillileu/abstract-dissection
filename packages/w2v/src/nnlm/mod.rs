//! Feed-forward next-word language models, independent of word2vec arithmetic.
pub mod config;
pub mod downpour;
pub mod model;
pub mod session;
pub mod training;

pub use config::{HiddenActivation, NnlmTrainingConfig};
pub use downpour::{
    NnlmAdaGradState, NnlmDownpourTrainingSession, NnlmDownpourTrainingState, NnlmReplicaState,
};
pub use model::NnlmModel;
pub use session::{NnlmTrainingSession, NnlmTrainingState};

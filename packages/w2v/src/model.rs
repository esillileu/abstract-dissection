use crate::config::Real;
use std::sync::atomic::AtomicU32;

#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbeddingKind {
    Input = 0,
    Output = 1,
}

/// Coordinates store float bit patterns. The relaxed CAS operations arrive in stage 2.
pub struct Model {
    pub input_embeddings: Vec<AtomicU32>,
    pub output_embeddings: Vec<AtomicU32>,
    pub vocab_size: usize,
    pub embedding_dimension: usize,
}

#[derive(Clone, Debug, Default)]
pub struct SigmoidTable {
    pub values: Vec<Real>,
    pub max: Real,
}

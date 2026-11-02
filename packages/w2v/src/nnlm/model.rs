use super::NnlmTrainingConfig;
use crate::{
    Status, Vocabulary,
    config::Real,
    random::{Rng, RngPurpose, derive_seed},
};

#[derive(Clone, Debug, PartialEq)]
pub struct NnlmModel {
    pub input_embeddings: Vec<Real>,
    /// Row-major H by (N * D).
    pub hidden_weights: Vec<Real>,
    pub hidden_bias: Vec<Real>,
    /// Row-major (V - 1) by H, indexed by Huffman internal node.
    pub output_weights: Vec<Real>,
    pub vocab_size: usize,
    pub embedding_dimension: usize,
    pub history_length: usize,
    pub hidden_dimension: usize,
}

fn zeros(n: usize) -> Result<Vec<Real>, Status> {
    if n > isize::MAX as usize / size_of::<Real>() {
        return Err(Status::InvalidArgument);
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(n)
        .map_err(|_| Status::OutOfMemory)?;
    values.resize(n, 0.0);
    Ok(values)
}

impl NnlmModel {
    pub(crate) fn validate(&self, vocabulary: &Vocabulary, config: &NnlmTrainingConfig) -> bool {
        let v = vocabulary.entries.len();
        config.validate() == Status::Ok
            && v >= 2
            && self.vocab_size == v
            && self.embedding_dimension == config.embedding_dimension
            && self.history_length == config.history_length
            && self.hidden_dimension == config.hidden_dimension
            && v.checked_mul(config.embedding_dimension) == Some(self.input_embeddings.len())
            && self.hidden_weights.len()
                == config.embedding_dimension * config.history_length * config.hidden_dimension
            && self.hidden_bias.len() == config.hidden_dimension
            && (v - 1).checked_mul(config.hidden_dimension) == Some(self.output_weights.len())
            && self
                .input_embeddings
                .iter()
                .chain(&self.hidden_weights)
                .chain(&self.hidden_bias)
                .chain(&self.output_weights)
                .all(|v| v.is_finite())
    }
    pub fn create(vocab: &Vocabulary, config: &NnlmTrainingConfig) -> Result<Self, Status> {
        if config.validate() != Status::Ok || vocab.entries.len() < 2 {
            return Err(Status::InvalidArgument);
        }
        let v = vocab.entries.len();
        let d = config.embedding_dimension;
        let h = config.hidden_dimension;
        let p = config.history_length * d;
        let mut input_embeddings = zeros(v.checked_mul(d).ok_or(Status::InvalidArgument)?)?;
        let mut hidden_weights = zeros(p.checked_mul(h).ok_or(Status::InvalidArgument)?)?;
        let mut rng = Rng::new(
            derive_seed(config.root_seed, 0, RngPurpose::Model),
            config.rng_algorithm,
        );
        for value in &mut input_embeddings {
            *value = (rng.uniform() - 0.5) / d as Real;
        }
        // Reconstruction mechanism: Xavier uniform for the nonlinear layer.
        // This initialization is not claimed as a paper-specified choice.
        let scale = (6.0 / (p as Real + h as Real)).sqrt();
        for value in &mut hidden_weights {
            *value = (2.0 * rng.uniform() - 1.0) * scale;
        }
        Ok(Self {
            input_embeddings,
            hidden_weights,
            hidden_bias: zeros(h)?,
            output_weights: zeros((v - 1).checked_mul(h).ok_or(Status::InvalidArgument)?)?,
            vocab_size: v,
            embedding_dimension: d,
            history_length: config.history_length,
            hidden_dimension: h,
        })
    }
}

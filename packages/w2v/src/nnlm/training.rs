use super::{HiddenActivation, NnlmModel};
use crate::{Status, Vocabulary, config::Real};
use std::collections::BTreeMap;

/// Sparse rows include each repeated history word only once, with summed derivatives.
#[derive(Clone, Debug)]
pub struct NnlmGradient {
    pub embeddings: BTreeMap<usize, Vec<Real>>,
    pub projection: Vec<Real>,
    pub hidden_bias: Vec<Real>,
    pub output: BTreeMap<usize, Vec<Real>>,
    pub loss: f64,
}

impl NnlmGradient {
    pub fn hidden_weights(&self) -> Vec<Real> {
        let mut weights = Vec::with_capacity(self.hidden_bias.len() * self.projection.len());
        for &dz in &self.hidden_bias {
            for &p in &self.projection {
                weights.push(dz * p);
            }
        }
        weights
    }
}

/// NNLM parameter access for a dense model or a sparse replica cache.
pub(crate) trait Parameters {
    fn dimensions(&self) -> (usize, usize, usize, usize);
    fn input_row(&mut self, id: usize) -> &[Real];
    fn output_row(&mut self, id: usize) -> &[Real];
    fn hidden_weights(&self) -> &[Real];
    fn hidden_bias(&self) -> &[Real];
}

impl Parameters for &NnlmModel {
    fn dimensions(&self) -> (usize, usize, usize, usize) {
        (
            self.vocab_size,
            self.embedding_dimension,
            self.history_length,
            self.hidden_dimension,
        )
    }
    fn input_row(&mut self, id: usize) -> &[Real] {
        &self.input_embeddings[id * self.embedding_dimension..(id + 1) * self.embedding_dimension]
    }
    fn output_row(&mut self, id: usize) -> &[Real] {
        &self.output_weights[id * self.hidden_dimension..(id + 1) * self.hidden_dimension]
    }
    fn hidden_weights(&self) -> &[Real] {
        &self.hidden_weights
    }
    fn hidden_bias(&self) -> &[Real] {
        &self.hidden_bias
    }
}

pub fn forward_backward(
    model: &NnlmModel,
    vocab: &Vocabulary,
    activation: HiddenActivation,
    history: &[usize],
    target: usize,
) -> Result<NnlmGradient, Status> {
    forward_backward_parameters(&mut &*model, vocab, activation, history, target)
}

pub(crate) fn forward_backward_parameters(
    model: &mut impl Parameters,
    vocab: &Vocabulary,
    activation: HiddenActivation,
    history: &[usize],
    target: usize,
) -> Result<NnlmGradient, Status> {
    let (v, d, n, h) = model.dimensions();
    if history.len() != n
        || history.iter().any(|id| *id >= v)
        || target >= v
        || vocab.entries.len() != v
    {
        return Err(Status::InvalidArgument);
    }
    let p = d * history.len();
    let mut projection = Vec::with_capacity(p);
    for id in history {
        projection.extend_from_slice(model.input_row(*id));
    }
    let mut hidden = vec![0.0; h];
    for (row, value) in hidden.iter_mut().enumerate() {
        let z = model.hidden_bias()[row]
            + model.hidden_weights()[row * p..(row + 1) * p]
                .iter()
                .zip(&projection)
                .map(|(w, x)| w * x)
                .sum::<Real>();
        *value = match activation {
            HiddenActivation::Tanh => z.tanh(),
            HiddenActivation::Sigmoid => logistic(z),
        };
    }
    let entry = &vocab.entries[target];
    if entry.huffman_path.len() != entry.huffman_bits.len()
        || entry.huffman_path.iter().any(|node| *node >= v - 1)
        || entry.huffman_bits.iter().any(|bit| *bit > 1)
    {
        return Err(Status::CorruptData);
    }
    let mut gradient = NnlmGradient {
        embeddings: BTreeMap::new(),
        projection: Vec::new(),
        hidden_bias: vec![0.0; h],
        output: BTreeMap::new(),
        loss: 0.0,
    };
    let mut dh = vec![0.0; h];
    for (&node, &bit) in entry.huffman_path.iter().zip(&entry.huffman_bits) {
        let weights = model.output_row(node);
        let score: Real = weights.iter().zip(&hidden).map(|(w, x)| w * x).sum();
        let label = 1.0 - bit as Real;
        let delta = logistic(score) - label;
        let score64 = score as f64;
        gradient.loss += score64.max(0.0) + (-score64.abs()).exp().ln_1p() - label as f64 * score64;
        let row = gradient.output.entry(node).or_insert_with(|| vec![0.0; h]);
        for j in 0..h {
            dh[j] += delta * weights[j];
            row[j] += delta * hidden[j];
        }
    }
    let mut dp = vec![0.0; p];
    for j in 0..h {
        let derivative = match activation {
            HiddenActivation::Tanh => 1.0 - hidden[j] * hidden[j],
            HiddenActivation::Sigmoid => hidden[j] * (1.0 - hidden[j]),
        };
        let dz = dh[j] * derivative;
        gradient.hidden_bias[j] = dz;
        crate::simd::scaled_accumulate(&mut dp, &model.hidden_weights()[j * p..(j + 1) * p], dz);
    }
    for (position, id) in history.iter().enumerate() {
        let row = gradient
            .embeddings
            .entry(*id)
            .or_insert_with(|| vec![0.0; d]);
        for k in 0..d {
            row[k] += dp[position * d + k];
        }
    }
    gradient.projection = projection;
    Ok(gradient)
}

pub(crate) fn logistic(x: Real) -> Real {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

pub fn apply_sgd(model: &mut NnlmModel, gradient: &NnlmGradient, rate: Real) {
    for (id, row) in &gradient.embeddings {
        for (value, derivative) in model.input_embeddings
            [id * model.embedding_dimension..(id + 1) * model.embedding_dimension]
            .iter_mut()
            .zip(row)
        {
            *value -= rate * derivative;
        }
    }
    let p = model.history_length * model.embedding_dimension;
    for (j, &dz) in gradient.hidden_bias.iter().enumerate() {
        let row = &mut model.hidden_weights[j * p..(j + 1) * p];
        for (value, &proj) in row.iter_mut().zip(&gradient.projection) {
            *value -= rate * dz * proj;
        }
    }
    for (value, derivative) in model.hidden_bias.iter_mut().zip(&gradient.hidden_bias) {
        *value -= rate * derivative;
    }
    for (id, row) in &gradient.output {
        for (value, derivative) in model.output_weights
            [id * model.hidden_dimension..(id + 1) * model.hidden_dimension]
            .iter_mut()
            .zip(row)
        {
            *value -= rate * derivative;
        }
    }
}

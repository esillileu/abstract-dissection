use super::gradient::{GradientBatch, ParameterKind, global_row_index, local_row_index};
use crate::{
    atomic_float,
    config::{Real, Status},
    model::Model,
};
use std::sync::Arc;

pub struct ParameterServerShard {
    pub shard_id: usize,
    pub shard_count: usize,
    pub embedding_dimension: usize,
    pub vocab_size: usize,
    pub gamma: Real,
    pub epsilon: Real,
    pub model: Arc<Model>,
    pub input_adagrad: Vec<Real>,
    pub output_adagrad: Vec<Real>,
}

impl ParameterServerShard {
    pub fn initialize(
        shard_id: usize,
        shard_count: usize,
        vocab_size: usize,
        embedding_dimension: usize,
        gamma: Real,
        epsilon: Real,
        model: Arc<Model>,
    ) -> Result<Self, Status> {
        if shard_id >= shard_count || shard_count == 0 || embedding_dimension == 0 {
            return Err(Status::InvalidArgument);
        }
        let local_rows = vocab_size.div_ceil(shard_count);
        let element_count = local_rows
            .checked_mul(embedding_dimension)
            .ok_or(Status::InvalidArgument)?;
        let mut input_adagrad = Vec::new();
        let mut output_adagrad = Vec::new();
        input_adagrad
            .try_reserve_exact(element_count)
            .map_err(|_| Status::OutOfMemory)?;
        output_adagrad
            .try_reserve_exact(element_count)
            .map_err(|_| Status::OutOfMemory)?;
        input_adagrad.resize(element_count, 0.0);
        output_adagrad.resize(element_count, 0.0);
        Ok(Self {
            shard_id,
            shard_count,
            embedding_dimension,
            vocab_size,
            gamma,
            epsilon,
            model,
            input_adagrad,
            output_adagrad,
        })
    }

    #[inline]
    pub fn apply_batch(&mut self, batch: &GradientBatch) {
        let dim = self.embedding_dimension;
        for (idx, header) in batch.headers.iter().enumerate() {
            let grad_slice = &batch.gradients[idx * dim..(idx + 1) * dim];
            self.apply_gradient(header.kind, header.row_index, grad_slice);
        }
    }

    #[inline]
    pub fn apply_gradient(&mut self, kind: ParameterKind, row_index: usize, grad_slice: &[Real]) {
        let dim = self.embedding_dimension;
        let local_idx = local_row_index(row_index, self.shard_count);
        let start = local_idx * dim;
        let accum = match kind {
            ParameterKind::Input => &mut self.input_adagrad[start..start + dim],
            ParameterKind::Output => &mut self.output_adagrad[start..start + dim],
        };
        let model_start = row_index * dim;
        let model_row = match kind {
            ParameterKind::Input => &self.model.input_embeddings[model_start..model_start + dim],
            ParameterKind::Output => &self.model.output_embeddings[model_start..model_start + dim],
        };
        for i in 0..dim {
            let g = grad_slice[i];
            accum[i] += g * g;
            let step = self.gamma * g / (accum[i] + self.epsilon).sqrt();
            let current = atomic_float::load(&model_row[i]);
            atomic_float::store(&model_row[i], current + step);
        }
    }

    pub fn export_adagrad(&self, kind: ParameterKind, global_adagrad: &mut [Real]) {
        let dim = self.embedding_dimension;
        let local_rows = self.vocab_size.div_ceil(self.shard_count);
        for local_idx in 0..local_rows {
            let global_row = global_row_index(kind, local_idx, self.shard_id, self.shard_count);
            if global_row < self.vocab_size {
                let lstart = local_idx * dim;
                let gstart = global_row * dim;
                let src = match kind {
                    ParameterKind::Input => &self.input_adagrad[lstart..lstart + dim],
                    ParameterKind::Output => &self.output_adagrad[lstart..lstart + dim],
                };
                global_adagrad[gstart..gstart + dim].copy_from_slice(src);
            }
        }
    }

    pub fn restore_adagrad(&mut self, kind: ParameterKind, global_adagrad: &[Real]) {
        let dim = self.embedding_dimension;
        let local_rows = self.vocab_size.div_ceil(self.shard_count);
        for local_idx in 0..local_rows {
            let global_row = global_row_index(kind, local_idx, self.shard_id, self.shard_count);
            if global_row < self.vocab_size {
                let lstart = local_idx * dim;
                let gstart = global_row * dim;
                let src = &global_adagrad[gstart..gstart + dim];
                let dst = match kind {
                    ParameterKind::Input => &mut self.input_adagrad[lstart..lstart + dim],
                    ParameterKind::Output => &mut self.output_adagrad[lstart..lstart + dim],
                };
                dst.copy_from_slice(src);
            }
        }
    }
}

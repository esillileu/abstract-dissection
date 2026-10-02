use crate::{
    atomic_float,
    config::{Real, UpdateStrategy},
    model::Model,
    simd,
    training::shared_add,
};

pub trait ParameterBackend {
    fn embedding_dimension(&self) -> usize;
    fn load_input_row(&mut self, token: usize, destination: &mut [Real]);
    fn accumulate_input_row(&mut self, token: usize, destination: &mut [Real]);
    fn load_output_row(&mut self, token: usize, destination: &mut [Real]);
    fn apply_output_gradient(
        &mut self,
        token: usize,
        hidden: &[Real],
        output_snapshot: &mut [Real],
        raw_error: Real,
        learning_rate: Real,
        hidden_gradient: &mut [Real],
    );
    fn apply_input_gradient(&mut self, token: usize, hidden_gradient: &[Real], learning_rate: Real);
}

pub struct SharedModelBackend<'a> {
    pub model: &'a Model,
    pub update_strategy: UpdateStrategy,
}

impl<'a> SharedModelBackend<'a> {
    #[inline(always)]
    pub fn new(model: &'a Model, update_strategy: UpdateStrategy) -> Self {
        Self {
            model,
            update_strategy,
        }
    }
}

impl<'a> ParameterBackend for SharedModelBackend<'a> {
    #[inline(always)]
    fn embedding_dimension(&self) -> usize {
        self.model.embedding_dimension
    }

    #[inline(always)]
    fn load_input_row(&mut self, token: usize, destination: &mut [Real]) {
        let dim = self.model.embedding_dimension;
        let start = token * dim;
        let row = &self.model.input_embeddings[start..start + dim];
        for (dst, src) in destination.iter_mut().zip(row) {
            *dst = atomic_float::load(src);
        }
    }

    #[inline(always)]
    fn accumulate_input_row(&mut self, token: usize, destination: &mut [Real]) {
        let dim = self.model.embedding_dimension;
        let start = token * dim;
        let row = &self.model.input_embeddings[start..start + dim];
        for (dst, src) in destination.iter_mut().zip(row) {
            *dst += atomic_float::load(src);
        }
    }

    #[inline(always)]
    fn load_output_row(&mut self, token: usize, destination: &mut [Real]) {
        let dim = self.model.embedding_dimension;
        let start = token * dim;
        let row = &self.model.output_embeddings[start..start + dim];
        for (dst, src) in destination.iter_mut().zip(row) {
            *dst = atomic_float::load(src);
        }
    }

    #[inline(always)]
    fn apply_output_gradient(
        &mut self,
        token: usize,
        hidden: &[Real],
        output_snapshot: &mut [Real],
        raw_error: Real,
        learning_rate: Real,
        hidden_gradient: &mut [Real],
    ) {
        let gradient_scale = raw_error * learning_rate;
        simd::scaled_accumulate(hidden_gradient, output_snapshot, gradient_scale);
        let dim = self.model.embedding_dimension;
        let start = token * dim;
        let output_row = &self.model.output_embeddings[start..start + dim];
        match self.update_strategy {
            UpdateStrategy::AtomicCas => {
                for (snapshot, value) in output_snapshot.iter_mut().zip(hidden) {
                    *snapshot = gradient_scale * value;
                }
                for (shared, delta) in output_row.iter().zip(output_snapshot.iter()) {
                    atomic_float::add(shared, *delta);
                }
            }
            UpdateStrategy::Hogwild => {
                simd::scaled_accumulate(output_snapshot, hidden, gradient_scale);
                for (shared, value) in output_row.iter().zip(output_snapshot.iter()) {
                    atomic_float::store(shared, *value);
                }
            }
        }
    }

    #[inline(always)]
    fn apply_input_gradient(
        &mut self,
        token: usize,
        hidden_gradient: &[Real],
        _learning_rate: Real,
    ) {
        let dim = self.model.embedding_dimension;
        let start = token * dim;
        let input_row = &self.model.input_embeddings[start..start + dim];
        for (coordinate, value) in input_row.iter().enumerate() {
            shared_add(value, hidden_gradient[coordinate], self.update_strategy);
        }
    }
}

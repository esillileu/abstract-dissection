use crate::{
    atomic_float,
    config::{Real, RngAlgorithm, Status},
    random::{Rng, RngPurpose, derive_seed},
    vocab::Vocabulary,
};
use std::sync::atomic::AtomicU32;

#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbeddingKind {
    Input = 0,
    Output = 1,
}

/// Coordinates store float bit patterns for relaxed atomic operations.
pub struct Model {
    pub input_embeddings: Vec<AtomicU32>,
    pub output_embeddings: Vec<AtomicU32>,
    pub vocab_size: usize,
    pub embedding_dimension: usize,
}
impl Model {
    pub fn create(
        vocab: &Vocabulary,
        embedding_dimension: usize,
        root_seed: u64,
        rng_algorithm: RngAlgorithm,
    ) -> Result<Self, Status> {
        let Some(element_count) = vocab.entries.len().checked_mul(embedding_dimension) else {
            return Err(Status::InvalidArgument);
        };
        if element_count == 0 || element_count > usize::MAX / size_of::<AtomicU32>() {
            return Err(Status::InvalidArgument);
        }
        let mut input_embeddings = Vec::new();
        let mut output_embeddings = Vec::new();
        input_embeddings
            .try_reserve_exact(element_count)
            .map_err(|_| Status::OutOfMemory)?;
        output_embeddings
            .try_reserve_exact(element_count)
            .map_err(|_| Status::OutOfMemory)?;
        let mut rng = Rng::new(derive_seed(root_seed, 0, RngPurpose::Model), rng_algorithm);
        for _ in 0..element_count {
            let random_value = rng.uniform();
            let initialized_value = (random_value - 0.5) / embedding_dimension as Real;
            input_embeddings.push(AtomicU32::new(initialized_value.to_bits()));
            output_embeddings.push(AtomicU32::new(0.0f32.to_bits()));
        }
        Ok(Self {
            input_embeddings,
            output_embeddings,
            vocab_size: vocab.entries.len(),
            embedding_dimension,
        })
    }

    pub fn snapshot_into(&self, kind: EmbeddingKind, destination: &mut [Real]) -> Status {
        let required_count = self.vocab_size * self.embedding_dimension;
        if destination.len() != required_count {
            return Status::InvalidArgument;
        }
        let source = if kind == EmbeddingKind::Input {
            &self.input_embeddings
        } else {
            &self.output_embeddings
        };
        for (destination, coordinate) in destination.iter_mut().zip(source.iter()) {
            *destination = atomic_float::load(coordinate);
        }
        Status::Ok
    }
}

#[derive(Clone, Debug, Default)]
pub struct SigmoidTable {
    pub values: Vec<Real>,
    pub max: Real,
}
impl SigmoidTable {
    pub fn initialize(table_size: usize, maximum: Real) -> Result<Self, Status> {
        if table_size < 2
            || !maximum.is_finite()
            || maximum <= 0.0
            || table_size > usize::MAX / size_of::<Real>()
        {
            return Err(Status::InvalidArgument);
        }
        let mut values = Vec::new();
        values
            .try_reserve_exact(table_size)
            .map_err(|_| Status::OutOfMemory)?;
        for index in 0..table_size {
            let scaled_index = index as Real / table_size as Real;
            let input = (scaled_index * 2.0 - 1.0) * maximum;
            let exponential = (input as f64).exp() as Real;
            values.push(exponential / (exponential + 1.0));
        }
        Ok(Self {
            values,
            max: maximum,
        })
    }

    pub fn lookup(&self, value: Real) -> Real {
        if value <= -self.max {
            return 0.0;
        }
        if value >= self.max {
            return 1.0;
        }
        let index_scale = (self.values.len() as Real / self.max / 2.0) as usize;
        let table_index = ((value + self.max) * index_scale as Real) as usize;
        self.values[table_index.min(self.values.len() - 1)]
    }
}

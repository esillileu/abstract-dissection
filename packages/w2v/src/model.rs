use crate::config::{Real, Status};
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

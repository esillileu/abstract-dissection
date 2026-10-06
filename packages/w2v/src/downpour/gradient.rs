use crate::config::Real;

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ParameterKind {
    Input = 0,
    Output = 1,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GradientRowHeader {
    pub kind: ParameterKind,
    pub row_index: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GradientBatch {
    pub replica_id: usize,
    pub batch_id: u64,
    pub headers: Vec<GradientRowHeader>,
    pub gradients: Vec<Real>,
}

impl GradientBatch {
    pub fn new(replica_id: usize, batch_id: u64) -> Self {
        Self {
            replica_id,
            batch_id,
            headers: Vec::new(),
            gradients: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.headers.is_empty()
    }
}

/// Deterministic, mutually-exclusive shard assignment for parameter rows.
/// Shard assignment is guaranteed to cover [0, shard_count).
#[inline(always)]
pub fn shard_index(kind: ParameterKind, row_index: usize, shard_count: usize) -> usize {
    match kind {
        ParameterKind::Input => row_index % shard_count,
        ParameterKind::Output => (row_index + 1) % shard_count,
    }
}

/// Maps global row index to a shard-local row index.
#[inline(always)]
pub fn local_row_index(row_index: usize, shard_count: usize) -> usize {
    row_index / shard_count
}

/// Inverts local row index back to global row index for a given shard.
#[inline(always)]
pub fn global_row_index(
    kind: ParameterKind,
    local_index: usize,
    shard_id: usize,
    shard_count: usize,
) -> usize {
    match kind {
        ParameterKind::Input => local_index * shard_count + shard_id,
        ParameterKind::Output => {
            // shard_id = (row + 1) % shard_count
            // row % shard_count = (shard_id + shard_count - 1) % shard_count
            let rem = (shard_id + shard_count - 1) % shard_count;
            local_index * shard_count + rem
        }
    }
}

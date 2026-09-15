use crate::{
    config::{MAX_CODE_LENGTH, Status},
    random::Rng,
};

/// Token bytes remain bytes: the C tokenizer does not require UTF-8.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VocabularyEntry {
    pub token: Vec<u8>,
    pub count: u64,
    pub huffman_path: Vec<usize>,
    pub huffman_bits: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct Vocabulary {
    pub entries: Vec<VocabularyEntry>,
    pub hash_slots: Vec<Option<usize>>,
    pub retained_token_count: u64,
}

#[derive(Clone, Debug, Default)]
pub struct NegativeSampler {
    pub table: Vec<usize>,
}

impl Vocabulary {
    /// Assign the same parent-node paths and bits as the modular C oracle.
    pub fn assign_huffman(&mut self) -> Status {
        let leaf_count = self.entries.len();
        if leaf_count == 0 {
            return Status::InvalidArgument;
        }
        if leaf_count == 1 {
            return Status::Ok;
        }
        let Some(node_count) = leaf_count.checked_mul(2).and_then(|n| n.checked_sub(1)) else {
            return Status::InvalidArgument;
        };
        let mut counts = Vec::new();
        let mut parents = Vec::new();
        let mut bits = Vec::new();
        if counts.try_reserve_exact(node_count).is_err()
            || parents.try_reserve_exact(node_count).is_err()
            || bits.try_reserve_exact(node_count).is_err()
        {
            return Status::OutOfMemory;
        }
        counts.resize(node_count, 1_000_000_000_000_000u64);
        parents.resize(node_count, 0usize);
        bits.resize(node_count, 0u8);
        for (index, entry) in self.entries.iter().enumerate() {
            counts[index] = entry.count;
        }

        let mut next_leaf = leaf_count;
        let mut next_internal = leaf_count;
        for node in leaf_count..node_count {
            let mut chosen = [0usize; 2];
            for child in &mut chosen {
                let choose_leaf = next_leaf > 0
                    && (next_internal >= node || counts[next_leaf - 1] < counts[next_internal]);
                if choose_leaf {
                    next_leaf -= 1;
                    *child = next_leaf;
                } else {
                    *child = next_internal;
                    next_internal += 1;
                }
            }
            counts[node] = counts[chosen[0]].wrapping_add(counts[chosen[1]]);
            parents[chosen[0]] = node;
            parents[chosen[1]] = node;
            bits[chosen[1]] = 1;
        }

        for index in 0..leaf_count {
            let mut node = index;
            let mut reverse_path = [0usize; MAX_CODE_LENGTH];
            let mut reverse_bits = [0u8; MAX_CODE_LENGTH];
            let mut length = 0;
            while node != node_count - 1 {
                if length == MAX_CODE_LENGTH {
                    return Status::CorruptData;
                }
                reverse_path[length] = parents[node] - leaf_count;
                reverse_bits[length] = bits[node];
                length += 1;
                node = parents[node];
            }
            let entry = &mut self.entries[index];
            entry.huffman_path.clear();
            entry.huffman_bits.clear();
            if entry.huffman_path.try_reserve_exact(length).is_err()
                || entry.huffman_bits.try_reserve_exact(length).is_err()
            {
                return Status::OutOfMemory;
            }
            for path_index in (0..length).rev() {
                entry.huffman_path.push(reverse_path[path_index]);
                entry.huffman_bits.push(reverse_bits[path_index]);
            }
        }
        Status::Ok
    }
}

impl NegativeSampler {
    pub fn initialize(vocab: &Vocabulary, table_size: usize) -> Result<Self, Status> {
        if vocab.entries.is_empty()
            || table_size == 0
            || table_size > usize::MAX / size_of::<usize>()
        {
            return Err(Status::InvalidArgument);
        }
        let mut table = Vec::new();
        table
            .try_reserve_exact(table_size)
            .map_err(|_| Status::OutOfMemory)?;
        let mut total_weight = 0.0f64;
        for entry in &vocab.entries {
            total_weight += (entry.count as f64).powf(0.75);
        }
        if total_weight <= 0.0 || !total_weight.is_finite() {
            return Err(Status::CorruptData);
        }
        let mut entry_index = 0;
        let mut cumulative_probability = (vocab.entries[0].count as f64).powf(0.75) / total_weight;
        for table_index in 0..table_size {
            let quantile = table_index as f64 / table_size as f64;
            table.push(entry_index);
            if quantile > cumulative_probability && entry_index + 1 < vocab.entries.len() {
                entry_index += 1;
                cumulative_probability +=
                    (vocab.entries[entry_index].count as f64).powf(0.75) / total_weight;
            }
        }
        Ok(Self { table })
    }

    /// Returns the sampled entry and raw random value used by boundary fallback.
    pub fn draw(&self, rng: &mut Rng) -> (usize, u64) {
        let random_value = rng.next_u64();
        let index = ((random_value >> 16) as usize) % self.table.len();
        (self.table[index], random_value)
    }
}

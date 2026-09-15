use crate::{
    config::{MAX_CODE_LENGTH, Status, VocabularyConfig},
    corpus::Corpus,
    random::Rng,
};

fn token_hash(token: &[u8], capacity: usize) -> usize {
    let mut hash = 0u64;
    for &byte in token {
        hash = hash.wrapping_mul(257).wrapping_add(byte as u64);
    }
    (hash % capacity as u64) as usize
}

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
    pub fn find(&self, token: &[u8]) -> Option<usize> {
        let capacity = self.hash_slots.len();
        if capacity == 0 {
            return None;
        }
        let mut slot = token_hash(token, capacity);
        for _ in 0..capacity {
            match self.hash_slots[slot] {
                None => return None,
                Some(index) if index < self.entries.len() && self.entries[index].token == token => {
                    return Some(index);
                }
                _ => slot = (slot + 1) % capacity,
            }
        }
        None
    }

    fn rebuild_hash(&mut self) -> Status {
        self.hash_slots.fill(None);
        let capacity = self.hash_slots.len();
        for (index, entry) in self.entries.iter().enumerate() {
            let mut slot = token_hash(&entry.token, capacity);
            let mut probes = 0;
            while self.hash_slots[slot].is_some() && probes < capacity {
                slot = (slot + 1) % capacity;
                probes += 1;
            }
            if probes >= capacity {
                return Status::OutOfMemory;
            }
            self.hash_slots[slot] = Some(index);
        }
        Status::Ok
    }

    fn add_token(&mut self, token: &[u8]) -> Result<usize, Status> {
        if self.entries.len() >= self.hash_slots.len() {
            return Err(Status::InvalidArgument);
        }
        if let Some(index) = self.find(token) {
            return Ok(index);
        }
        if self.entries.len() == self.entries.capacity() {
            self.entries
                .try_reserve_exact(1000)
                .map_err(|_| Status::OutOfMemory)?;
        }
        let mut owned_token = Vec::new();
        owned_token
            .try_reserve_exact(token.len())
            .map_err(|_| Status::OutOfMemory)?;
        owned_token.extend_from_slice(token);
        let index = self.entries.len();
        self.entries.push(VocabularyEntry {
            token: owned_token,
            count: 0,
            huffman_path: Vec::new(),
            huffman_bits: Vec::new(),
        });
        let capacity = self.hash_slots.len();
        let mut slot = token_hash(token, capacity);
        while self.hash_slots[slot].is_some() {
            slot = (slot + 1) % capacity;
        }
        self.hash_slots[slot] = Some(index);
        Ok(index)
    }

    fn prune(&mut self, threshold: u64) -> Status {
        let mut index = 0;
        self.entries.retain(|entry| {
            let keep = index == 0 || entry.count > threshold;
            index += 1;
            keep
        });
        self.rebuild_hash()
    }

    pub fn build(corpus: &Corpus, config: &VocabularyConfig) -> Result<Self, Status> {
        if config.validate() != Status::Ok {
            return Err(Status::InvalidArgument);
        }
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(config.initial_capacity)
            .map_err(|_| Status::OutOfMemory)?;
        let mut hash_slots = Vec::new();
        hash_slots
            .try_reserve_exact(config.hash_capacity)
            .map_err(|_| Status::OutOfMemory)?;
        hash_slots.resize(config.hash_capacity, None);
        let mut vocab = Self {
            entries,
            hash_slots,
            retained_token_count: 0,
        };
        vocab.add_token(b"</s>")?;
        let mut tokenizer = corpus.tokenizer(0)?;
        let mut prune_threshold = 1u64;
        loop {
            let read = tokenizer.read_token()?;
            if read.at_eof {
                break;
            }
            let index = match vocab.find(&read.token) {
                Some(index) => index,
                None => vocab.add_token(&read.token)?,
            };
            vocab.entries[index].count = vocab.entries[index].count.wrapping_add(1);
            let capacity = config.hash_capacity;
            let crowded_limit = (capacity / 10) * 7 + ((capacity % 10) * 7) / 10;
            if vocab.entries.len() > crowded_limit {
                let status = vocab.prune(prune_threshold);
                if status != Status::Ok {
                    return Err(status);
                }
                prune_threshold = prune_threshold.wrapping_add(1);
            }
        }
        if vocab.entries.len() > 1 {
            // Equal-count qsort ties are platform-specific. The supported C
            // fixture preserves encounter order, so preserve it here too.
            vocab.entries[1..].sort_by_key(|entry| std::cmp::Reverse(entry.count));
        }
        let mut index = 0;
        vocab.entries.retain(|entry| {
            let keep = index == 0 || entry.count >= config.min_count;
            index += 1;
            keep
        });
        vocab.retained_token_count = vocab
            .entries
            .iter()
            .fold(0u64, |total, entry| total.wrapping_add(entry.count));
        let status = vocab.rebuild_hash();
        if status != Status::Ok {
            return Err(status);
        }
        let status = vocab.assign_huffman();
        if status != Status::Ok {
            return Err(status);
        }
        Ok(vocab)
    }

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

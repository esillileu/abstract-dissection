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

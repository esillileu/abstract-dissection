use std::path::PathBuf;

/// The C `Corpus` metadata contract. Corpus I/O is implemented in stage 2.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Corpus {
    pub path: PathBuf,
    pub byte_size: usize,
}

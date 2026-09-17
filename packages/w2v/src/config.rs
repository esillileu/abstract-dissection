//! Rust counterparts to `reference/include/w2v/config.h` and `src/config/config.c`.

pub const MAX_TOKEN_LENGTH: usize = 100;
pub const MAX_SENTENCE_LENGTH: usize = 1000;
pub const MAX_CODE_LENGTH: usize = 40;
pub type Real = f32;

#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Ok = 0,
    InvalidArgument = 1,
    OutOfMemory = 2,
    IoError = 3,
    CorruptData = 4,
    ThreadError = 5,
    SchemaMismatch = 6,
    IdentityMismatch = 7,
    InvalidState = 8,
}
impl Status {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::InvalidArgument => "invalid argument",
            Self::OutOfMemory => "out of memory",
            Self::IoError => "I/O error",
            Self::CorruptData => "corrupt data",
            Self::ThreadError => "thread error",
            Self::SchemaMismatch => "schema mismatch",
            Self::IdentityMismatch => "identity mismatch",
            Self::InvalidState => "invalid state",
        }
    }
}

#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelKind {
    Cbow = 0,
    SkipGram = 1,
}
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObjectiveKind {
    HierarchicalSoftmax = 0,
    NegativeSampling = 1,
}
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HsOutOfRangePolicy {
    Skip = 0,
    UseBoundaryValue = 1,
}
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RngAlgorithm {
    Lcg = 0,
    Xorshift = 1,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VocabularyConfig {
    pub initial_capacity: usize,
    pub hash_capacity: usize,
    pub min_count: u64,
}
impl Default for VocabularyConfig {
    fn default() -> Self {
        Self {
            initial_capacity: 1000,
            hash_capacity: 30_000_000,
            min_count: 5,
        }
    }
}
impl VocabularyConfig {
    pub fn validate(&self) -> Status {
        if self.initial_capacity == 0
            || self.hash_capacity < 2
            || self.min_count == 0
            || self.initial_capacity > usize::MAX / size_of::<*const ()>()
            || self.hash_capacity > usize::MAX / size_of::<usize>()
        {
            Status::InvalidArgument
        } else {
            Status::Ok
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrainingConfig {
    pub model_kind: ModelKind,
    pub objective_kind: ObjectiveKind,
    pub embedding_dimension: usize,
    pub window_radius: usize,
    pub epochs: usize,
    pub thread_count: usize,
    pub learning_rate_update_interval: usize,
    pub initial_learning_rate: Real,
    pub subsampling_threshold: Real,
    pub negative_sample_count: usize,
    pub root_seed: u64,
    pub rng_algorithm: RngAlgorithm,
    pub negative_table_size: usize,
    pub sigmoid_table_size: usize,
    pub sigmoid_max: Real,
    pub hs_out_of_range_policy: HsOutOfRangePolicy,
}
impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            model_kind: ModelKind::Cbow,
            objective_kind: ObjectiveKind::NegativeSampling,
            embedding_dimension: 100,
            window_radius: 5,
            epochs: 5,
            thread_count: 12,
            learning_rate_update_interval: 10_000,
            initial_learning_rate: 0.05,
            subsampling_threshold: 1e-3,
            negative_sample_count: 5,
            root_seed: 1,
            rng_algorithm: RngAlgorithm::Lcg,
            negative_table_size: 100_000_000,
            sigmoid_table_size: 1000,
            sigmoid_max: 6.0,
            hs_out_of_range_policy: HsOutOfRangePolicy::Skip,
        }
    }
}
impl TrainingConfig {
    pub fn for_model(model_kind: ModelKind) -> Self {
        let mut config = Self {
            model_kind,
            ..Self::default()
        };
        if model_kind == ModelKind::SkipGram {
            config.initial_learning_rate = 0.025;
        }
        config
    }
    pub fn validate(&self) -> Status {
        if self.embedding_dimension == 0
            || self.window_radius == 0
            || self.window_radius > usize::MAX / 2
            || self.epochs == 0
            || self.thread_count == 0
            || self.learning_rate_update_interval == 0
            || !self.initial_learning_rate.is_finite()
            || self.initial_learning_rate <= 0.0
            || !self.subsampling_threshold.is_finite()
            || self.subsampling_threshold < 0.0
            || !self.sigmoid_max.is_finite()
            || self.sigmoid_max <= 0.0
            || self.sigmoid_table_size < 2
            || self.embedding_dimension > usize::MAX / size_of::<Real>()
            || (self.objective_kind == ObjectiveKind::NegativeSampling
                && (self.negative_sample_count == 0 || self.negative_table_size == 0))
        {
            Status::InvalidArgument
        } else {
            Status::Ok
        }
    }
}

use w2v::config::{MAX_CODE_LENGTH, MAX_SENTENCE_LENGTH, MAX_TOKEN_LENGTH, Real};
use w2v::{
    HsOutOfRangePolicy, ModelKind, ObjectiveKind, RngAlgorithm, Status, TrainingConfig,
    VocabularyConfig,
};

#[test]
fn c_constants_and_discriminants() {
    assert_eq!(
        (MAX_TOKEN_LENGTH, MAX_SENTENCE_LENGTH, MAX_CODE_LENGTH),
        (100, 1000, 40)
    );
    assert_eq!(size_of::<Real>(), 4);
    assert_eq!(ModelKind::Cbow as i32, 0);
    assert_eq!(ModelKind::SkipGram as i32, 1);
    assert_eq!(ObjectiveKind::HierarchicalSoftmax as i32, 0);
    assert_eq!(ObjectiveKind::NegativeSampling as i32, 1);
    assert_eq!(RngAlgorithm::Lcg as i32, 0);
    assert_eq!(RngAlgorithm::Xorshift as i32, 1);
    assert_eq!(HsOutOfRangePolicy::Skip as i32, 0);
    assert_eq!(HsOutOfRangePolicy::UseBoundaryValue as i32, 1);
    let names = [
        "ok",
        "invalid argument",
        "out of memory",
        "I/O error",
        "corrupt data",
        "thread error",
    ];
    let statuses = [
        Status::Ok,
        Status::InvalidArgument,
        Status::OutOfMemory,
        Status::IoError,
        Status::CorruptData,
        Status::ThreadError,
    ];
    for (index, status) in statuses.into_iter().enumerate() {
        assert_eq!(status as i32, index as i32);
        assert_eq!(status.as_str(), names[index]);
    }
}

#[test]
fn c_defaults_and_model_specific_rate() {
    let vocab = VocabularyConfig::default();
    assert_eq!(
        (vocab.initial_capacity, vocab.hash_capacity, vocab.min_count),
        (1000, 30_000_000, 5)
    );
    assert_eq!(vocab.validate(), Status::Ok);

    let config = TrainingConfig::default();
    assert_eq!(config.model_kind, ModelKind::Cbow);
    assert_eq!(config.objective_kind, ObjectiveKind::NegativeSampling);
    assert_eq!(
        (
            config.embedding_dimension,
            config.window_radius,
            config.epochs,
            config.thread_count,
            config.learning_rate_update_interval
        ),
        (100, 5, 5, 12, 10_000)
    );
    assert_eq!(config.initial_learning_rate.to_bits(), 0.05f32.to_bits());
    assert_eq!(config.subsampling_threshold.to_bits(), 1e-3f32.to_bits());
    assert_eq!(
        (
            config.negative_sample_count,
            config.root_seed,
            config.negative_table_size,
            config.sigmoid_table_size
        ),
        (5, 1, 100_000_000, 1000)
    );
    assert_eq!(config.rng_algorithm, RngAlgorithm::Lcg);
    assert_eq!(config.sigmoid_max, 6.0);
    assert_eq!(config.hs_out_of_range_policy, HsOutOfRangePolicy::Skip);
    assert_eq!(config.validate(), Status::Ok);

    let skip = TrainingConfig::for_model(ModelKind::SkipGram);
    assert_eq!(skip.model_kind, ModelKind::SkipGram);
    assert_eq!(skip.initial_learning_rate.to_bits(), 0.025f32.to_bits());
    assert_eq!(skip.validate(), Status::Ok);
    assert_eq!(TrainingConfig::for_model(ModelKind::Cbow), config);
}

#[test]
fn c_validation_boundaries() {
    let mut vocab = VocabularyConfig::default();
    vocab.initial_capacity = 0;
    assert_eq!(vocab.validate(), Status::InvalidArgument);
    vocab = VocabularyConfig::default();
    vocab.hash_capacity = 1;
    assert_eq!(vocab.validate(), Status::InvalidArgument);
    vocab = VocabularyConfig::default();
    vocab.min_count = 0;
    assert_eq!(vocab.validate(), Status::InvalidArgument);

    let mut config = TrainingConfig::default();
    config.window_radius = usize::MAX;
    assert_eq!(config.validate(), Status::InvalidArgument);
    config = TrainingConfig::default();
    config.initial_learning_rate = f32::NAN;
    assert_eq!(config.validate(), Status::InvalidArgument);
    config = TrainingConfig::default();
    config.subsampling_threshold = -1.0;
    assert_eq!(config.validate(), Status::InvalidArgument);
    config = TrainingConfig::default();
    config.negative_sample_count = 0;
    assert_eq!(config.validate(), Status::InvalidArgument);
    config.objective_kind = ObjectiveKind::HierarchicalSoftmax;
    assert_eq!(config.validate(), Status::Ok);
}

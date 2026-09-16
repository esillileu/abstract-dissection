use std::{fs, sync::atomic::Ordering};
use w2v::{
    Corpus, EmbeddingKind, Model, ModelKind, ObjectiveKind, RngAlgorithm, Trainer, TrainingConfig,
    Vocabulary, VocabularyConfig, training::worker::Worker,
};

fn fixture() -> (Corpus, Vocabulary) {
    let path = std::env::temp_dir().join(format!(
        "w2v-stage6-{}-{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    fs::write(
        &path,
        b"alpha beta alpha gamma\nbeta alpha delta\ngamma beta alpha\n",
    )
    .unwrap();
    let corpus = Corpus::create(&path).unwrap();
    let vocab = Vocabulary::build(
        &corpus,
        &VocabularyConfig {
            initial_capacity: 2,
            hash_capacity: 17,
            min_count: 1,
        },
    )
    .unwrap();
    (corpus, vocab)
}

fn hash_float_bits(values: &[f32]) -> u64 {
    values
        .iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .fold(1469598103934665603u64, |hash, byte| {
            (hash ^ byte as u64).wrapping_mul(1099511628211)
        })
}

#[test]
fn c_single_thread_golden_cases() {
    let (corpus, vocab) = fixture();
    let cases = [
        (
            ModelKind::Cbow,
            ObjectiveKind::HierarchicalSoftmax,
            RngAlgorithm::Lcg,
            0x514f61880ff65c1f,
            0xef981cde17cb9eb2,
        ),
        (
            ModelKind::Cbow,
            ObjectiveKind::NegativeSampling,
            RngAlgorithm::Lcg,
            0x998a6cbb98dc7f6c,
            0x7b15e9239642b63f,
        ),
        (
            ModelKind::SkipGram,
            ObjectiveKind::HierarchicalSoftmax,
            RngAlgorithm::Lcg,
            0xd71dad0b1ef4a0cc,
            0x7af1d53846d24b44,
        ),
        (
            ModelKind::SkipGram,
            ObjectiveKind::NegativeSampling,
            RngAlgorithm::Lcg,
            0xfbe35d673fc7086c,
            0x84924104f81133f8,
        ),
        (
            ModelKind::Cbow,
            ObjectiveKind::NegativeSampling,
            RngAlgorithm::Xorshift,
            0xcbe69a7cc0673513,
            0x722439a66b1227d7,
        ),
    ];
    for (kind, objective, algorithm, expected_input, expected_output) in cases {
        let mut config = TrainingConfig::for_model(kind);
        config.objective_kind = objective;
        config.rng_algorithm = algorithm;
        config.embedding_dimension = 8;
        config.window_radius = 2;
        config.epochs = 2;
        config.thread_count = 1;
        config.subsampling_threshold = 0.0;
        config.negative_sample_count = 2;
        config.negative_table_size = 257;
        config.sigmoid_table_size = 101;
        let model = Model::create(&vocab, 8, config.root_seed, algorithm).unwrap();
        let trainer = Trainer::create(&corpus, &vocab, &model, &config).unwrap();
        trainer.train().unwrap();
        assert_eq!(trainer.processed_tokens(), 26);
        let mut snapshot = vec![0.0; vocab.entries.len() * 8];
        model.snapshot_into(EmbeddingKind::Input, &mut snapshot);
        assert_eq!(
            hash_float_bits(&snapshot),
            expected_input,
            "input {kind:?} {objective:?} {algorithm:?}"
        );
        model.snapshot_into(EmbeddingKind::Output, &mut snapshot);
        assert_eq!(
            hash_float_bits(&snapshot),
            expected_output,
            "output {kind:?} {objective:?} {algorithm:?}"
        );
    }
    fs::remove_file(&corpus.path).unwrap();
}

#[test]
fn c_learning_rate_interval() {
    let (corpus, vocab) = fixture();
    let config = TrainingConfig {
        embedding_dimension: 2,
        learning_rate_update_interval: 3,
        negative_table_size: 7,
        ..TrainingConfig::default()
    };
    let model = Model::create(&vocab, 2, config.root_seed, config.rng_algorithm).unwrap();
    let trainer = Trainer::create(&corpus, &vocab, &model, &config).unwrap();
    let mut worker = Worker::initialize(&trainer, 0).unwrap();
    for count in [2, 3] {
        worker.local_token_count = count;
        trainer.processed_tokens.store(count, Ordering::Relaxed);
        worker.update_learning_rate(&trainer);
        assert_eq!(worker.learning_rate, config.initial_learning_rate);
    }
    worker.local_token_count = 4;
    trainer.processed_tokens.store(4, Ordering::Relaxed);
    worker.update_learning_rate(&trainer);
    let updated = worker.learning_rate;
    assert!(updated < config.initial_learning_rate);
    assert_eq!(worker.last_learning_rate_update_count, 4);
    worker.local_token_count = 7;
    worker.update_learning_rate(&trainer);
    assert_eq!(worker.learning_rate, updated);
    worker.local_token_count = 8;
    trainer.processed_tokens.store(8, Ordering::Relaxed);
    worker.update_learning_rate(&trainer);
    assert!(worker.learning_rate < updated);
    fs::remove_file(&corpus.path).unwrap();
}

#[test]
fn subsampling_is_applied_after_counting_and_before_sentence_storage() {
    let (corpus, vocab) = fixture();
    let base = TrainingConfig {
        embedding_dimension: 2,
        thread_count: 1,
        negative_table_size: 7,
        subsampling_threshold: 0.0,
        ..TrainingConfig::default()
    };
    let model = Model::create(&vocab, 2, base.root_seed, base.rng_algorithm).unwrap();
    let disabled = Trainer::create(&corpus, &vocab, &model, &base).unwrap();
    let mut worker = Worker::initialize(&disabled, 0).unwrap();
    assert!(!worker.fill_sentence(&disabled).unwrap());
    assert_eq!(worker.sentence.len(), 4);
    assert_eq!(worker.local_token_count, 5); // 네 단어와 줄 경계
    assert_eq!(disabled.processed_tokens(), 5);
    let untouched_subsampling_state = worker.subsampling_rng.state;

    let enabled_config = TrainingConfig {
        subsampling_threshold: 1e-9,
        ..base
    };
    let enabled = Trainer::create(&corpus, &vocab, &model, &enabled_config).unwrap();
    let mut worker = Worker::initialize(&enabled, 0).unwrap();
    assert!(!worker.fill_sentence(&enabled).unwrap());
    assert!(worker.sentence.is_empty());
    assert_eq!(worker.local_token_count, 5);
    assert_eq!(enabled.processed_tokens(), 5);
    assert_ne!(worker.subsampling_rng.state, untouched_subsampling_state);
    fs::remove_file(&corpus.path).unwrap();
}

#[test]
fn parallel_training_completes_with_finite_embeddings() {
    let (corpus, vocab) = fixture();
    let config = TrainingConfig {
        embedding_dimension: 16,
        window_radius: 2,
        epochs: 8,
        thread_count: 4,
        subsampling_threshold: 0.0,
        negative_sample_count: 4,
        negative_table_size: 257,
        sigmoid_table_size: 101,
        ..TrainingConfig::default()
    };
    let model = Model::create(&vocab, 16, config.root_seed, config.rng_algorithm).unwrap();
    let trainer = Trainer::create(&corpus, &vocab, &model, &config).unwrap();
    trainer.train().unwrap();
    assert!(trainer.processed_tokens() > 0);
    let mut snapshot = vec![0.0; vocab.entries.len() * 16];
    model.snapshot_into(EmbeddingKind::Input, &mut snapshot);
    assert!(snapshot.iter().all(|value| value.is_finite()));
    fs::remove_file(&corpus.path).unwrap();
}

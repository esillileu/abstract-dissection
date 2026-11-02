use std::{fs, sync::Arc, time::Instant};
use w2v::{
    Corpus, DownpourConfig, ObjectiveKind, RngAlgorithm, Vocabulary, VocabularyConfig,
    nnlm::{HiddenActivation, NnlmDownpourTrainingSession, NnlmTrainingConfig},
};

#[test]
#[ignore]
fn profile_downpour_stages() {
    let path = std::env::temp_dir().join(format!("nnlm-profile-{}", std::process::id()));
    let words: Vec<String> = (0..1000).map(|i| format!("word_{i}")).collect();
    let mut text = String::new();
    for _ in 0..100 {
        text.push_str(&words[..260].join(" "));
        text.push('\n');
    }
    fs::write(&path, text.as_bytes()).unwrap();

    let corpus = Arc::new(Corpus::create(path).unwrap());
    let vocab = Arc::new(
        Vocabulary::build(
            &corpus,
            &VocabularyConfig {
                initial_capacity: 1000,
                hash_capacity: 30000,
                min_count: 1,
                max_lexical_words: 2000,
            },
        )
        .unwrap(),
    );

    // Profile 1 worker thread
    let config = NnlmTrainingConfig {
        embedding_dimension: 640,
        history_length: 8,
        hidden_dimension: 640,
        hidden_activation: HiddenActivation::Tanh,
        epochs: 1,
        thread_count: 1,
        objective_kind: ObjectiveKind::HierarchicalSoftmax,
        initial_learning_rate: 0.05,
        observation_interval: 10000,
        root_seed: 1,
        rng_algorithm: RngAlgorithm::Lcg,
    };
    let downpour = DownpourConfig {
        parameter_server_shards: 4,
        mini_batch_targets: 250,
        queue_capacity: 128,
        ..DownpourConfig::default()
    };

    println!("\n=== Profiling 1 Worker Thread (Table 3 Dimensions: D=640, H=640, N=8, B=250) ===");
    let mut session = NnlmDownpourTrainingSession::create(
        corpus.clone(),
        vocab.clone(),
        config.clone(),
        downpour.clone(),
        None,
    )
    .unwrap();
    let start = Instant::now();
    let report = session.train_epoch().unwrap();
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "Processed {} tokens in {:.3}s -> {:.1} tokens/s",
        report.epoch_tokens,
        elapsed,
        report.epoch_tokens as f64 / elapsed
    );

    // Profile 20 worker threads
    println!("\n=== Profiling 20 Worker Threads ===");
    let mut config_20 = config.clone();
    config_20.thread_count = 20;
    let mut session_20 = NnlmDownpourTrainingSession::create(
        corpus.clone(),
        vocab.clone(),
        config_20,
        downpour.clone(),
        None,
    )
    .unwrap();
    let start_20 = Instant::now();
    let report_20 = session_20.train_epoch().unwrap();
    let elapsed_20 = start_20.elapsed().as_secs_f64();
    println!(
        "Processed {} tokens in {:.3}s -> {:.1} tokens/s",
        report_20.epoch_tokens,
        elapsed_20,
        report_20.epoch_tokens as f64 / elapsed_20
    );
}

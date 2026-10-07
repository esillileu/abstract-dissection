use std::{fs, sync::Arc};
use w2v::{
    Corpus, DownpourConfig, ObjectiveKind, RngAlgorithm, Status, Vocabulary, VocabularyConfig,
    nnlm::{
        HiddenActivation, NnlmDownpourTrainingSession, NnlmModel, NnlmTrainingConfig,
        downpour::{ParameterKind, ParameterServerShard, ReplicaCache, SharedParameters},
        training::forward_backward,
    },
};

fn fixture() -> (
    Arc<Corpus>,
    Arc<Vocabulary>,
    NnlmTrainingConfig,
    DownpourConfig,
) {
    let path = std::env::temp_dir().join(format!(
        "nnlm-downpour-{}-{}",
        std::process::id(),
        std::thread::current().name().unwrap()
    ));
    fs::write(&path, b"a b a c d a b\nb a c b d a").unwrap();
    let corpus = Arc::new(Corpus::create(path).unwrap());
    let vocab = Arc::new(
        Vocabulary::build(
            &corpus,
            &VocabularyConfig {
                initial_capacity: 8,
                hash_capacity: 31,
                min_count: 1,
                max_lexical_words: 0,
            },
        )
        .unwrap(),
    );
    let config = NnlmTrainingConfig {
        embedding_dimension: 2,
        history_length: 2,
        hidden_dimension: 3,
        hidden_activation: HiddenActivation::Tanh,
        epochs: 3,
        thread_count: 1,
        objective_kind: ObjectiveKind::HierarchicalSoftmax,
        initial_learning_rate: 0.1,
        observation_interval: 1,
        root_seed: 19,
        rng_algorithm: RngAlgorithm::Lcg,
    };
    let downpour = DownpourConfig {
        parameter_server_shards: 3,
        mini_batch_targets: 2,
        queue_capacity: 1,
        ..DownpourConfig::default()
    };
    (corpus, vocab, config, downpour)
}

#[test]
fn adagrad_known_values_for_all_four_parameter_kinds() {
    let (_, vocab, config, _) = fixture();
    let mut model = NnlmModel::create(&vocab, &config).unwrap();
    model.input_embeddings.fill(0.0);
    model.hidden_weights.fill(0.0);
    model.hidden_bias.fill(0.0);
    model.output_weights.fill(0.0);
    let shared = Arc::new(SharedParameters::new(&model));
    let mut server = ParameterServerShard::new(0, 1, shared.clone(), 0.1, 1e-6).unwrap();
    for (kind, width) in [
        (ParameterKind::Input, 2),
        (ParameterKind::Hidden, 4),
        (ParameterKind::Bias, 3),
        (ParameterKind::Output, 3),
    ] {
        server.apply_gradient(kind, 0, &vec![2.0; width]).unwrap();
        server.apply_gradient(kind, 0, &vec![-1.0; width]).unwrap();
        let actual = shared.snapshot();
        let values = match kind {
            ParameterKind::Input => actual.input_embeddings,
            ParameterKind::Hidden => actual.hidden_weights,
            ParameterKind::Bias => actual.hidden_bias,
            ParameterKind::Output => actual.output_weights,
        };
        let expected = -0.1 * 2.0 / (4.0f32 + 1e-6).sqrt() + 0.1 / (5.0f32 + 1e-6).sqrt();
        for value in &values[..width] {
            assert_eq!(*value, expected);
        }
        assert!(values[width..].iter().all(|v| *v == 0.0));
    }
}

#[test]
fn sparse_replica_retains_stale_batch_snapshot() {
    let (_, vocab, config, _) = fixture();
    let model = NnlmModel::create(&vocab, &config).unwrap();
    let shared = Arc::new(SharedParameters::new(&model));
    let mut server = ParameterServerShard::new(0, 1, shared.clone(), 0.1, 1e-6).unwrap();
    let mut cache = ReplicaCache::new(shared.clone());
    let target = vocab.find(b"d").unwrap();
    let before = cache
        .gradient(&vocab, config.hidden_activation, &[1, 1], target)
        .unwrap();
    assert_eq!(
        cache.cached_rows(),
        (1, vocab.entries[target].huffman_path.len())
    );
    let node = vocab.entries[target].huffman_path[0];
    server
        .apply_gradient(ParameterKind::Output, node, &[0.5, 0.2, -0.3])
        .unwrap();
    let stale = cache
        .gradient(&vocab, config.hidden_activation, &[1, 1], target)
        .unwrap();
    let fresh = ReplicaCache::new(shared)
        .gradient(&vocab, config.hidden_activation, &[1, 1], target)
        .unwrap();
    assert_eq!(stale.loss, before.loss);
    assert_eq!(stale.output, before.output);
    assert_ne!(fresh.loss, before.loss);
}

#[test]
fn mini_batch_sums_derivatives_before_adagrad_squares_them() {
    let (corpus, vocab, config, mut downpour) = fixture();
    downpour.mini_batch_targets = 100;
    let model = NnlmModel::create(&vocab, &config).unwrap();
    let mut sum = vec![0.0; model.output_weights.len()];
    for sentence in [b"a b a c d a b".as_slice(), b"b a c b d a".as_slice()] {
        let ids: Vec<_> = sentence
            .split(|byte| *byte == b' ')
            .map(|word| vocab.find(word).unwrap())
            .collect();
        for window in ids.windows(3) {
            let gradient = forward_backward(
                &model,
                &vocab,
                config.hidden_activation,
                &window[..2],
                window[2],
            )
            .unwrap();
            for (node, row) in gradient.output {
                for (j, value) in row.iter().enumerate() {
                    sum[node * 3 + j] += value;
                }
            }
        }
    }
    let mut session =
        NnlmDownpourTrainingSession::create(corpus, vocab, config, downpour.clone(), None).unwrap();
    session.train_epoch().unwrap();
    let state = session.export_state().unwrap();
    for ((actual, accum), g) in state
        .training
        .model
        .output_weights
        .iter()
        .zip(&state.adagrad.output)
        .zip(sum)
    {
        assert_eq!(*accum, g * g);
        assert_eq!(
            *actual,
            -downpour.adagrad_gamma * g / (g * g + downpour.adagrad_epsilon).sqrt()
        );
    }
    assert_eq!(state.replicas[0].batch_count, 1);
}

#[test]
fn single_replica_resume_and_shard_counts_are_bit_identical() {
    let (corpus, vocab, config, downpour) = fixture();
    let mut continuous = NnlmDownpourTrainingSession::create(
        corpus.clone(),
        vocab.clone(),
        config.clone(),
        downpour.clone(),
        None,
    )
    .unwrap();
    continuous.train_epoch().unwrap();
    let checkpoint = continuous.export_state().unwrap();
    let mut resumed = NnlmDownpourTrainingSession::restore(
        corpus.clone(),
        vocab.clone(),
        config.clone(),
        downpour.clone(),
        &checkpoint,
        None,
    )
    .unwrap();
    while !continuous.is_complete() {
        continuous.train_epoch().unwrap();
        resumed.train_epoch().unwrap();
    }
    let expected = continuous.export_state().unwrap();
    assert_eq!(resumed.export_state().unwrap(), expected);
    for accum in [
        &expected.adagrad.input,
        &expected.adagrad.hidden,
        &expected.adagrad.bias,
        &expected.adagrad.output,
    ] {
        assert!(accum.iter().all(|v| v.is_finite() && *v >= 0.0));
        assert!(accum.iter().any(|v| *v > 0.0));
    }
    for shards in [1, 7] {
        let mut alternative = downpour.clone();
        alternative.parameter_server_shards = shards;
        let mut session = NnlmDownpourTrainingSession::create(
            corpus.clone(),
            vocab.clone(),
            config.clone(),
            alternative,
            None,
        )
        .unwrap();
        while !session.is_complete() {
            session.train_epoch().unwrap();
        }
        let state = session.export_state().unwrap();
        assert_eq!(state.training.model, expected.training.model);
        assert_eq!(state.adagrad, expected.adagrad);
    }
}

#[test]
fn multiple_replicas_partition_sentences_without_lost_or_duplicate_tokens() {
    let (corpus, vocab, mut config, mut downpour) = fixture();
    downpour.parameter_server_shards = 7;
    for workers in [2, 4, 30] {
        config.thread_count = workers;
        let mut session = NnlmDownpourTrainingSession::create(
            corpus.clone(),
            vocab.clone(),
            config.clone(),
            downpour.clone(),
            None,
        )
        .unwrap();
        for _ in 0..config.epochs {
            let report = session.train_epoch().unwrap();
            assert_eq!(report.epoch_tokens, 13);
            assert_eq!(report.objective_loss_count, 9);
            assert!(report.objective_loss_sum.is_finite());
        }
        let state = session.export_state().unwrap();
        assert_eq!(state.training.processed_tokens, 39);
        let mut resumed = NnlmDownpourTrainingSession::restore(
            corpus.clone(),
            vocab.clone(),
            config.clone(),
            downpour.clone(),
            &state,
            None,
        )
        .unwrap();
        assert!(resumed.is_complete());
        assert_eq!(resumed.train_epoch(), Err(Status::InvalidState));
    }
}

#[test]
fn resume_rejects_wrong_optimizer_identity_shape_and_worker_state() {
    let (corpus, vocab, config, downpour) = fixture();
    let mut session = NnlmDownpourTrainingSession::create(
        corpus.clone(),
        vocab.clone(),
        config.clone(),
        downpour.clone(),
        None,
    )
    .unwrap();
    session.train_epoch().unwrap();
    let state = session.export_state().unwrap();
    let mut changed = downpour.clone();
    changed.adagrad_epsilon *= 2.0;
    assert!(matches!(
        NnlmDownpourTrainingSession::restore(
            corpus.clone(),
            vocab.clone(),
            config.clone(),
            changed,
            &state,
            None
        ),
        Err(Status::IdentityMismatch)
    ));
    for change in 0..5 {
        let mut corrupt = state.clone();
        match change {
            0 => corrupt.adagrad.hidden.pop().map(|_| ()).unwrap(),
            1 => corrupt.adagrad.bias[0] = -1.0,
            2 => corrupt.replicas[0].end_byte -= 1,
            3 => corrupt.replicas[0].processed_tokens += 1,
            _ => corrupt.training.model.hidden_weights[0] = f32::NAN,
        }
        assert!(matches!(
            NnlmDownpourTrainingSession::restore(
                corpus.clone(),
                vocab.clone(),
                config.clone(),
                downpour.clone(),
                &corrupt,
                None
            ),
            Err(Status::InvalidState)
        ));
    }
}

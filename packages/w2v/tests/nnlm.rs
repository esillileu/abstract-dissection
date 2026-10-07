use std::{fs, sync::Arc};
use w2v::{
    Corpus, ObjectiveKind, RngAlgorithm, Status, Vocabulary, VocabularyConfig,
    nnlm::{
        HiddenActivation, NnlmModel, NnlmTrainingConfig, NnlmTrainingSession,
        training::{apply_sgd, forward_backward},
    },
};

fn fixture() -> (Arc<Corpus>, Arc<Vocabulary>, NnlmTrainingConfig) {
    let path = std::env::temp_dir().join(format!(
        "nnlm-{}-{}",
        std::process::id(),
        std::thread::current().name().unwrap()
    ));
    fs::write(&path, b"a b a c d a b\nb a c b d a").unwrap();
    let corpus = Arc::new(Corpus::create(path).unwrap());
    let vocabulary = Arc::new(
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
        epochs: 2,
        thread_count: 1,
        objective_kind: ObjectiveKind::HierarchicalSoftmax,
        initial_learning_rate: 0.1,
        observation_interval: 1,
        root_seed: 19,
        rng_algorithm: RngAlgorithm::Lcg,
    };
    (corpus, vocabulary, config)
}

fn nonzero_model(vocab: &Vocabulary, config: &NnlmTrainingConfig) -> NnlmModel {
    let mut model = NnlmModel::create(vocab, config).unwrap();
    for (i, value) in model.output_weights.iter_mut().enumerate() {
        *value = (i as f32 - 3.0) * 0.03;
    }
    model
}

#[test]
fn numerical_gradients_cover_every_parameter_and_repeated_history() {
    let (_, vocab, mut config) = fixture();
    for activation in [HiddenActivation::Tanh, HiddenActivation::Sigmoid] {
        config.hidden_activation = activation;
        let model = nonzero_model(&vocab, &config);
        let history = [vocab.find(b"a").unwrap(); 2];
        let target = vocab.find(b"d").unwrap();
        let gradient = forward_backward(&model, &vocab, activation, &history, target).unwrap();
        assert_eq!(gradient.embeddings.len(), 1);
        for kind in 0..4 {
            let values = match kind {
                0 => &model.input_embeddings,
                1 => &model.hidden_weights,
                2 => &model.hidden_bias,
                _ => &model.output_weights,
            };
            for index in 0..values.len() {
                let mut plus = model.clone();
                let mut minus = model.clone();
                let epsilon = 0.002;
                for (candidate, delta) in [(&mut plus, epsilon), (&mut minus, -epsilon)] {
                    let parameters = match kind {
                        0 => &mut candidate.input_embeddings,
                        1 => &mut candidate.hidden_weights,
                        2 => &mut candidate.hidden_bias,
                        _ => &mut candidate.output_weights,
                    };
                    parameters[index] += delta;
                }
                let numeric = (forward_backward(&plus, &vocab, activation, &history, target)
                    .unwrap()
                    .loss
                    - forward_backward(&minus, &vocab, activation, &history, target)
                        .unwrap()
                        .loss)
                    / (2.0 * epsilon as f64);
                let analytic = match kind {
                    0 => gradient
                        .embeddings
                        .get(&(index / 2))
                        .map_or(0.0, |row| row[index % 2]),
                    1 => gradient.hidden_weights[index],
                    2 => gradient.hidden_bias[index],
                    _ => gradient
                        .output
                        .get(&(index / 3))
                        .map_or(0.0, |row| row[index % 3]),
                } as f64;
                assert!(
                    (numeric - analytic).abs() < 0.0001 + 0.003 * analytic.abs(),
                    "{activation:?} parameter {kind}:{index}: numeric={numeric}, analytic={analytic}"
                );
            }
        }
    }
}

#[test]
fn only_target_huffman_nodes_are_updated() {
    let (_, vocab, config) = fixture();
    let mut model = nonzero_model(&vocab, &config);
    let target = vocab.find(b"a").unwrap();
    let before = model.output_weights.clone();
    let gradient =
        forward_backward(&model, &vocab, config.hidden_activation, &[1, 2], target).unwrap();
    apply_sgd(&mut model, &gradient, 0.1);
    let path = &vocab.entries[target].huffman_path;
    assert!(path.len() < model.vocab_size - 1);
    for node in 0..model.vocab_size - 1 {
        let range = node * model.hidden_dimension..(node + 1) * model.hidden_dimension;
        if !path.contains(&node) {
            assert_eq!(&model.output_weights[range.clone()], &before[range]);
        } else {
            assert_ne!(&model.output_weights[range.clone()], &before[range]);
        }
    }
}

#[test]
fn repeated_word_derivatives_equal_sum_of_position_derivatives() {
    let (_, vocab, config) = fixture();
    let mut model = nonzero_model(&vocab, &config);
    let a = vocab.find(b"a").unwrap();
    let b = vocab.find(b"b").unwrap();
    let a_row = model.input_embeddings[a * 2..a * 2 + 2].to_vec();
    model.input_embeddings[b * 2..b * 2 + 2].copy_from_slice(&a_row);
    let repeated = forward_backward(&model, &vocab, config.hidden_activation, &[a, a], 3).unwrap();
    let separate = forward_backward(&model, &vocab, config.hidden_activation, &[a, b], 3).unwrap();
    for k in 0..2 {
        assert_eq!(
            repeated.embeddings[&a][k],
            separate.embeddings[&a][k] + separate.embeddings[&b][k]
        );
    }
}

#[test]
fn single_thread_seed_is_bit_deterministic_and_observation_does_not_change_weights() {
    let (corpus, vocab, config) = fixture();
    let mut a =
        NnlmTrainingSession::create(corpus.clone(), vocab.clone(), config.clone(), None).unwrap();
    let mut b =
        NnlmTrainingSession::create(corpus.clone(), vocab.clone(), config.clone(), None).unwrap();
    let mut unobserved = config.clone();
    unobserved.observation_interval = 0;
    let mut c = NnlmTrainingSession::create(corpus, vocab, unobserved, None).unwrap();
    while !a.is_complete() {
        let report = a.train_epoch().unwrap();
        b.train_epoch().unwrap();
        c.train_epoch().unwrap();
        assert!(report.objective_loss_sum.is_finite());
        assert_eq!(report.epoch_tokens, 13);
        assert_eq!(report.objective_loss_count, 9);
        assert_eq!(a.state, b.state);
        assert_eq!(a.state.model, c.state.model);
        assert_eq!(a.state.processed_tokens, c.state.processed_tokens);
    }
    assert_eq!(a.train_epoch(), Err(Status::InvalidState));
    assert!(a.state.model.output_weights.iter().any(|x| *x != 0.0));
    assert!(
        a.state.model.hidden_weights
            != NnlmModel::create(&a.vocabulary, &config)
                .unwrap()
                .hidden_weights
    );
}

#[test]
fn unsupported_config_and_invalid_inputs_are_rejected() {
    let (corpus, vocab, mut config) = fixture();
    config.thread_count = 2;
    assert!(NnlmTrainingSession::create(corpus, vocab.clone(), config.clone(), None).is_err());
    config.thread_count = 1;
    let model = NnlmModel::create(&vocab, &config).unwrap();
    assert!(forward_backward(&model, &vocab, config.hidden_activation, &[1], 2).is_err());
    config.objective_kind = ObjectiveKind::NegativeSampling;
    assert_eq!(config.validate(), Status::InvalidArgument);
    config.objective_kind = ObjectiveKind::HierarchicalSoftmax;
    config.history_length = usize::MAX;
    assert_eq!(config.validate(), Status::InvalidArgument);
}

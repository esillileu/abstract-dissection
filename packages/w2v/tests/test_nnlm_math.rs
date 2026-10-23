use std::sync::Arc;
use w2v::{
    Corpus, DownpourConfig, ObjectiveKind, RngAlgorithm, Vocabulary, VocabularyConfig,
    config::Real,
    nnlm::{
        HiddenActivation, NnlmModel, NnlmTrainingConfig,
        training::forward_backward,
    },
};

fn create_small_model() -> (
    Arc<Corpus>,
    Arc<Vocabulary>,
    NnlmTrainingConfig,
    DownpourConfig,
    NnlmModel,
) {
    let path = std::env::temp_dir().join(format!(
        "nnlm-test-grad-{}-{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("t")
    ));
    std::fs::write(&path, b"the quick brown fox jumps over the lazy dog\na quick dog jumps\n").unwrap();
    let corpus = Arc::new(Corpus::create(path).unwrap());
    let vocab = Arc::new(
        Vocabulary::build(
            &corpus,
            &VocabularyConfig {
                initial_capacity: 16,
                hash_capacity: 31,
                min_count: 1,
                max_lexical_words: 0,
            },
        )
        .unwrap(),
    );
    let config = NnlmTrainingConfig {
        embedding_dimension: 4,
        history_length: 2,
        hidden_dimension: 5,
        hidden_activation: HiddenActivation::Tanh,
        epochs: 1,
        thread_count: 1,
        objective_kind: ObjectiveKind::HierarchicalSoftmax,
        initial_learning_rate: 0.05,
        observation_interval: 0,
        root_seed: 42,
        rng_algorithm: RngAlgorithm::Lcg,
    };
    let downpour = DownpourConfig {
        parameter_server_shards: 1,
        mini_batch_targets: 4,
        adagrad_gamma: 0.05,
        adagrad_epsilon: 1e-6,
        queue_capacity: 16,
        ..DownpourConfig::default()
    };
    let model = NnlmModel::create(&vocab, &config).unwrap();
    (corpus, vocab, config, downpour, model)
}

#[test]
fn test_gradient_against_finite_differences() {
    let (_, vocab, config, _, mut model) = create_small_model();
    let history = [1, 2];
    let target = 3;
    let eps = 1e-3 as Real;

    // Baseline analytical gradient
    let grad = forward_backward(&model, &vocab, config.hidden_activation, &history, target).unwrap();

    // 1. Check hidden bias gradient
    for j in 0..model.hidden_dimension {
        let orig = model.hidden_bias[j];
        model.hidden_bias[j] = orig + eps;
        let loss_plus = forward_backward(&model, &vocab, config.hidden_activation, &history, target).unwrap().loss;
        model.hidden_bias[j] = orig - eps;
        let loss_minus = forward_backward(&model, &vocab, config.hidden_activation, &history, target).unwrap().loss;
        model.hidden_bias[j] = orig;

        let num_grad = ((loss_plus - loss_minus) / (2.0 * eps as f64)) as Real;
        let ana_grad = grad.hidden_bias[j];
        let diff = (num_grad - ana_grad).abs();
        assert!(
            diff < 1e-3,
            "hidden_bias[{}] diff {} too large: num={}, ana={}",
            j, diff, num_grad, ana_grad
        );
    }

    // 2. Check hidden weights gradient
    let _p = config.history_length * config.embedding_dimension;
    let ana_hw = grad.hidden_weights();
    for idx in 0..model.hidden_weights.len() {
        let orig = model.hidden_weights[idx];
        model.hidden_weights[idx] = orig + eps;
        let loss_plus = forward_backward(&model, &vocab, config.hidden_activation, &history, target).unwrap().loss;
        model.hidden_weights[idx] = orig - eps;
        let loss_minus = forward_backward(&model, &vocab, config.hidden_activation, &history, target).unwrap().loss;
        model.hidden_weights[idx] = orig;

        let num_grad = ((loss_plus - loss_minus) / (2.0 * eps as f64)) as Real;
        let ana_grad = ana_hw[idx];
        let diff = (num_grad - ana_grad).abs();
        assert!(
            diff < 1e-3,
            "hidden_weights[{}] diff {} too large: num={}, ana={}",
            idx, diff, num_grad, ana_grad
        );
    }

    // 3. Check output weights gradient
    for (&node, ana_row) in &grad.output {
        for j in 0..config.hidden_dimension {
            let idx = node * config.hidden_dimension + j;
            let orig = model.output_weights[idx];
            model.output_weights[idx] = orig + eps;
            let loss_plus = forward_backward(&model, &vocab, config.hidden_activation, &history, target).unwrap().loss;
            model.output_weights[idx] = orig - eps;
            let loss_minus = forward_backward(&model, &vocab, config.hidden_activation, &history, target).unwrap().loss;
            model.output_weights[idx] = orig;

            let num_grad = ((loss_plus - loss_minus) / (2.0 * eps as f64)) as Real;
            let ana_grad = ana_row[j];
            let diff = (num_grad - ana_grad).abs();
            assert!(
                diff < 1e-3,
                "output_weights node {} j {} diff {} too large: num={}, ana={}",
                node, j, diff, num_grad, ana_grad
            );
        }
    }

    // 4. Check input embeddings gradient
    for (&w_id, ana_row) in &grad.embeddings {
        for k in 0..config.embedding_dimension {
            let idx = w_id * config.embedding_dimension + k;
            let orig = model.input_embeddings[idx];
            model.input_embeddings[idx] = orig + eps;
            let loss_plus = forward_backward(&model, &vocab, config.hidden_activation, &history, target).unwrap().loss;
            model.input_embeddings[idx] = orig - eps;
            let loss_minus = forward_backward(&model, &vocab, config.hidden_activation, &history, target).unwrap().loss;
            model.input_embeddings[idx] = orig;

            let num_grad = ((loss_plus - loss_minus) / (2.0 * eps as f64)) as Real;
            let ana_grad = ana_row[k];
            let diff = (num_grad - ana_grad).abs();
            assert!(
                diff < 1e-3,
                "input_embeddings w_id {} k {} diff {} too large: num={}, ana={}",
                w_id, k, diff, num_grad, ana_grad
            );
        }
    }
}

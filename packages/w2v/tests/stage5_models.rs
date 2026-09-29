use std::{
    path::PathBuf,
    sync::{Arc, atomic::AtomicU64},
};
use w2v::{
    Corpus, EmbeddingKind, HsOutOfRangePolicy, Model, ModelKind, NegativeSampler, ObjectiveKind,
    RngAlgorithm, SigmoidTable, Trainer, TrainingConfig, Vocabulary, VocabularyEntry, atomic_float,
    random::Rng,
    training::{
        ModelStep, cbow_train, hierarchical_softmax_train, negative_sampling_train, skip_gram_train,
    },
};

fn fixture_vocab() -> Vocabulary {
    let tokens: [&[u8]; 5] = [b"</s>", b"alpha", b"beta", b"gamma", b"delta"];
    let counts = [3, 4, 3, 2, 1];
    let mut vocab = Vocabulary {
        entries: tokens
            .into_iter()
            .zip(counts)
            .map(|(token, count)| VocabularyEntry {
                token: token.to_vec(),
                count,
                huffman_path: Vec::new(),
                huffman_bits: Vec::new(),
            })
            .collect(),
        hash_slots: vec![None; 17],
        retained_token_count: 13,
    };
    assert_eq!(vocab.assign_huffman(), w2v::Status::Ok);
    vocab
}

struct StepOutput {
    input: Vec<f32>,
    output: Vec<f32>,
    window_state: u64,
    negative_state: u64,
}
fn run_fixed_step(kind: ModelKind, objective: ObjectiveKind) -> StepOutput {
    let corpus = Arc::new(Corpus {
        path: PathBuf::new(),
        byte_size: 0,
    });
    let vocab = Arc::new(fixture_vocab());
    let model = Arc::new(Model::create(&vocab, 2, 1, RngAlgorithm::Lcg).unwrap());
    let mut config = TrainingConfig::for_model(kind);
    config.embedding_dimension = 2;
    config.objective_kind = objective;
    config.window_radius = 1;
    config.negative_sample_count = 1;
    config.negative_table_size = 7;
    config.sigmoid_table_size = 3;
    config.subsampling_threshold = 0.0;
    config.initial_learning_rate = 0.05;
    let sigmoid_table = SigmoidTable {
        values: vec![0.5; 3],
        max: 6.0,
    };
    let mut negative_sampler = if objective == ObjectiveKind::NegativeSampling {
        NegativeSampler { table: vec![2; 7] }
    } else {
        NegativeSampler::default()
    };
    if kind == ModelKind::SkipGram && objective == ObjectiveKind::NegativeSampling {
        negative_sampler.table[0] = 1;
        negative_sampler.table[4] = 3;
    }
    let trainer = Trainer {
        corpus: Arc::clone(&corpus),
        corpus_digest: String::new(),
        vocab: Arc::clone(&vocab),
        model: Arc::clone(&model),
        negative_sampler,
        sigmoid_table,
        config,
        processed_tokens: AtomicU64::new(0),
    };
    for coordinate in model
        .input_embeddings
        .iter()
        .chain(&model.output_embeddings)
    {
        atomic_float::store(coordinate, 0.0);
    }
    if kind == ModelKind::Cbow {
        for (index, value) in [(2, 0.25), (3, -0.5), (6, 0.75), (7, 0.5)] {
            atomic_float::store(&model.input_embeddings[index], value);
        }
        let output_start = if objective == ObjectiveKind::HierarchicalSoftmax {
            6
        } else {
            4
        };
        atomic_float::store(&model.output_embeddings[output_start], 0.2);
        atomic_float::store(&model.output_embeddings[output_start + 1], -0.4);
    } else {
        atomic_float::store(&model.input_embeddings[4], 0.5);
        atomic_float::store(&model.input_embeddings[5], -0.25);
        for (index, value) in [(2, 0.2), (3, -0.4), (6, 0.6), (7, 0.3)] {
            atomic_float::store(&model.output_embeddings[index], value);
        }
    }
    let sentence = [1, 2, 3];
    let mut hidden = [0.0; 2];
    let mut hidden_gradient = [0.0; 2];
    let mut output_snapshot = [0.0; 2];
    let mut window_rng = Rng::new(1, RngAlgorithm::Lcg);
    let mut negative_rng = Rng::new(1, RngAlgorithm::Lcg);
    let mut step = ModelStep {
        trainer: &trainer,
        target_token: 2,
        learning_rate: 0.05,
        sentence: &sentence,
        sentence_position: 1,
        hidden: &mut hidden,
        hidden_gradient: &mut hidden_gradient,
        output_snapshot: &mut output_snapshot,
        window_rng: &mut window_rng,
        negative_rng: &mut negative_rng,
        observe_objective: false,
    };
    if kind == ModelKind::Cbow {
        cbow_train(&mut step).unwrap();
    } else {
        skip_gram_train(&mut step).unwrap();
    }
    let mut input = vec![0.0; 10];
    let mut output = vec![0.0; 10];
    assert_eq!(
        model.snapshot_into(EmbeddingKind::Input, &mut input),
        w2v::Status::Ok
    );
    assert_eq!(
        model.snapshot_into(EmbeddingKind::Output, &mut output),
        w2v::Status::Ok
    );
    StepOutput {
        input,
        output,
        window_state: window_rng.state,
        negative_state: negative_rng.state,
    }
}

#[test]
fn c_fixed_cbow_hs_and_negative() {
    for objective in [
        ObjectiveKind::HierarchicalSoftmax,
        ObjectiveKind::NegativeSampling,
    ] {
        let result = run_fixed_step(ModelKind::Cbow, objective);
        assert_eq!(&result.input[2..4], &[0.255, -0.51]);
        assert_eq!(&result.input[6..8], &[0.755, 0.49]);
        let first = if objective == ObjectiveKind::HierarchicalSoftmax {
            6
        } else {
            4
        };
        assert_eq!(&result.output[first..first + 2], &[0.2125, -0.4]);
        if objective == ObjectiveKind::HierarchicalSoftmax {
            assert_eq!(result.output[2], -0.0125);
            assert_eq!(result.negative_state, 1);
        } else {
            assert_eq!(result.negative_state, 25214903928);
        }
        assert_eq!(result.window_state, 25214903928);
    }
}

#[test]
fn c_fixed_skip_gram_hs_and_negative() {
    let hs = run_fixed_step(ModelKind::SkipGram, ObjectiveKind::HierarchicalSoftmax);
    assert_eq!(hs.input[4].to_bits(), 0x3f013333);
    assert_eq!(hs.input[5].to_bits(), 0xbe850a3d);
    assert_eq!(
        hs.output[..8]
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        [
            0xbc46a7f0, 0x3bd2f1aa, 0x3e59374c, 0xbed01894, 0x3c4ccccd, 0xbbcccccd, 0x3f198107,
            0x3e998106
        ]
    );
    assert_eq!(hs.window_state, 25214903928);
    assert_eq!(hs.negative_state, 1);

    let negative = run_fixed_step(ModelKind::SkipGram, ObjectiveKind::NegativeSampling);
    assert_eq!(&negative.input[4..6], &[0.52, -0.2525]);
    assert_eq!(&negative.output[2..4], &[0.2125, -0.40625]);
    assert_eq!(&negative.output[6..8], &[0.612625, 0.2935]);
    assert_eq!(negative.window_state, 25214903928);
    assert_eq!(negative.negative_state, 8602081314781131043);
}

#[test]
fn c_hs_boundary_policy() {
    let corpus = Arc::new(Corpus {
        path: PathBuf::new(),
        byte_size: 0,
    });
    let vocab = Arc::new(fixture_vocab());
    let model = Arc::new(Model::create(&vocab, 2, 1, RngAlgorithm::Lcg).unwrap());
    let config = TrainingConfig {
        embedding_dimension: 2,
        objective_kind: ObjectiveKind::HierarchicalSoftmax,
        ..TrainingConfig::default()
    };
    let trainer = Trainer {
        corpus: Arc::clone(&corpus),
        corpus_digest: String::new(),
        vocab: Arc::clone(&vocab),
        model: Arc::clone(&model),
        negative_sampler: NegativeSampler::default(),
        sigmoid_table: SigmoidTable::initialize(1000, 6.0).unwrap(),
        config,
        processed_tokens: AtomicU64::new(0),
    };
    let entry = &vocab.entries[1];
    let sign = if entry.huffman_bits[0] == 0 {
        -1.0
    } else {
        1.0
    };
    let boundary = sign * trainer.sigmoid_table.max;
    let output_index = entry.huffman_path[0] * 2;
    atomic_float::store(&model.output_embeddings[output_index], boundary);
    let hidden = [1.0, 0.0];
    let mut gradient = [0.0, 0.0];
    hierarchical_softmax_train(&trainer, 1, 0.05, &hidden, &mut gradient, false).unwrap();
    assert_eq!(
        atomic_float::load(&model.output_embeddings[output_index]),
        boundary
    );
    let mut boundary_trainer = trainer;
    boundary_trainer.config.hs_out_of_range_policy = HsOutOfRangePolicy::UseBoundaryValue;
    hierarchical_softmax_train(&boundary_trainer, 1, 0.05, &hidden, &mut gradient, false).unwrap();
    assert_ne!(
        atomic_float::load(&model.output_embeddings[output_index]),
        boundary
    );
}

#[test]
fn c_negative_boundary_fallback() {
    let corpus = Arc::new(Corpus {
        path: PathBuf::new(),
        byte_size: 0,
    });
    let vocab = Arc::new(fixture_vocab());
    let model = Arc::new(Model::create(&vocab, 2, 1, RngAlgorithm::Lcg).unwrap());
    for coordinate in &model.output_embeddings {
        atomic_float::store(coordinate, 0.0);
    }
    let config = TrainingConfig {
        embedding_dimension: 2,
        negative_sample_count: 1,
        ..TrainingConfig::default()
    };
    let trainer = Trainer {
        corpus: Arc::clone(&corpus),
        corpus_digest: String::new(),
        vocab: Arc::clone(&vocab),
        model: Arc::clone(&model),
        negative_sampler: NegativeSampler { table: vec![0; 7] },
        sigmoid_table: SigmoidTable {
            values: vec![0.5; 3],
            max: 6.0,
        },
        config,
        processed_tokens: AtomicU64::new(0),
    };
    let mut rng = Rng::new(1, RngAlgorithm::Lcg);
    let mut gradient = [0.0; 2];
    negative_sampling_train(
        &trainer,
        2,
        0.05,
        &mut rng,
        &[0.25, -0.5],
        &mut gradient,
        false,
    )
    .unwrap();
    assert_eq!(rng.state, 25214903928);
    assert_eq!(
        model.output_embeddings[2..4]
            .iter()
            .map(|coordinate| atomic_float::load(coordinate).to_bits())
            .collect::<Vec<_>>(),
        [0xbbcccccd, 0x3c4ccccd]
    );
    assert_eq!(
        model.output_embeddings[4..6]
            .iter()
            .map(|coordinate| atomic_float::load(coordinate).to_bits())
            .collect::<Vec<_>>(),
        [0x3bcccccd, 0xbc4ccccd]
    );
}

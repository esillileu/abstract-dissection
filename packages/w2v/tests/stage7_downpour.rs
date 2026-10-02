use std::{fs, sync::Arc};
use w2v::{
    Corpus, DownpourConfig, DownpourTrainingSession, DownpourTrainingState, Model, ModelKind,
    ObjectiveKind, RngAlgorithm, Trainer, TrainingConfig, Vocabulary, VocabularyConfig,
    atomic_float,
    downpour::{
        ParameterKind, ParameterServerShard, ReplicaBackend, global_row_index, local_row_index,
        shard_index,
    },
    training::ParameterBackend,
};

fn fixture() -> (Arc<Corpus>, Arc<Vocabulary>) {
    let path = std::env::temp_dir().join(format!(
        "w2v-stage7-{}-{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    fs::write(
        &path,
        b"alpha beta alpha gamma\nbeta alpha delta\ngamma beta alpha\n",
    )
    .unwrap();
    let corpus = Arc::new(Corpus::create(&path).unwrap());
    let vocab = Arc::new(
        Vocabulary::build(
            &corpus,
            &VocabularyConfig {
                initial_capacity: 2,
                hash_capacity: 17,
                min_count: 1,
                max_lexical_words: 0,
            },
        )
        .unwrap(),
    );
    (corpus, vocab)
}

#[test]
fn test_adagrad_known_values() {
    let (_corpus, vocab) = fixture();
    let dim = 2;
    let model = Arc::new(Model::create(&vocab, dim, 1, RngAlgorithm::Lcg).unwrap());
    // Zero out model embeddings
    for val in model
        .input_embeddings
        .iter()
        .chain(&model.output_embeddings)
    {
        atomic_float::store(val, 0.0);
    }
    let gamma = 0.1f32;
    let epsilon = 1e-6f32;
    let mut shard = ParameterServerShard::initialize(
        0,
        1,
        vocab.entries.len(),
        dim,
        gamma,
        epsilon,
        Arc::clone(&model),
    )
    .unwrap();

    // Step 1: apply gradient [2.0, -1.0] to input row 0
    let g1 = [2.0f32, -1.0f32];
    shard.apply_gradient(ParameterKind::Input, 0, &g1);

    let expected_accum1_0 = 4.0f32;
    let expected_accum1_1 = 1.0f32;
    let expected_param1_0 = gamma * 2.0 / (expected_accum1_0 + epsilon).sqrt();
    let expected_param1_1 = -gamma / (expected_accum1_1 + epsilon).sqrt();

    assert!(
        (shard.input_adagrad[0] - expected_accum1_0).abs() < 1e-6,
        "accum[0] = {}, expected {}",
        shard.input_adagrad[0],
        expected_accum1_0
    );
    assert!(
        (shard.input_adagrad[1] - expected_accum1_1).abs() < 1e-6,
        "accum[1] = {}, expected {}",
        shard.input_adagrad[1],
        expected_accum1_1
    );
    let p0 = atomic_float::load(&model.input_embeddings[0]);
    let p1 = atomic_float::load(&model.input_embeddings[1]);
    assert!(
        (p0 - expected_param1_0).abs() < 1e-6,
        "param[0] = {}, expected {}",
        p0,
        expected_param1_0
    );
    assert!(
        (p1 - expected_param1_1).abs() < 1e-6,
        "param[1] = {}, expected {}",
        p1,
        expected_param1_1
    );

    // Step 2: apply gradient [1.0, 3.0] to input row 0
    let g2 = [1.0f32, 3.0f32];
    shard.apply_gradient(ParameterKind::Input, 0, &g2);

    let expected_accum2_0 = expected_accum1_0 + 1.0f32;
    let expected_accum2_1 = expected_accum1_1 + 9.0f32;
    let expected_param2_0 = expected_param1_0 + gamma * 1.0 / (expected_accum2_0 + epsilon).sqrt();
    let expected_param2_1 = expected_param1_1 + gamma * 3.0 / (expected_accum2_1 + epsilon).sqrt();

    assert!(
        (shard.input_adagrad[0] - expected_accum2_0).abs() < 1e-6,
        "accum[0] = {}, expected {}",
        shard.input_adagrad[0],
        expected_accum2_0
    );
    assert!(
        (shard.input_adagrad[1] - expected_accum2_1).abs() < 1e-6,
        "accum[1] = {}, expected {}",
        shard.input_adagrad[1],
        expected_accum2_1
    );
    let p0_2 = atomic_float::load(&model.input_embeddings[0]);
    let p1_2 = atomic_float::load(&model.input_embeddings[1]);
    assert!(
        (p0_2 - expected_param2_0).abs() < 1e-6,
        "param[0] = {}, expected {}",
        p0_2,
        expected_param2_0
    );
    assert!(
        (p1_2 - expected_param2_1).abs() < 1e-6,
        "param[1] = {}, expected {}",
        p1_2,
        expected_param2_1
    );
}

#[test]
fn test_ps_routing_mutually_exclusive_and_exhaustive() {
    let vocab_size = 100;
    for shard_count in [1, 2, 3, 5, 8] {
        for kind in [ParameterKind::Input, ParameterKind::Output] {
            let mut shard_assigned_rows = vec![Vec::new(); shard_count];
            for row in 0..vocab_size {
                let sid = shard_index(kind, row, shard_count);
                assert!(sid < shard_count);
                shard_assigned_rows[sid].push(row);

                let local = local_row_index(row, shard_count);
                let reconstructed = global_row_index(kind, local, sid, shard_count);
                assert_eq!(
                    reconstructed, row,
                    "Failed inversion for kind {:?} row {} shard_count {}",
                    kind, row, shard_count
                );
            }
            // Check mutual exclusivity: no overlap across shards
            let mut total_rows = 0;
            for rows in &shard_assigned_rows {
                total_rows += rows.len();
            }
            assert_eq!(total_rows, vocab_size);
        }
    }
}

#[test]
fn test_forced_staleness() {
    let (_corpus, vocab) = fixture();
    let dim = 2;
    let model = Arc::new(Model::create(&vocab, dim, 1, RngAlgorithm::Lcg).unwrap());
    // Initial parameter P0 = [1.0, 2.0] at input row 0
    atomic_float::store(&model.input_embeddings[0], 1.0);
    atomic_float::store(&model.input_embeddings[1], 2.0);

    let gamma = 0.1f32;
    let epsilon = 1e-6f32;
    let mut shard = ParameterServerShard::initialize(
        0,
        1,
        vocab.entries.len(),
        dim,
        gamma,
        epsilon,
        Arc::clone(&model),
    )
    .unwrap();

    let mut replica_a = ReplicaBackend::new(dim, Arc::clone(&model));
    let mut replica_b = ReplicaBackend::new(dim, Arc::clone(&model));

    // Both replicas read parameter row 0: P0 = [1.0, 2.0]
    let mut snap_a = [0.0; 2];
    replica_a.load_input_row(0, &mut snap_a);
    assert_eq!(snap_a, [1.0, 2.0]);

    let mut snap_b = [0.0; 2];
    replica_b.load_input_row(0, &mut snap_b);
    assert_eq!(snap_b, [1.0, 2.0]);

    // Replica A computes gradient gA = [0.5, 0.5] and pushes to PS
    let g_a = [0.5f32, 0.5f32];
    shard.apply_gradient(ParameterKind::Input, 0, &g_a);

    // Authoritative model is now updated to P1
    let p0_after_a = atomic_float::load(&model.input_embeddings[0]);
    let p1_after_a = atomic_float::load(&model.input_embeddings[1]);
    let expected_p0 = 1.0 + gamma * 0.5 / (0.25 + epsilon).sqrt();
    let expected_p1 = 2.0 + gamma * 0.5 / (0.25 + epsilon).sqrt();
    assert!((p0_after_a - expected_p0).abs() < 1e-6);
    assert!((p1_after_a - expected_p1).abs() < 1e-6);

    // Replica B loads input row 0 AGAIN during the same mini-batch
    let mut snap_b_again = [0.0; 2];
    replica_b.load_input_row(0, &mut snap_b_again);
    // Replica B MUST see the STALE snapshot [1.0, 2.0], NOT the updated model [1.1, 2.1]!
    assert_eq!(
        snap_b_again,
        [1.0, 2.0],
        "Replica B did not see stale parameter snapshot"
    );

    // Replica B pushes its gradient gB = [1.0, 1.0] to PS
    let g_b = [1.0f32, 1.0f32];
    shard.apply_gradient(ParameterKind::Input, 0, &g_b);

    // PS applied gA then gB sequentially to AdaGrad state
    let expected_accum_0 = 0.25 + 1.0;
    let expected_accum_1 = 0.25 + 1.0;
    assert!((shard.input_adagrad[0] - expected_accum_0).abs() < 1e-6);
    assert!((shard.input_adagrad[1] - expected_accum_1).abs() < 1e-6);

    let expected_final_p0 = expected_p0 + gamma * 1.0 / (expected_accum_0 + epsilon).sqrt();
    let expected_final_p1 = expected_p1 + gamma * 1.0 / (expected_accum_1 + epsilon).sqrt();
    let final_p0 = atomic_float::load(&model.input_embeddings[0]);
    let final_p1 = atomic_float::load(&model.input_embeddings[1]);
    assert!((final_p0 - expected_final_p0).abs() < 1e-6);
    assert!((final_p1 - expected_final_p1).abs() < 1e-6);
}

#[test]
fn test_downpour_single_replica_cbow_hs() {
    let (corpus, vocab) = fixture();
    let mut config = TrainingConfig::for_model(ModelKind::Cbow);
    config.objective_kind = ObjectiveKind::HierarchicalSoftmax;
    config.embedding_dimension = 8;
    config.window_radius = 2;
    config.epochs = 2;
    config.thread_count = 1;
    config.subsampling_threshold = 0.0;
    config.sigmoid_table_size = 101;
    let model = Arc::new(Model::create(&vocab, 8, config.root_seed, config.rng_algorithm).unwrap());
    let trainer = Arc::new(Trainer::create(corpus, vocab, model, &config).unwrap());

    let downpour_config = DownpourConfig {
        parameter_server_shards: 1,
        mini_batch_targets: 10,
        adagrad_gamma: 0.05,
        adagrad_epsilon: 1e-6,
        fetch_interval: 1,
        push_interval: 1,
        queue_capacity: 16,
    };
    let mut session = DownpourTrainingSession::create(trainer, downpour_config).unwrap();
    assert_eq!(session.completed_epochs(), 0);
    assert!(!session.is_complete());

    let report1 = session.train_epoch().unwrap();
    assert_eq!(report1.epoch, 1);
    assert!(report1.epoch_tokens > 0);
    assert_eq!(session.completed_epochs(), 1);

    let report2 = session.train_epoch().unwrap();
    assert_eq!(report2.epoch, 2);
    assert!(session.is_complete());

    let state = session.export_state().unwrap();
    assert_eq!(state.completed_epochs, 2);
    assert!(state.input_embeddings.iter().all(|x| x.is_finite()));
    assert!(state.output_embeddings.iter().all(|x| x.is_finite()));
    assert!(state.input_adagrad.iter().all(|x| x.is_finite()));
    assert!(state.output_adagrad.iter().all(|x| x.is_finite()));
}

#[test]
fn test_downpour_single_replica_skipgram_neg() {
    let (corpus, vocab) = fixture();
    let mut config = TrainingConfig::for_model(ModelKind::SkipGram);
    config.objective_kind = ObjectiveKind::NegativeSampling;
    config.embedding_dimension = 8;
    config.window_radius = 2;
    config.epochs = 2;
    config.thread_count = 1;
    config.subsampling_threshold = 0.0;
    config.negative_sample_count = 2;
    config.negative_table_size = 257;
    config.sigmoid_table_size = 101;
    let model = Arc::new(Model::create(&vocab, 8, config.root_seed, config.rng_algorithm).unwrap());
    let trainer = Arc::new(Trainer::create(corpus, vocab, model, &config).unwrap());

    let downpour_config = DownpourConfig {
        parameter_server_shards: 1,
        mini_batch_targets: 10,
        adagrad_gamma: 0.05,
        adagrad_epsilon: 1e-6,
        fetch_interval: 1,
        push_interval: 1,
        queue_capacity: 16,
    };
    let mut session = DownpourTrainingSession::create(trainer, downpour_config).unwrap();
    while !session.is_complete() {
        session.train_epoch().unwrap();
    }
    assert_eq!(session.completed_epochs(), 2);
    let state = session.export_state().unwrap();
    assert!(state.input_embeddings.iter().all(|x| x.is_finite()));
    assert!(state.output_embeddings.iter().all(|x| x.is_finite()));
    assert!(state.input_adagrad.iter().all(|x| x.is_finite()));
    assert!(state.output_adagrad.iter().all(|x| x.is_finite()));
}

#[test]
fn test_downpour_multi_replica_deadlock_free() {
    let (corpus, vocab) = fixture();
    let mut config = TrainingConfig::for_model(ModelKind::Cbow);
    config.objective_kind = ObjectiveKind::HierarchicalSoftmax;
    config.embedding_dimension = 8;
    config.window_radius = 2;
    config.epochs = 2;
    config.thread_count = 2; // 2 replicas
    config.subsampling_threshold = 0.0;
    config.sigmoid_table_size = 101;
    let model = Arc::new(Model::create(&vocab, 8, config.root_seed, config.rng_algorithm).unwrap());
    let trainer = Arc::new(Trainer::create(corpus, vocab, model, &config).unwrap());

    let downpour_config = DownpourConfig {
        parameter_server_shards: 2, // 2 PS shards
        mini_batch_targets: 5,
        adagrad_gamma: 0.05,
        adagrad_epsilon: 1e-6,
        fetch_interval: 1,
        push_interval: 1,
        queue_capacity: 8,
    };
    let mut session = DownpourTrainingSession::create(trainer, downpour_config).unwrap();
    for _ in 0..2 {
        let report = session.train_epoch().unwrap();
        assert!(report.epoch_tokens > 0);
    }
    assert_eq!(session.completed_epochs(), 2);
    assert!(session.is_complete());
    let state = session.export_state().unwrap();
    assert!(state.input_embeddings.iter().all(|x| x.is_finite()));
    assert!(state.output_embeddings.iter().all(|x| x.is_finite()));
}

#[test]
fn test_downpour_checkpoint_roundtrip() {
    let (corpus, vocab) = fixture();
    let mut config = TrainingConfig::for_model(ModelKind::Cbow);
    config.objective_kind = ObjectiveKind::HierarchicalSoftmax;
    config.embedding_dimension = 8;
    config.window_radius = 2;
    config.epochs = 2;
    config.thread_count = 2;
    config.subsampling_threshold = 0.0;
    config.sigmoid_table_size = 101;
    let model = Arc::new(Model::create(&vocab, 8, config.root_seed, config.rng_algorithm).unwrap());
    let trainer = Arc::new(
        Trainer::create(
            Arc::clone(&corpus),
            Arc::clone(&vocab),
            Arc::clone(&model),
            &config,
        )
        .unwrap(),
    );

    let downpour_config = DownpourConfig {
        parameter_server_shards: 2,
        mini_batch_targets: 5,
        adagrad_gamma: 0.05,
        adagrad_epsilon: 1e-6,
        fetch_interval: 1,
        push_interval: 1,
        queue_capacity: 8,
    };
    let mut session =
        DownpourTrainingSession::create(Arc::clone(&trainer), downpour_config.clone()).unwrap();
    session.train_epoch().unwrap();
    assert_eq!(session.completed_epochs(), 1);

    let state: DownpourTrainingState = session.export_state().unwrap();
    assert_eq!(state.completed_epochs, 1);
    assert_eq!(state.replicas.len(), 2);

    // Create a fresh session and restore from state
    let restored_model =
        Arc::new(Model::create(&vocab, 8, config.root_seed, config.rng_algorithm).unwrap());
    let restored_trainer = Arc::new(
        Trainer::create(
            Arc::clone(&corpus),
            Arc::clone(&vocab),
            restored_model,
            &config,
        )
        .unwrap(),
    );
    let mut restored_session =
        DownpourTrainingSession::restore(restored_trainer, downpour_config, &state).unwrap();

    assert_eq!(restored_session.completed_epochs(), 1);
    let state_restored = restored_session.export_state().unwrap();
    assert_eq!(state_restored.input_embeddings, state.input_embeddings);
    assert_eq!(state_restored.output_embeddings, state.output_embeddings);
    assert_eq!(state_restored.input_adagrad, state.input_adagrad);
    assert_eq!(state_restored.output_adagrad, state.output_adagrad);
    assert_eq!(state_restored.processed_tokens, state.processed_tokens);
    assert_eq!(state_restored.replicas, state.replicas);

    // Train second epoch on restored session
    let report = restored_session.train_epoch().unwrap();
    assert_eq!(report.epoch, 2);
    assert!(restored_session.is_complete());
}

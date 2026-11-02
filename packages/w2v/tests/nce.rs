use std::{path::PathBuf, sync::Arc};
use w2v::{
    ContextPolicy, Corpus, EmbeddingKind, Model, ModelKind, ObjectiveKind, RngAlgorithm, Status,
    Trainer, TrainingConfig, UpdateStrategy, Vocabulary, VocabularyEntry,
    downpour::{GradientBatch, ParameterKind, ParameterServerShard, ReplicaBackend},
    random::Rng,
    training::{
        ModelStep, ObjectiveScratch, ParameterBackend, SharedModelBackend, cbow_train_with_backend,
        nce::{binary_loss_gradient, train_with_backend},
    },
    vocab::NceNoiseSampler,
};

fn vocabulary() -> Vocabulary {
    Vocabulary {
        entries: [100, 1, 16, 81]
            .into_iter()
            .enumerate()
            .map(|(i, count)| VocabularyEntry {
                token: format!("w{i}").into_bytes(),
                count,
                huffman_path: vec![],
                huffman_bits: vec![],
            })
            .collect(),
        hash_slots: vec![None; 17],
        retained_token_count: 198,
    }
}

fn trainer() -> Trainer {
    let vocab = Arc::new(vocabulary());
    let config = TrainingConfig {
        model_kind: ModelKind::SkipGram,
        objective_kind: ObjectiveKind::Nce,
        embedding_dimension: 2,
        negative_sample_count: 5,
        negative_table_size: 0,
        ..TrainingConfig::default()
    };
    let model = Arc::new(Model::create(&vocab, 2, 1, RngAlgorithm::Lcg).unwrap());
    Trainer::create_with_digest(
        Arc::new(Corpus {
            path: PathBuf::new(),
            byte_size: 0,
            full_tokens: false,
        }),
        vocab,
        model,
        &config,
        Some("fixture".into()),
    )
    .unwrap()
}

#[test]
fn binary_likelihood_matches_independent_density_ratio_and_is_not_neg() {
    for (score, q, k) in [(0.7f64, 0.1f64, 5), (-1.2, 0.7, 15), (0.0, 0.25, 5)] {
        let p = score.exp();
        let posterior = p / (p + k as f64 * q);
        for data in [true, false] {
            let (loss, derivative) = binary_loss_gradient(score, q, k, data).unwrap();
            let expected = -if data { posterior } else { 1.0 - posterior }.ln();
            assert!((loss - expected).abs() < 1e-14);
            assert!((derivative - (posterior - if data { 1.0 } else { 0.0 })).abs() < 1e-14);
            let neg_posterior = 1.0 / (1.0 + (-score).exp());
            let neg_loss = -if data {
                neg_posterior
            } else {
                1.0 - neg_posterior
            }
            .ln();
            assert!((loss - neg_loss).abs() > 0.01);
        }
    }
    for score in [-1000.0, 1000.0] {
        for data in [true, false] {
            let (loss, derivative) = binary_loss_gradient(score, 1e-12, 5, data).unwrap();
            assert!(loss.is_finite() && derivative.is_finite());
        }
    }
    assert_eq!(
        binary_loss_gradient(0.0, 0.0, 5, true),
        Err(Status::InvalidArgument)
    );
    assert_eq!(
        binary_loss_gradient(0.0, 0.5, 0, true),
        Err(Status::InvalidArgument)
    );
}

#[test]
fn noise_masses_probabilities_and_seeded_draws_agree() {
    let sampler = NceNoiseSampler::initialize(&vocabulary()).unwrap();
    // 1^(.75), 16^(.75), 81^(.75) = 1, 8, 27, with EOS excluded.
    assert_eq!(sampler.probability(0), 0.0);
    let sum: f64 = (1..4).map(|i| sampler.probability(i)).sum();
    assert!((sum - 1.0).abs() < 1e-15);
    for (i, mass) in [1.0, 8.0, 27.0].into_iter().enumerate() {
        let q = sampler.probability(i + 1);
        assert!((q - mass / 36.0).abs() < 1e-15);
        let (_, d) = binary_loss_gradient(0.0, q, 5, false).unwrap();
        assert!((d - 1.0 / (1.0 + 5.0 * mass / 36.0)).abs() < 1e-15);
    }
    for algorithm in [RngAlgorithm::Lcg, RngAlgorithm::Xorshift] {
        let mut first = Rng::new(77, algorithm);
        let mut second = first;
        let mut counts = [0usize; 4];
        for _ in 0..200_000 {
            let token = sampler.draw(&mut first);
            assert_eq!(token, sampler.draw(&mut second));
            counts[token] += 1;
        }
        assert_eq!(counts[0], 0);
        for (i, &count) in counts.iter().enumerate().skip(1) {
            assert!((count as f64 / 200_000.0 - sampler.probability(i)).abs() < 0.004);
        }
    }
}

const OUTPUT: [f32; 8] = [0.0, 0.0, 0.2, -0.4, -0.8, 0.5, 0.6, 0.1];
const INPUT: [f32; 8] = [0.0, 0.0, 0.3, -0.7, 0.5, 0.2, 0.4, -1.0];

// Independent density-ratio reference: do not use binary_loss_gradient.
fn reference_loss(h: &[f64], output: &[f64], target: usize, trainer: &Trainer) -> f64 {
    // The audit counterexample: seed 17 produces [1,3,3,3,3]. This deliberately
    // includes four repetitions and (when target=3) positive/noise collisions.
    [target, 1, 3, 3, 3, 3]
        .into_iter()
        .enumerate()
        .map(|(i, w)| {
            let p = (h[0] * output[2 * w] + h[1] * output[2 * w + 1]).exp();
            let noise = 5.0 * trainer.nce_noise_sampler.probability(w);
            -(if i == 0 {
                p / (p + noise)
            } else {
                noise / (p + noise)
            })
            .ln()
        })
        .sum()
}

fn finite_difference(f: impl Fn(&[f64]) -> f64, x: &[f64]) -> Vec<f64> {
    let eps = 1e-5;
    (0..x.len())
        .map(|i| {
            let mut plus = x.to_vec();
            plus[i] += eps;
            let mut minus = x.to_vec();
            minus[i] -= eps;
            (f(&plus) - f(&minus)) / (2.0 * eps)
        })
        .collect()
}

fn snapshot(trainer: &Trainer, kind: EmbeddingKind) -> Vec<f32> {
    let mut values = vec![0.0; 8];
    assert_eq!(trainer.model.snapshot_into(kind, &mut values), Status::Ok);
    values
}

fn restore(trainer: &Trainer, input: &[f32]) {
    assert_eq!(
        trainer.model.restore_from(EmbeddingKind::Input, input),
        Status::Ok
    );
    assert_eq!(
        trainer.model.restore_from(EmbeddingKind::Output, &OUTPUT),
        Status::Ok
    );
}

fn assert_sgd(actual: &[f32], initial: &[f32], gradient: &[f64], lr: f32) {
    for i in 0..actual.len() {
        let expected = initial[i] as f64 - lr as f64 * gradient[i];
        assert!(
            (actual[i] as f64 - expected).abs() < 1e-7,
            "coordinate {i}: actual={} expected={expected}",
            actual[i]
        );
    }
}

fn apply_and_check_replica(
    trainer: &Trainer,
    backend: &ReplicaBackend,
    input: &[f32],
    gi: &[f64],
    go: &[f64],
) {
    // ReplicaBackend accumulates gradients; exercise its actual production
    // parameter-server AdaGrad update too, including its accumulator state.
    let gamma = 0.05f32;
    let epsilon = 1e-6f32;
    let mut shard =
        ParameterServerShard::initialize(0, 1, 4, 2, gamma, epsilon, Arc::clone(&trainer.model))
            .unwrap();
    shard.apply_batch(&GradientBatch {
        replica_id: 0,
        batch_id: 0,
        headers: backend.grad_headers.clone(),
        gradients: backend.grad_arena.clone(),
    });
    for (kind, initial, gradient, accumulator) in [
        (EmbeddingKind::Input, input, gi, &shard.input_adagrad),
        (
            EmbeddingKind::Output,
            OUTPUT.as_slice(),
            go,
            &shard.output_adagrad,
        ),
    ] {
        let actual = snapshot(trainer, kind);
        for i in 0..8 {
            let g = gradient[i];
            let expected = initial[i] as f64 - gamma as f64 * g / (g * g + epsilon as f64).sqrt();
            assert!((actual[i] as f64 - expected).abs() < 2e-7);
            assert!((accumulator[i] as f64 - g * g).abs() < 2e-7);
        }
    }
}

#[test]
fn duplicate_and_positive_collision_gradients_update_real_backends_correctly() {
    let trainer = trainer();
    let h = INPUT[2..4].iter().map(|&v| v as f64).collect::<Vec<_>>();
    let u = OUTPUT.map(f64::from);
    let gh = finite_difference(|x| reference_loss(x, &u, 3, &trainer), &h);
    let go = finite_difference(|x| reference_loss(&h, x, 3, &trainer), &u);
    let mut gi = vec![0.0; 8];
    gi[2..4].copy_from_slice(&gh);
    let mut rng = Rng::new(17, RngAlgorithm::Lcg);
    assert_eq!(
        (0..5)
            .map(|_| trainer.nce_noise_sampler.draw(&mut rng))
            .collect::<Vec<_>>(),
        [1, 3, 3, 3, 3]
    );
    for strategy in [UpdateStrategy::AtomicCas, UpdateStrategy::Hogwild] {
        restore(&trainer, &INPUT);
        let mut backend = SharedModelBackend::new(&trainer.model, strategy);
        let mut hidden_gradient = [0.0; 2];
        let mut output_snapshot = [0.0; 2];
        let loss = train_with_backend(
            &trainer,
            &mut backend,
            3,
            0.05,
            &mut Rng::new(17, RngAlgorithm::Lcg),
            &INPUT[2..4],
            ObjectiveScratch {
                hidden_gradient: &mut hidden_gradient,
                output_snapshot: &mut output_snapshot,
            },
            true,
        )
        .unwrap();
        assert_eq!(loss.count, 6);
        assert!((loss.sum - reference_loss(&h, &u, 3, &trainer)).abs() < 1e-7);
        for j in 0..2 {
            assert!((hidden_gradient[j] as f64 + 0.05f32 as f64 * gh[j]).abs() < 2e-8);
        }
        backend.apply_input_gradient(1, &hidden_gradient, 0.05);
        assert_sgd(&snapshot(&trainer, EmbeddingKind::Input), &INPUT, &gi, 0.05);
        assert_sgd(
            &snapshot(&trainer, EmbeddingKind::Output),
            &OUTPUT,
            &go,
            0.05,
        );
        println!(
            "audit {strategy:?}: loss={:.12}, hidden_delta={hidden_gradient:?}",
            loss.sum
        );
    }
    restore(&trainer, &INPUT);
    let mut backend = ReplicaBackend::new(2, Arc::clone(&trainer.model));
    let mut hidden_gradient = [0.0; 2];
    let mut output_snapshot = [0.0; 2];
    let loss = train_with_backend(
        &trainer,
        &mut backend,
        3,
        0.05,
        &mut Rng::new(17, RngAlgorithm::Lcg),
        &INPUT[2..4],
        ObjectiveScratch {
            hidden_gradient: &mut hidden_gradient,
            output_snapshot: &mut output_snapshot,
        },
        true,
    )
    .unwrap();
    assert!((loss.sum - reference_loss(&h, &u, 3, &trainer)).abs() < 1e-7);
    for j in 0..2 {
        assert!((hidden_gradient[j] as f64 + gh[j]).abs() < 2e-7);
    }
    backend.apply_input_gradient(1, &hidden_gradient, 0.05);
    assert_eq!(backend.grad_headers.len(), 3); // one input and two unique output rows
    for w in [1, 3] {
        let off = backend.grad_map[&(ParameterKind::Output, w)];
        for j in 0..2 {
            assert!((backend.grad_arena[off + j] as f64 + go[2 * w + j]).abs() < 2e-7);
        }
    }
    apply_and_check_replica(&trainer, &backend, &INPUT, &gi, &go);
}

fn cbow_step<B: ParameterBackend>(trainer: &Trainer, backend: &mut B, sentence: &[usize]) -> f64 {
    let mut hidden = [0.0; 2];
    let mut hg = [0.0; 2];
    let mut output = [0.0; 2];
    cbow_train_with_backend(
        &mut ModelStep {
            trainer,
            target_token: sentence[1],
            learning_rate: 0.05,
            sentence,
            sentence_position: 1,
            hidden: &mut hidden,
            hidden_gradient: &mut hg,
            output_snapshot: &mut output,
            window_rng: &mut Rng::new(1, RngAlgorithm::Lcg),
            negative_rng: &mut Rng::new(17, RngAlgorithm::Lcg),
            observe_objective: true,
        },
        backend,
    )
    .unwrap()
    .sum
}

#[test]
fn cbow_nce_mean_gradient_matches_finite_difference_in_both_backends() {
    let mut trainer = trainer();
    trainer.config.model_kind = ModelKind::Cbow;
    trainer.config.context_policy = ContextPolicy::Fixed;
    trainer.config.window_radius = 1;
    let input = [0.0, 0.0, 0.2, -0.4, 0.5, 0.2, 0.4, -1.0];
    let x = input.map(f64::from);
    let u = OUTPUT.map(f64::from);
    for sentence in [[1, 3, 3], [1, 3, 1]] {
        let objective = |x: &[f64], u: &[f64]| {
            let h = [
                (x[2 * sentence[0]] + x[2 * sentence[2]]) / 2.0,
                (x[2 * sentence[0] + 1] + x[2 * sentence[2] + 1]) / 2.0,
            ];
            reference_loss(&h, u, sentence[1], &trainer)
        };
        let gi = finite_difference(|v| objective(v, &u), &x);
        let go = finite_difference(|v| objective(&x, v), &u);
        for strategy in [UpdateStrategy::AtomicCas, UpdateStrategy::Hogwild] {
            restore(&trainer, &input);
            let mut backend = SharedModelBackend::new(&trainer.model, strategy);
            let loss = cbow_step(&trainer, &mut backend, &sentence);
            assert!((loss - objective(&x, &u)).abs() < 1e-7);
            assert_sgd(&snapshot(&trainer, EmbeddingKind::Input), &input, &gi, 0.05);
            assert_sgd(
                &snapshot(&trainer, EmbeddingKind::Output),
                &OUTPUT,
                &go,
                0.05,
            );
        }
        restore(&trainer, &input);
        let mut backend = ReplicaBackend::new(2, Arc::clone(&trainer.model));
        cbow_step(&trainer, &mut backend, &sentence);
        apply_and_check_replica(&trainer, &backend, &input, &gi, &go);
    }
}

#[test]
fn extreme_logits_preserve_representable_tail_gradients() {
    // q=.2,k=5 makes z=score. Reference tail uses exp(-|z|) directly;
    // finite differences differentiate the independently written stable loss.
    for z in [30.0f64, 36.0, 37.0, 40.0, 100.0] {
        let tail = (-z).exp();
        for (score, data, sign) in [(z, true, -1.0), (-z, false, 1.0)] {
            let (loss, gradient) = binary_loss_gradient(score, 0.2, 5, data).unwrap();
            let expected = sign * tail / (1.0 + tail);
            assert!(gradient != 0.0);
            assert!(((gradient - expected) / expected).abs() < 1e-14);
            let reference_loss = |s: f64| {
                if data {
                    (-s).exp().ln_1p()
                } else {
                    s.exp().ln_1p()
                }
            };
            let numerical = (reference_loss(score + 1e-5) - reference_loss(score - 1e-5)) / 2e-5;
            assert!(((gradient - numerical) / gradient).abs() < 1e-8);
            assert!((loss - tail.ln_1p()).abs() < 1e-14);
        }
    }
    for z in [-1000.0, 1000.0] {
        for data in [true, false] {
            let (loss, g) = binary_loss_gradient(z, 0.2, 5, data).unwrap();
            assert!(loss.is_finite() && g.is_finite());
            assert_eq!(loss, if (z > 0.0) == data { 0.0 } else { 1000.0 });
            assert_eq!(
                g,
                if data {
                    if z > 0.0 { -0.0 } else { -1.0 }
                } else if z > 0.0 {
                    1.0
                } else {
                    0.0
                }
            );
        }
    }
}

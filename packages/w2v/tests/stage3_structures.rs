use w2v::{
    EmbeddingKind, Model, NegativeSampler, RngAlgorithm, Status, Vocabulary, VocabularyEntry,
    random::Rng,
};

fn fixture_vocab() -> Vocabulary {
    let tokens: [&[u8]; 5] = [b"</s>", b"alpha", b"beta", b"gamma", b"delta"];
    let counts = [3, 4, 3, 2, 1];
    Vocabulary {
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
    }
}

fn hash_float_bits(values: &[f32]) -> u64 {
    let mut hash = 1469598103934665603u64;
    for value in values {
        for byte in value.to_bits().to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(1099511628211);
        }
    }
    hash
}

#[test]
fn c_huffman_paths_and_bits() {
    let mut vocab = fixture_vocab();
    assert_eq!(vocab.assign_huffman(), Status::Ok);
    let expected = [
        (&[3, 2][..], &[1, 1][..]),
        (&[3, 2][..], &[1, 0][..]),
        (&[3, 1][..], &[0, 1][..]),
        (&[3, 1, 0][..], &[0, 0, 1][..]),
        (&[3, 1, 0][..], &[0, 0, 0][..]),
    ];
    for (entry, (path, bits)) in vocab.entries.iter().zip(expected) {
        assert_eq!(entry.huffman_path, path);
        assert_eq!(entry.huffman_bits, bits);
    }
    assert_eq!(vocab.assign_huffman(), Status::Ok);
    let mut empty = Vocabulary {
        entries: Vec::new(),
        hash_slots: Vec::new(),
        retained_token_count: 0,
    };
    assert_eq!(empty.assign_huffman(), Status::InvalidArgument);
}

#[test]
fn c_negative_table_and_draw() {
    let vocab = fixture_vocab();
    let sampler = NegativeSampler::initialize(&vocab, 1000).unwrap();
    assert_eq!(
        [
            sampler.table[0],
            sampler.table[250],
            sampler.table[500],
            sampler.table[750],
            sampler.table[999]
        ],
        [0, 1, 1, 3, 4]
    );
    assert_eq!(
        sampler.table.iter().filter(|&&entry| entry == 1).count(),
        281
    );
    if size_of::<usize>() == 8 {
        let mut hash = 1469598103934665603u64;
        for entry in &sampler.table {
            for byte in entry.to_le_bytes() {
                hash ^= byte as u64;
                hash = hash.wrapping_mul(1099511628211);
            }
        }
        assert_eq!(hash, 0x0f3d6874986ffd01);
    }
    let mut rng = Rng::new(1, RngAlgorithm::Lcg);
    assert_eq!(sampler.draw(&mut rng), (3, 25214903928));
    assert_eq!(
        NegativeSampler::initialize(&vocab, 0).unwrap_err(),
        Status::InvalidArgument
    );
    let mut zero = fixture_vocab();
    for entry in &mut zero.entries {
        entry.count = 0;
    }
    assert_eq!(
        NegativeSampler::initialize(&zero, 7).unwrap_err(),
        Status::CorruptData
    );
}

#[test]
fn c_model_initialization_and_snapshot() {
    let vocab = fixture_vocab();
    for (algorithm, expected_hash) in [
        (RngAlgorithm::Lcg, 0x1481c956b48bc5f8),
        (RngAlgorithm::Xorshift, 0x2e41636ac144f2d5),
    ] {
        let model = Model::create(&vocab, 8, 1, algorithm).unwrap();
        let mut input = vec![0.0; 40];
        let mut output = vec![1.0; 40];
        assert_eq!(
            model.snapshot_into(EmbeddingKind::Input, &mut input),
            Status::Ok
        );
        assert_eq!(
            model.snapshot_into(EmbeddingKind::Output, &mut output),
            Status::Ok
        );
        assert_eq!(hash_float_bits(&input), expected_hash);
        assert!(output.iter().all(|&value| value.to_bits() == 0));
        assert_eq!(
            model.snapshot_into(EmbeddingKind::Input, &mut input[..39]),
            Status::InvalidArgument
        );
    }
    assert_eq!(
        Model::create(&vocab, 0, 1, RngAlgorithm::Lcg).err(),
        Some(Status::InvalidArgument)
    );
    assert_eq!(
        Model::create(&vocab, usize::MAX, 1, RngAlgorithm::Lcg).err(),
        Some(Status::InvalidArgument)
    );
}

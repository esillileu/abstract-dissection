use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::PathBuf,
    sync::atomic::AtomicU32,
};
use w2v::{
    Corpus, RngAlgorithm, Status, Vocabulary, VocabularyConfig, atomic_float,
    random::Rng,
    training::{context_position, context_radius, objective_apply_update, objective_score},
};

struct FixtureFile(PathBuf);
impl FixtureFile {
    fn new(label: &str, bytes: &[u8]) -> Self {
        let path =
            std::env::temp_dir().join(format!("w2v-stage4-{}-{}", std::process::id(), label));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        file.write_all(bytes).unwrap();
        Self(path)
    }
    fn corpus(&self) -> Corpus {
        Corpus::create(&self.0).unwrap()
    }
}
impl Drop for FixtureFile {
    fn drop(&mut self) {
        fs::remove_file(&self.0).unwrap();
    }
}

fn small_config(hash_capacity: usize) -> VocabularyConfig {
    VocabularyConfig {
        initial_capacity: 2,
        hash_capacity,
        min_count: 1,
    }
}

#[test]
fn c_vocabulary_counts_order_paths_and_lookup() {
    let file = FixtureFile::new(
        "corpus",
        b"alpha beta alpha gamma\nbeta alpha delta\ngamma beta alpha\n",
    );
    let vocab = Vocabulary::build(&file.corpus(), &small_config(17)).unwrap();
    let tokens: Vec<_> = vocab
        .entries
        .iter()
        .map(|entry| entry.token.as_slice())
        .collect();
    assert_eq!(
        tokens,
        [b"</s>".as_slice(), b"alpha", b"beta", b"gamma", b"delta"]
    );
    let counts: Vec<_> = vocab.entries.iter().map(|entry| entry.count).collect();
    assert_eq!(counts, [3, 4, 3, 2, 1]);
    assert_eq!(vocab.retained_token_count, 13);
    let expected_paths: [&[usize]; 5] = [&[3, 2], &[3, 2], &[3, 1], &[3, 1, 0], &[3, 1, 0]];
    for (entry, path) in vocab.entries.iter().zip(expected_paths) {
        assert_eq!(entry.huffman_path, path);
    }
    assert_eq!(vocab.find(b"alpha"), Some(1));
    assert_eq!(vocab.find(b"delta"), Some(4));
    assert_eq!(vocab.find(b"missing"), None);
}

#[test]
fn c_pruning_and_last_token_at_eof() {
    let prune = FixtureFile::new("prune", b"a b c d e f g h\n");
    let vocab = Vocabulary::build(&prune.corpus(), &small_config(10)).unwrap();
    assert_eq!(vocab.entries.len(), 2);
    assert_eq!(vocab.entries[0].token, b"</s>");
    assert_eq!(vocab.entries[1].token, b"h");
    assert_eq!(vocab.find(b"a"), None);
    assert_eq!(vocab.find(b"h"), Some(1));

    let eof = FixtureFile::new("eof", b"one two");
    let vocab = Vocabulary::build(&eof.corpus(), &small_config(17)).unwrap();
    assert_eq!(vocab.entries.len(), 2);
    assert_eq!(vocab.entries[0].count, 0);
    assert_eq!(vocab.entries[1].token, b"one");
    assert_eq!(vocab.find(b"two"), None);
    assert_eq!(
        Vocabulary::build(
            &eof.corpus(),
            &VocabularyConfig {
                hash_capacity: 1,
                ..small_config(17)
            }
        )
        .err(),
        Some(Status::InvalidArgument)
    );
}

#[test]
fn c_context_order_and_rng_consumption() {
    let positions: Vec<_> = (0..=4)
        .filter_map(|offset| context_position(5, 2, 2, offset))
        .collect();
    assert_eq!(positions, [0, 1, 3, 4]);
    assert_eq!(context_position(3, 0, 1, 0), None);
    assert_eq!(context_position(3, 0, 1, 2), Some(1));
    let mut rng = Rng::new(1, RngAlgorithm::Lcg);
    assert_eq!(context_radius(&mut rng, 2), 2);
    assert_eq!(rng.state, 25214903928);
}

#[test]
fn c_objective_score_and_gradient_bits() {
    let hidden = [0.25, -0.5, 0.75];
    let mut gradient = [0.1, -0.2, 0.3];
    let output = [AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0)];
    for (coordinate, value) in output.iter().zip([0.2, -0.4, 0.3]) {
        atomic_float::store(coordinate, value);
    }
    assert_eq!(objective_score(&hidden, &output).to_bits(), 0x3ef33334);
    assert_eq!(
        objective_apply_update(&hidden, &mut gradient, &output, 0.025),
        w2v::Status::Ok
    );
    assert_eq!(
        gradient.map(f32::to_bits),
        [0x3dd70a3e, 0xbe570a3e, 0x3e9d70a4]
    );
    let output_bits = output
        .each_ref()
        .map(|coordinate| atomic_float::load(coordinate).to_bits());
    assert_eq!(output_bits, [0x3e533333, 0xbed33333, 0x3ea33334]);
}

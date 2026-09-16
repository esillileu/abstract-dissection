use std::{
    fs::{self, OpenOptions},
    io::{Cursor, Write},
    sync::{Arc, atomic::AtomicU32},
};
use w2v::{
    Corpus, RngAlgorithm, Status, atomic_float,
    corpus::Tokenizer,
    model::SigmoidTable,
    random::{Rng, RngPurpose, derive_seed},
};

#[test]
fn c_rng_sequences_and_streams() {
    let expected_lcg = [
        25214903928,
        8602081314781131043,
        4749291277619109362,
        15888805192744905749,
    ];
    let expected_xorshift = [
        5180492295206395165,
        12380297144915551517,
        13389498078930870103,
        5599127315341312413,
    ];
    let mut lcg = Rng::new(1, RngAlgorithm::Lcg);
    let mut xorshift = Rng::new(1, RngAlgorithm::Xorshift);
    for expected in expected_lcg {
        assert_eq!(lcg.next_u64(), expected);
    }
    for expected in expected_xorshift {
        assert_eq!(xorshift.next_u64(), expected);
    }

    let same = derive_seed(7, 2, RngPurpose::Window);
    assert_eq!(same, 6264802608708859485);
    assert_eq!(derive_seed(7, 2, RngPurpose::Negative), 5192747596696415132);
    assert_eq!(same, derive_seed(7, 2, RngPurpose::Window));
    assert_ne!(same, derive_seed(7, 2, RngPurpose::Negative));
    assert_ne!(same, derive_seed(7, 3, RngPurpose::Window));
    let mut lcg = Rng::new(1, RngAlgorithm::Lcg);
    assert_eq!(lcg.uniform().to_bits(), 0x3f667800);
    let mut xorshift = Rng::new(1, RngAlgorithm::Xorshift);
    assert_eq!(xorshift.uniform().to_bits(), 0x3e8fc99c);
    let mut xorshift_zero = Rng::new(0, RngAlgorithm::Xorshift);
    assert_eq!(xorshift_zero.state, 0x6a09e667f3bcc909);
    assert!(xorshift_zero.uniform() >= 0.0 && xorshift_zero.state != 0);
}

#[test]
fn c_tokenizer_boundaries_and_eof() {
    let mut input = vec![b'x'; 120];
    input.extend_from_slice(b"  short\r\n\tlast");
    let mut tokenizer = Tokenizer::new(Cursor::new(input));
    let first = tokenizer.read_token().unwrap();
    assert_eq!(first.token, vec![b'x'; 99]);
    assert!(!first.at_eof);
    assert_eq!(tokenizer.read_token().unwrap().token, b"short");
    assert_eq!(tokenizer.read_token().unwrap().token, b"</s>");
    let last = tokenizer.read_token().unwrap();
    assert_eq!(last.token, b"last");
    assert!(last.at_eof);
    assert_eq!(tokenizer.read_token().unwrap().token, b"");

    let mut tokenizer = Tokenizer::new(Cursor::new(b"a\n\n"));
    assert_eq!(tokenizer.read_token().unwrap().token, b"a");
    assert_eq!(tokenizer.read_token().unwrap().token, b"</s>");
    assert_eq!(tokenizer.read_token().unwrap().token, b"</s>");
    assert!(tokenizer.read_token().unwrap().at_eof);
}

#[test]
fn c_atomic_float_bits_and_concurrent_addition() {
    let coordinate = Arc::new(AtomicU32::new(0));
    atomic_float::store(&coordinate, -0.0);
    assert_eq!(
        atomic_float::load(&coordinate).to_bits(),
        (-0.0f32).to_bits()
    );
    atomic_float::store(&coordinate, 0.0);
    let threads: Vec<_> = (0..4)
        .map(|_| {
            let coordinate = Arc::clone(&coordinate);
            std::thread::spawn(move || {
                for _ in 0..1000 {
                    atomic_float::add(&coordinate, 0.25);
                }
            })
        })
        .collect();
    for thread in threads {
        thread.join().unwrap();
    }
    assert_eq!(atomic_float::load(&coordinate), 1000.0);
}

#[test]
fn c_sigmoid_lookup_values_and_validation() {
    let table = SigmoidTable::initialize(101, 6.0).unwrap();
    assert_eq!(table.lookup(-6.0), 0.0);
    assert_eq!(table.lookup(6.0), 1.0);
    assert_eq!(table.lookup(0.0).to_bits(), 0x3eda41df);
    assert_eq!(table.lookup(-1.0).to_bits(), 0x3e647be7);
    assert_eq!(table.lookup(1.0).to_bits(), 0x3f2864fd);
    assert_eq!(
        SigmoidTable::initialize(1, 6.0).unwrap_err(),
        Status::InvalidArgument
    );
    assert_eq!(
        SigmoidTable::initialize(101, f32::NAN).unwrap_err(),
        Status::InvalidArgument
    );
}

#[test]
fn c_corpus_metadata_and_error_status() {
    let path = std::env::temp_dir().join(format!("w2v-stage2-{}.txt", std::process::id()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .unwrap();
    file.write_all(b"alpha beta\n").unwrap();
    drop(file);
    let corpus = Corpus::create(&path).unwrap();
    assert_eq!(corpus.path, path);
    assert_eq!(corpus.byte_size, 11);
    assert_eq!(
        corpus.tokenizer(0).unwrap().read_token().unwrap().token,
        b"alpha"
    );
    fs::remove_file(&path).unwrap();
    assert_eq!(Corpus::create(&path).unwrap_err(), Status::IoError);
    assert_eq!(Corpus::create("").unwrap_err(), Status::InvalidArgument);
}

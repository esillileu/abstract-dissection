use std::{env, path::Path, process::ExitCode, sync::Arc};
use w2v::{
    Corpus, Model, ModelKind, ObjectiveKind, RngAlgorithm, Trainer, TrainingConfig, Vocabulary,
    VocabularyConfig,
};

fn usage(program: &str) -> ! {
    eprintln!(
        "usage: {program} CORPUS cbow|skipgram hs|negative THREADS EPOCHS [NEGATIVE_TABLE_SIZE]"
    );
    std::process::exit(2);
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if !(6..=7).contains(&args.len()) {
        usage(&args[0]);
    }
    let model_kind = match args[2].as_str() {
        "cbow" => ModelKind::Cbow,
        "skipgram" => ModelKind::SkipGram,
        _ => usage(&args[0]),
    };
    let objective_kind = match args[3].as_str() {
        "hs" => ObjectiveKind::HierarchicalSoftmax,
        "negative" => ObjectiveKind::NegativeSampling,
        _ => usage(&args[0]),
    };
    let parse = |index: usize| args[index].parse::<usize>().ok().filter(|value| *value > 0);
    let Some(threads) = parse(4) else {
        usage(&args[0])
    };
    let Some(epochs) = parse(5) else {
        usage(&args[0])
    };
    let table_size = if args.len() == 7 {
        let Some(value) = parse(6) else {
            usage(&args[0])
        };
        value
    } else {
        100_000_000
    };

    let result = (|| {
        let corpus = Arc::new(Corpus::create(Path::new(&args[1]))?);
        let vocab = Arc::new(Vocabulary::build(
            &corpus,
            &VocabularyConfig {
                initial_capacity: 1000,
                hash_capacity: 30_000_000,
                min_count: 5,
                max_lexical_words: 0,
            },
        )?);
        let mut config = TrainingConfig::for_model(model_kind);
        config.objective_kind = objective_kind;
        config.embedding_dimension = 100;
        config.window_radius = 5;
        config.epochs = epochs;
        config.thread_count = threads;
        config.subsampling_threshold = 0.0;
        config.negative_sample_count = 5;
        config.root_seed = 1;
        config.rng_algorithm = RngAlgorithm::Lcg;
        config.negative_table_size = table_size;
        let model = Arc::new(Model::create(
            &vocab,
            config.embedding_dimension,
            config.root_seed,
            config.rng_algorithm,
        )?);
        let trainer = Arc::new(Trainer::create(
            Arc::clone(&corpus),
            Arc::clone(&vocab),
            model,
            &config,
        )?);
        trainer.train()?;
        Ok::<_, w2v::Status>((trainer.processed_tokens(), vocab.entries.len()))
    })();
    match result {
        Ok((processed, vocab_size)) => {
            println!("processed_tokens={processed} vocab_size={vocab_size}");
            ExitCode::SUCCESS
        }
        Err(status) => {
            eprintln!("rust runner: {}", status.as_str());
            ExitCode::FAILURE
        }
    }
}

//! Sparse replica caches and sharded asynchronous NNLM AdaGrad updates.
use super::{
    NnlmModel, NnlmTrainingConfig, NnlmTrainingState,
    training::{NnlmGradient, Parameters, forward_backward_parameters},
};
use crate::{
    Corpus, DownpourConfig, EpochReport, Observation, StateDescriptor, Status, Vocabulary,
    atomic_float, config::Real, corpus::Tokenizer,
};
use std::{
    collections::{BTreeMap, VecDeque},
    fs::File,
    io::{BufRead, BufReader, Read, Seek, SeekFrom},
    sync::{
        Arc,
        atomic::{AtomicU32, AtomicU64, Ordering},
        mpsc::{SyncSender, sync_channel},
    },
    thread,
    time::Instant,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(usize)]
pub enum ParameterKind {
    Input,
    Hidden,
    Bias,
    Output,
}
const KINDS: [ParameterKind; 4] = [
    ParameterKind::Input,
    ParameterKind::Hidden,
    ParameterKind::Bias,
    ParameterKind::Output,
];

#[derive(Clone, Debug, PartialEq)]
pub struct NnlmAdaGradState {
    pub input: Vec<Real>,
    pub hidden: Vec<Real>,
    pub bias: Vec<Real>,
    pub output: Vec<Real>,
}
impl NnlmAdaGradState {
    fn arrays(&self) -> [&[Real]; 4] {
        [&self.input, &self.hidden, &self.bias, &self.output]
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NnlmReplicaState {
    pub replica_id: usize,
    pub start_byte: usize,
    pub end_byte: usize,
    pub processed_tokens: u64,
    pub objective_count: u64,
    pub batch_count: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NnlmDownpourTrainingState {
    pub training: NnlmTrainingState,
    pub adagrad: NnlmAdaGradState,
    pub replicas: Vec<NnlmReplicaState>,
}

pub struct SharedParameters {
    arrays: [Vec<AtomicU32>; 4],
    dimensions: (usize, usize, usize, usize),
}
impl SharedParameters {
    pub fn new(model: &NnlmModel) -> Self {
        Self {
            arrays: [
                &model.input_embeddings,
                &model.hidden_weights,
                &model.hidden_bias,
                &model.output_weights,
            ]
            .map(|a| a.iter().map(|v| AtomicU32::new(v.to_bits())).collect()),
            dimensions: (
                model.vocab_size,
                model.embedding_dimension,
                model.history_length,
                model.hidden_dimension,
            ),
        }
    }
    fn rows_width(&self, kind: ParameterKind) -> (usize, usize) {
        let (v, d, n, h) = self.dimensions;
        match kind {
            ParameterKind::Input => (v, d),
            ParameterKind::Hidden => (h, n * d),
            ParameterKind::Bias => (1, h),
            ParameterKind::Output => (v - 1, h),
        }
    }
    fn row(&self, kind: ParameterKind, id: usize) -> &[AtomicU32] {
        let (_, width) = self.rows_width(kind);
        &self.arrays[kind as usize][id * width..(id + 1) * width]
    }
    fn load_row(&self, kind: ParameterKind, id: usize) -> Vec<Real> {
        self.row(kind, id).iter().map(atomic_float::load).collect()
    }
    pub fn snapshot(&self) -> NnlmModel {
        let (v, d, n, h) = self.dimensions;
        NnlmModel {
            input_embeddings: self.arrays[0].iter().map(atomic_float::load).collect(),
            hidden_weights: self.arrays[1].iter().map(atomic_float::load).collect(),
            hidden_bias: self.arrays[2].iter().map(atomic_float::load).collect(),
            output_weights: self.arrays[3].iter().map(atomic_float::load).collect(),
            vocab_size: v,
            embedding_dimension: d,
            history_length: n,
            hidden_dimension: h,
        }
    }
}

struct Batch {
    rows: Vec<(ParameterKind, usize, Vec<Real>)>,
    acknowledgement: Option<SyncSender<()>>,
}

pub struct ParameterServerShard {
    id: usize,
    count: usize,
    model: Arc<SharedParameters>,
    gamma: Real,
    epsilon: Real,
    accumulators: [Vec<Real>; 4],
}
impl ParameterServerShard {
    pub fn new(
        id: usize,
        count: usize,
        model: Arc<SharedParameters>,
        gamma: Real,
        epsilon: Real,
    ) -> Result<Self, Status> {
        if count == 0
            || id >= count
            || !gamma.is_finite()
            || gamma <= 0.0
            || !epsilon.is_finite()
            || epsilon <= 0.0
        {
            return Err(Status::InvalidArgument);
        }
        let mut accumulators: [Vec<Real>; 4] = std::array::from_fn(|_| Vec::new());
        for kind in KINDS {
            let (rows, width) = model.rows_width(kind);
            let length = rows
                .div_ceil(count)
                .checked_mul(width)
                .ok_or(Status::InvalidArgument)?;
            accumulators[kind as usize]
                .try_reserve_exact(length)
                .map_err(|_| Status::OutOfMemory)?;
            accumulators[kind as usize].resize(length, 0.0);
        }
        Ok(Self {
            id,
            count,
            model,
            gamma,
            epsilon,
            accumulators,
        })
    }
    pub fn apply_gradient(
        &mut self,
        kind: ParameterKind,
        row: usize,
        gradient: &[Real],
    ) -> Result<(), Status> {
        let (rows, width) = self.model.rows_width(kind);
        if row >= rows
            || gradient.len() != width
            || row % self.count != self.id
            || gradient.iter().any(|g| !g.is_finite())
        {
            return Err(Status::InvalidArgument);
        }
        let start = row / self.count * width;
        let accum = &mut self.accumulators[kind as usize][start..start + width];
        for ((parameter, accumulator), g) in
            self.model.row(kind, row).iter().zip(accum).zip(gradient)
        {
            *accumulator += g * g;
            let step = self.gamma * g / (*accumulator + self.epsilon).sqrt();
            atomic_float::store(parameter, atomic_float::load(parameter) - step);
        }
        Ok(())
    }
    fn export_into(&self, arrays: &mut [Vec<Real>; 4]) {
        for kind in KINDS {
            let (rows, width) = self.model.rows_width(kind);
            for row in (self.id..rows).step_by(self.count) {
                let local = row / self.count * width;
                arrays[kind as usize][row * width..(row + 1) * width]
                    .copy_from_slice(&self.accumulators[kind as usize][local..local + width]);
            }
        }
    }
    fn restore(&mut self, arrays: [&[Real]; 4]) {
        for kind in KINDS {
            let (rows, width) = self.model.rows_width(kind);
            for row in (self.id..rows).step_by(self.count) {
                let local = row / self.count * width;
                self.accumulators[kind as usize][local..local + width]
                    .copy_from_slice(&arrays[kind as usize][row * width..(row + 1) * width]);
            }
        }
    }
}

/// Dense hidden parameters and only the sparse rows used in the current batch.
pub struct ReplicaCache {
    shared: Arc<SharedParameters>,
    input: BTreeMap<usize, Vec<Real>>,
    output: BTreeMap<usize, Vec<Real>>,
    hidden: Vec<Real>,
    bias: Vec<Real>,
}
impl ReplicaCache {
    pub fn new(shared: Arc<SharedParameters>) -> Self {
        let hidden = shared.arrays[1].iter().map(atomic_float::load).collect();
        let bias = shared.load_row(ParameterKind::Bias, 0);
        Self {
            shared,
            input: BTreeMap::new(),
            output: BTreeMap::new(),
            hidden,
            bias,
        }
    }
    pub fn gradient(
        &mut self,
        vocabulary: &Vocabulary,
        activation: super::HiddenActivation,
        history: &[usize],
        target: usize,
    ) -> Result<NnlmGradient, Status> {
        forward_backward_parameters(self, vocabulary, activation, history, target)
    }
    pub fn cached_rows(&self) -> (usize, usize) {
        (self.input.len(), self.output.len())
    }
}
impl Parameters for ReplicaCache {
    fn dimensions(&self) -> (usize, usize, usize, usize) {
        self.shared.dimensions
    }
    fn input_row(&mut self, id: usize) -> &[Real] {
        self.input
            .entry(id)
            .or_insert_with(|| self.shared.load_row(ParameterKind::Input, id))
    }
    fn output_row(&mut self, id: usize) -> &[Real] {
        self.output
            .entry(id)
            .or_insert_with(|| self.shared.load_row(ParameterKind::Output, id))
    }
    fn hidden_weights(&self) -> &[Real] {
        &self.hidden
    }
    fn hidden_bias(&self) -> &[Real] {
        &self.bias
    }
}

struct AccumulatedGradient {
    rows: BTreeMap<(ParameterKind, usize), Vec<Real>>,
    targets: usize,
}
impl AccumulatedGradient {
    fn new() -> Self {
        Self {
            rows: BTreeMap::new(),
            targets: 0,
        }
    }
    fn add_row(&mut self, kind: ParameterKind, id: usize, gradient: &[Real]) {
        let row = self
            .rows
            .entry((kind, id))
            .or_insert_with(|| vec![0.0; gradient.len()]);
        for (value, g) in row.iter_mut().zip(gradient) {
            *value += g;
        }
    }
    fn accumulate(&mut self, gradient: &NnlmGradient, projection: usize) {
        for (id, row) in &gradient.embeddings {
            self.add_row(ParameterKind::Input, *id, row);
        }
        for (id, row) in gradient.hidden_weights.chunks(projection).enumerate() {
            self.add_row(ParameterKind::Hidden, id, row);
        }
        self.add_row(ParameterKind::Bias, 0, &gradient.hidden_bias);
        for (id, row) in &gradient.output {
            self.add_row(ParameterKind::Output, *id, row);
        }
        self.targets += 1;
    }
    fn push(&mut self, senders: &[SyncSender<Batch>], synchronous: bool) -> Result<(), Status> {
        // Sum the mini-batch derivatives before squaring in AdaGrad, as the
        // existing W2V backend does. This is a reconstruction choice.
        let mut packets: Vec<_> = (0..senders.len()).map(|_| Vec::new()).collect();
        for ((kind, row), values) in std::mem::take(&mut self.rows) {
            packets[row % senders.len()].push((kind, row, values));
        }
        let (ack, received) = sync_channel(senders.len());
        let mut expected = 0;
        for (sender, rows) in senders.iter().zip(packets) {
            if rows.is_empty() {
                continue;
            }
            sender
                .send(Batch {
                    rows,
                    acknowledgement: synchronous.then(|| ack.clone()),
                })
                .map_err(|_| Status::ThreadError)?;
            expected += 1;
        }
        drop(ack);
        if synchronous {
            for _ in 0..expected {
                received.recv().map_err(|_| Status::ThreadError)?;
            }
        }
        self.targets = 0;
        Ok(())
    }
}

fn partitions(corpus: &Corpus, count: usize) -> Result<Vec<(usize, usize)>, Status> {
    let mut boundaries = vec![0];
    for id in 1..count {
        let offset = (corpus.byte_size as u128 * id as u128 / count as u128) as usize;
        if offset == 0 {
            boundaries.push(0);
            continue;
        }
        let mut file = File::open(&corpus.path).map_err(|_| Status::IoError)?;
        file.seek(SeekFrom::Start((offset - 1) as u64))
            .map_err(|_| Status::IoError)?;
        let mut reader = BufReader::new(file);
        let mut previous = [0];
        reader
            .read_exact(&mut previous)
            .map_err(|_| Status::IoError)?;
        let skipped = if previous[0] == b'\n' {
            0
        } else {
            reader.skip_until(b'\n').map_err(|_| Status::IoError)?
        };
        boundaries.push(offset + skipped);
    }
    boundaries.push(corpus.byte_size);
    Ok(boundaries
        .windows(2)
        .map(|pair| (pair[0], pair[1]))
        .collect())
}

fn digest(config: &NnlmTrainingConfig, downpour: &DownpourConfig) -> String {
    let mut hash = crate::identity::StableDigest::new();
    hash.update(config.digest().as_bytes());
    for value in [
        downpour.parameter_server_shards as u64,
        downpour.mini_batch_targets as u64,
        downpour.adagrad_gamma.to_bits() as u64,
        downpour.adagrad_epsilon.to_bits() as u64,
        downpour.queue_capacity as u64,
        downpour.fetch_interval as u64,
        downpour.push_interval as u64,
    ] {
        hash.value(value);
    }
    hash.finish()
}

pub struct NnlmDownpourTrainingSession {
    corpus: Arc<Corpus>,
    vocabulary: Arc<Vocabulary>,
    config: NnlmTrainingConfig,
    downpour: DownpourConfig,
    shared: Arc<SharedParameters>,
    shards: Vec<ParameterServerShard>,
    replicas: Vec<NnlmReplicaState>,
    descriptor: StateDescriptor,
    completed_epochs: usize,
    processed_tokens: AtomicU64,
    failed: bool,
}

impl NnlmDownpourTrainingSession {
    pub fn create(
        corpus: Arc<Corpus>,
        vocabulary: Arc<Vocabulary>,
        config: NnlmTrainingConfig,
        downpour: DownpourConfig,
        corpus_digest: Option<String>,
    ) -> Result<Self, Status> {
        if downpour.validate() != Status::Ok {
            return Err(Status::InvalidArgument);
        }
        let model = NnlmModel::create(&vocabulary, &config)?;
        let shared = Arc::new(SharedParameters::new(&model));
        let shards = (0..downpour.parameter_server_shards)
            .map(|id| {
                ParameterServerShard::new(
                    id,
                    downpour.parameter_server_shards,
                    shared.clone(),
                    downpour.adagrad_gamma,
                    downpour.adagrad_epsilon,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let replicas = partitions(&corpus, config.thread_count)?
            .into_iter()
            .enumerate()
            .map(|(id, (start, end))| NnlmReplicaState {
                replica_id: id,
                start_byte: start,
                end_byte: end,
                processed_tokens: 0,
                objective_count: 0,
                batch_count: 0,
            })
            .collect();
        let descriptor = StateDescriptor {
            schema_version: 1,
            config_digest: digest(&config, &downpour),
            vocabulary_digest: vocabulary.digest(),
            corpus_digest: corpus_digest.map_or_else(|| corpus.digest(), Ok)?,
        };
        Ok(Self {
            corpus,
            vocabulary,
            config,
            downpour,
            shared,
            shards,
            replicas,
            descriptor,
            completed_epochs: 0,
            processed_tokens: AtomicU64::new(0),
            failed: false,
        })
    }
    pub fn config_digest(&self) -> &str {
        &self.descriptor.config_digest
    }
    pub fn processed_tokens(&self) -> u64 {
        self.processed_tokens.load(Ordering::Relaxed)
    }
    pub fn completed_epochs(&self) -> usize {
        self.completed_epochs
    }
    pub fn is_complete(&self) -> bool {
        self.completed_epochs >= self.config.epochs
    }
    pub fn export_state(&self) -> Result<NnlmDownpourTrainingState, Status> {
        if self.failed {
            return Err(Status::InvalidState);
        }
        let mut arrays = self
            .shared
            .arrays
            .each_ref()
            .map(|array| vec![0.0; array.len()]);
        for shard in &self.shards {
            shard.export_into(&mut arrays);
        }
        let [input, hidden, bias, output] = arrays;
        Ok(NnlmDownpourTrainingState {
            training: NnlmTrainingState {
                descriptor: self.descriptor.clone(),
                vocabulary: self.vocabulary.export_state(),
                model: self.shared.snapshot(),
                completed_epochs: self.completed_epochs,
                processed_tokens: self.processed_tokens.load(Ordering::Relaxed),
            },
            adagrad: NnlmAdaGradState {
                input,
                hidden,
                bias,
                output,
            },
            replicas: self.replicas.clone(),
        })
    }
    pub fn restore(
        corpus: Arc<Corpus>,
        vocabulary: Arc<Vocabulary>,
        config: NnlmTrainingConfig,
        downpour: DownpourConfig,
        state: &NnlmDownpourTrainingState,
        corpus_digest: Option<String>,
    ) -> Result<Self, Status> {
        if state.training.descriptor.schema_version != 1 {
            return Err(Status::SchemaMismatch);
        }
        let corpus_digest = corpus_digest.map_or_else(|| corpus.digest(), Ok)?;
        if state.training.descriptor.config_digest != digest(&config, &downpour)
            || state.training.descriptor.vocabulary_digest != vocabulary.digest()
            || state.training.descriptor.corpus_digest != corpus_digest
            || Vocabulary::restore(&state.training.vocabulary)?.digest() != vocabulary.digest()
        {
            return Err(Status::IdentityMismatch);
        }
        if state.training.completed_epochs > config.epochs
            || !state.training.model.validate(&vocabulary, &config)
        {
            return Err(Status::InvalidState);
        }
        let mut session = Self::create(corpus, vocabulary, config, downpour, Some(corpus_digest))?;
        if state.replicas.len() != session.replicas.len()
            || state
                .replicas
                .iter()
                .zip(&session.replicas)
                .any(|(actual, expected)| {
                    actual.replica_id != expected.replica_id
                        || actual.start_byte != expected.start_byte
                        || actual.end_byte != expected.end_byte
                        || actual.objective_count > actual.processed_tokens
                        || actual.batch_count > actual.objective_count
                        || (actual.start_byte == actual.end_byte && actual.processed_tokens != 0)
                })
            || state
                .replicas
                .iter()
                .try_fold(0u64, |n, replica| n.checked_add(replica.processed_tokens))
                != Some(state.training.processed_tokens)
        {
            return Err(Status::InvalidState);
        }
        for (values, expected) in state
            .adagrad
            .arrays()
            .into_iter()
            .zip(&session.shared.arrays)
        {
            if values.len() != expected.len() || values.iter().any(|v| !v.is_finite() || *v < 0.0) {
                return Err(Status::InvalidState);
            }
        }
        let model = &state.training.model;
        for (destination, source) in session.shared.arrays.iter().zip([
            &model.input_embeddings,
            &model.hidden_weights,
            &model.hidden_bias,
            &model.output_weights,
        ]) {
            for (parameter, value) in destination.iter().zip(source) {
                atomic_float::store(parameter, *value);
            }
        }
        for shard in &mut session.shards {
            shard.restore(state.adagrad.arrays());
        }
        session.replicas.clone_from(&state.replicas);
        session.completed_epochs = state.training.completed_epochs;
        session
            .processed_tokens
            .store(state.training.processed_tokens, Ordering::Relaxed);
        Ok(session)
    }
    pub fn train_epoch(&mut self) -> Result<EpochReport, Status> {
        if self.failed || self.is_complete() {
            return Err(Status::InvalidState);
        }
        let start = Instant::now();
        let before = self.processed_tokens.load(Ordering::Relaxed);
        let mut senders = Vec::new();
        let mut receivers = Vec::new();
        for _ in &self.shards {
            let (tx, rx) = sync_channel::<Batch>(self.downpour.queue_capacity);
            senders.push(tx);
            receivers.push(rx);
        }
        let result = thread::scope(|scope| {
            let mut servers = Vec::new();
            for (shard, receiver) in self.shards.iter_mut().zip(receivers) {
                servers.push(scope.spawn(move || -> Result<(), Status> {
                    while let Ok(batch) = receiver.recv() {
                        for (kind, id, gradient) in batch.rows {
                            shard.apply_gradient(kind, id, &gradient)?;
                        }
                        if let Some(ack) = batch.acknowledgement {
                            ack.send(()).map_err(|_| Status::ThreadError)?;
                        }
                    }
                    Ok(())
                }));
            }
            let mut workers = Vec::new();
            let corpus = &self.corpus;
            let vocab = &self.vocabulary;
            let config = &self.config;
            let downpour = &self.downpour;
            let shared = &self.shared;
            let processed = &self.processed_tokens;
            let epoch = self.completed_epochs + 1;
            for replica in &mut self.replicas {
                let tx = senders.clone();
                workers.push(scope.spawn(move || {
                    run_replica(
                        replica,
                        ReplicaExecution {
                            corpus,
                            vocab,
                            config,
                            downpour,
                            shared,
                            processed,
                            start,
                            epoch,
                            senders: &tx,
                        },
                    )
                }));
            }
            drop(senders);
            let mut result = Ok(Vec::new());
            for worker in workers {
                match worker.join().unwrap_or(Err(Status::ThreadError)) {
                    Ok(report) => {
                        if let Ok(reports) = &mut result {
                            reports.push(report);
                        }
                    }
                    Err(status) => result = Err(status),
                }
            }
            for server in servers {
                if let Err(status) = server.join().unwrap_or(Err(Status::ThreadError)) {
                    result = Err(status);
                }
            }
            result
        });
        let reports = match result {
            Ok(reports) => reports,
            Err(status) => {
                self.failed = true;
                return Err(status);
            }
        };
        self.completed_epochs += 1;
        let processed = self.processed_tokens.load(Ordering::Relaxed);
        let elapsed = start.elapsed().as_secs_f64();
        let mut observations: Vec<_> = reports
            .iter()
            .flat_map(|r| r.observations.iter().cloned())
            .collect();
        observations.sort_by_key(|o| o.processed_tokens);
        Ok(EpochReport {
            epoch: self.completed_epochs,
            epoch_tokens: processed - before,
            processed_tokens: processed,
            learning_rate: self.downpour.adagrad_gamma,
            elapsed_seconds: elapsed,
            tokens_per_second: (processed - before) as f64 / elapsed,
            objective_loss_sum: reports.iter().map(|r| r.loss).sum(),
            objective_loss_count: reports.iter().map(|r| r.targets).sum(),
            observations,
        })
    }
}

struct ReplicaReport {
    loss: f64,
    targets: u64,
    observations: Vec<Observation>,
}
struct ReplicaExecution<'a> {
    corpus: &'a Corpus,
    vocab: &'a Vocabulary,
    config: &'a NnlmTrainingConfig,
    downpour: &'a DownpourConfig,
    shared: &'a Arc<SharedParameters>,
    processed: &'a AtomicU64,
    start: Instant,
    epoch: usize,
    senders: &'a [SyncSender<Batch>],
}
fn run_replica(
    replica: &mut NnlmReplicaState,
    execution: ReplicaExecution<'_>,
) -> Result<ReplicaReport, Status> {
    let ReplicaExecution {
        corpus,
        vocab,
        config,
        downpour,
        shared,
        processed,
        start,
        epoch,
        senders,
    } = execution;
    let mut file = File::open(&corpus.path).map_err(|_| Status::IoError)?;
    file.seek(SeekFrom::Start(replica.start_byte as u64))
        .map_err(|_| Status::IoError)?;
    let mut tokenizer = Tokenizer::new(BufReader::new(
        file.take((replica.end_byte - replica.start_byte) as u64),
    ));
    let mut history = VecDeque::with_capacity(config.history_length);
    let mut cache = ReplicaCache::new(shared.clone());
    let mut gradient = AccumulatedGradient::new();
    let mut report = ReplicaReport {
        loss: 0.0,
        targets: 0,
        observations: Vec::new(),
    };
    let before_tokens = replica.processed_tokens;
    loop {
        let read = tokenizer.read_token()?;
        if !read.token.is_empty() {
            match vocab.find(&read.token) {
                Some(0) | None => history.clear(),
                Some(id) => {
                    processed.fetch_add(1, Ordering::Relaxed);
                    replica.processed_tokens += 1;
                    if history.len() == config.history_length {
                        let input: Vec<_> = history.iter().copied().collect();
                        let target_gradient =
                            cache.gradient(vocab, config.hidden_activation, &input, id)?;
                        report.loss += target_gradient.loss;
                        report.targets += 1;
                        replica.objective_count += 1;
                        gradient.accumulate(
                            &target_gradient,
                            config.embedding_dimension * config.history_length,
                        );
                        if config.observation_interval > 0
                            && replica
                                .objective_count
                                .is_multiple_of(config.observation_interval as u64)
                        {
                            let elapsed = start.elapsed().as_secs_f64();
                            report.observations.push(Observation {
                                epoch,
                                processed_tokens: processed.load(Ordering::Relaxed),
                                learning_rate: downpour.adagrad_gamma,
                                objective_loss_sum: target_gradient.loss,
                                objective_loss_count: 1,
                                elapsed_seconds: elapsed,
                                tokens_per_second: (replica.processed_tokens - before_tokens)
                                    as f64
                                    / elapsed,
                            });
                        }
                        if gradient.targets == downpour.mini_batch_targets {
                            gradient.push(senders, config.thread_count == 1)?;
                            replica.batch_count += 1;
                            cache = ReplicaCache::new(shared.clone());
                        }
                        history.pop_front();
                    }
                    history.push_back(id);
                }
            }
        }
        if read.at_eof {
            break;
        }
    }
    if gradient.targets > 0 {
        gradient.push(senders, config.thread_count == 1)?;
        replica.batch_count += 1;
    }
    Ok(report)
}

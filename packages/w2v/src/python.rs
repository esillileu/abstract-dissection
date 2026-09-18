use crate::training::worker::WorkerState;
use crate::{
    Corpus, EmbeddingKind, EpochReport, HsOutOfRangePolicy, Model, ModelKind, ObjectiveKind,
    Observation, RngAlgorithm, Status, Trainer, TrainingConfig, TrainingSession, TrainingState,
    Vocabulary, VocabularyConfig, VocabularyEntry, VocabularyState,
};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::PyType,
};
use std::{path::PathBuf, sync::Arc};

fn status_error(status: Status) -> PyErr {
    let message = status.as_str();
    match status {
        Status::InvalidArgument
        | Status::InvalidState
        | Status::SchemaMismatch
        | Status::IdentityMismatch => PyValueError::new_err(message),
        _ => PyRuntimeError::new_err(message),
    }
}

fn parse_model_kind(value: &str) -> PyResult<ModelKind> {
    match value {
        "cbow" => Ok(ModelKind::Cbow),
        "skip_gram" => Ok(ModelKind::SkipGram),
        _ => Err(PyValueError::new_err(
            "model_kind must be 'cbow' or 'skip_gram'",
        )),
    }
}

fn parse_objective_kind(value: &str) -> PyResult<ObjectiveKind> {
    match value {
        "hierarchical_softmax" => Ok(ObjectiveKind::HierarchicalSoftmax),
        "negative_sampling" => Ok(ObjectiveKind::NegativeSampling),
        _ => Err(PyValueError::new_err(
            "objective_kind must be 'hierarchical_softmax' or 'negative_sampling'",
        )),
    }
}

fn parse_rng_algorithm(value: &str) -> PyResult<RngAlgorithm> {
    match value {
        "lcg" => Ok(RngAlgorithm::Lcg),
        "xorshift" => Ok(RngAlgorithm::Xorshift),
        _ => Err(PyValueError::new_err(
            "rng_algorithm must be 'lcg' or 'xorshift'",
        )),
    }
}

fn hs_policy(value: &str) -> PyResult<HsOutOfRangePolicy> {
    match value {
        "skip" => Ok(HsOutOfRangePolicy::Skip),
        "use_boundary_value" => Ok(HsOutOfRangePolicy::UseBoundaryValue),
        _ => Err(PyValueError::new_err(
            "hs_out_of_range_policy must be 'skip' or 'use_boundary_value'",
        )),
    }
}

fn array2_from_values<'py>(
    py: Python<'py>,
    values: &[f32],
    rows: usize,
    columns: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    if values.len() != rows * columns {
        return Err(status_error(Status::InvalidState));
    }
    PyArray2::from_vec2(
        py,
        &values
            .chunks(columns)
            .map(<[f32]>::to_vec)
            .collect::<Vec<_>>(),
    )
    .map_err(|_| status_error(Status::OutOfMemory))
}

#[pyclass(name = "Corpus")]
pub struct PyCorpus {
    inner: Arc<Corpus>,
}

#[pymethods]
impl PyCorpus {
    #[new]
    fn new(path: PathBuf) -> PyResult<Self> {
        Ok(Self {
            inner: Arc::new(Corpus::create(path).map_err(status_error)?),
        })
    }

    #[getter]
    fn path(&self) -> String {
        self.inner.path.display().to_string()
    }

    #[getter]
    fn byte_size(&self) -> usize {
        self.inner.byte_size
    }

    fn digest(&self) -> PyResult<String> {
        self.inner.digest().map_err(status_error)
    }
}

#[pyclass(name = "VocabularyConfig", frozen)]
pub struct PyVocabularyConfig {
    inner: VocabularyConfig,
}

#[pymethods]
impl PyVocabularyConfig {
    #[new]
    #[pyo3(signature = (initial_capacity=1000, hash_capacity=30_000_000, min_count=5))]
    fn new(initial_capacity: usize, hash_capacity: usize, min_count: u64) -> PyResult<Self> {
        let inner = VocabularyConfig {
            initial_capacity,
            hash_capacity,
            min_count,
        };
        if inner.validate() != Status::Ok {
            return Err(status_error(Status::InvalidArgument));
        }
        Ok(Self { inner })
    }
}

#[pyclass(name = "Vocabulary")]
pub struct PyVocabulary {
    inner: Arc<Vocabulary>,
}

#[pyclass(name = "VocabularyState", frozen)]
pub struct PyVocabularyState {
    inner: VocabularyState,
}

type VocabularyArrays<'py> = (
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u8>>,
);

fn vocabulary_arrays<'py>(py: Python<'py>, state: &VocabularyState) -> VocabularyArrays<'py> {
    let mut token_bytes = Vec::new();
    let mut token_offsets = vec![0u64];
    let mut counts = Vec::with_capacity(state.entries.len());
    let mut huffman_offsets = vec![0u64];
    let mut huffman_paths = Vec::new();
    let mut huffman_bits = Vec::new();
    for entry in &state.entries {
        token_bytes.extend_from_slice(&entry.token);
        token_offsets.push(token_bytes.len() as u64);
        counts.push(entry.count);
        huffman_paths.extend(entry.huffman_path.iter().map(|&value| value as u64));
        huffman_bits.extend_from_slice(&entry.huffman_bits);
        huffman_offsets.push(huffman_paths.len() as u64);
    }
    (
        token_bytes.into_pyarray(py),
        token_offsets.into_pyarray(py),
        counts.into_pyarray(py),
        huffman_offsets.into_pyarray(py),
        huffman_paths.into_pyarray(py),
        huffman_bits.into_pyarray(py),
    )
}

fn contiguous_slice<'a, T>(array: &'a PyReadonlyArray1<'_, T>) -> PyResult<&'a [T]>
where
    T: numpy::Element,
{
    array
        .as_slice()
        .map_err(|_| PyValueError::new_err("vocabulary state arrays must be C-contiguous"))
}

#[allow(clippy::too_many_arguments)]
fn vocabulary_state_from_arrays(
    token_bytes: &PyReadonlyArray1<'_, u8>,
    token_offsets: &PyReadonlyArray1<'_, u64>,
    counts: &PyReadonlyArray1<'_, u64>,
    huffman_offsets: &PyReadonlyArray1<'_, u64>,
    huffman_paths: &PyReadonlyArray1<'_, u64>,
    huffman_bits: &PyReadonlyArray1<'_, u8>,
    hash_capacity: usize,
    retained_token_count: u64,
) -> PyResult<VocabularyState> {
    let token_bytes = contiguous_slice(token_bytes)?;
    let token_offsets = contiguous_slice(token_offsets)?;
    let counts = contiguous_slice(counts)?;
    let huffman_offsets = contiguous_slice(huffman_offsets)?;
    let huffman_paths = contiguous_slice(huffman_paths)?;
    let huffman_bits = contiguous_slice(huffman_bits)?;
    if token_offsets.len() != counts.len() + 1
        || huffman_offsets.len() != counts.len() + 1
        || token_offsets.first() != Some(&0)
        || huffman_offsets.first() != Some(&0)
        || token_offsets.last() != Some(&(token_bytes.len() as u64))
        || huffman_offsets.last() != Some(&(huffman_paths.len() as u64))
        || huffman_paths.len() != huffman_bits.len()
        || hash_capacity < counts.len()
    {
        return Err(status_error(Status::CorruptData));
    }

    let mut entries = Vec::with_capacity(counts.len());
    for index in 0..counts.len() {
        let token_start =
            usize::try_from(token_offsets[index]).map_err(|_| status_error(Status::CorruptData))?;
        let token_end = usize::try_from(token_offsets[index + 1])
            .map_err(|_| status_error(Status::CorruptData))?;
        let path_start = usize::try_from(huffman_offsets[index])
            .map_err(|_| status_error(Status::CorruptData))?;
        let path_end = usize::try_from(huffman_offsets[index + 1])
            .map_err(|_| status_error(Status::CorruptData))?;
        if token_start >= token_end
            || token_end > token_bytes.len()
            || path_start > path_end
            || path_end > huffman_paths.len()
            || huffman_bits[path_start..path_end]
                .iter()
                .any(|bit| *bit > 1)
        {
            return Err(status_error(Status::CorruptData));
        }
        let paths = huffman_paths[path_start..path_end]
            .iter()
            .map(|&value| usize::try_from(value))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| status_error(Status::CorruptData))?;
        entries.push(VocabularyEntry {
            token: token_bytes[token_start..token_end].to_vec(),
            count: counts[index],
            huffman_path: paths,
            huffman_bits: huffman_bits[path_start..path_end].to_vec(),
        });
    }
    let state = VocabularyState {
        entries,
        hash_capacity,
        retained_token_count,
    };
    Vocabulary::restore(&state).map_err(status_error)?;
    Ok(state)
}

#[pymethods]
impl PyVocabulary {
    #[classmethod]
    #[pyo3(signature = (corpus, config=None))]
    fn build(
        _cls: &Bound<'_, PyType>,
        corpus: &PyCorpus,
        config: Option<&PyVocabularyConfig>,
    ) -> PyResult<Self> {
        let default_config = VocabularyConfig::default();
        let config = config.map_or(&default_config, |value| &value.inner);
        Ok(Self {
            inner: Arc::new(Vocabulary::build(&corpus.inner, config).map_err(status_error)?),
        })
    }

    #[getter]
    fn size(&self) -> usize {
        self.inner.entries.len()
    }

    #[getter]
    fn retained_token_count(&self) -> u64 {
        self.inner.retained_token_count
    }

    fn digest(&self) -> String {
        self.inner.digest()
    }

    fn token_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        self.inner
            .entries
            .iter()
            .flat_map(|entry| entry.token.iter().copied())
            .collect::<Vec<_>>()
            .into_pyarray(py)
    }

    fn token_offsets<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u64>> {
        let mut offset = 0u64;
        let mut offsets = Vec::with_capacity(self.inner.entries.len() + 1);
        offsets.push(offset);
        for entry in &self.inner.entries {
            offset += entry.token.len() as u64;
            offsets.push(offset);
        }
        offsets.into_pyarray(py)
    }

    fn counts<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u64>> {
        self.inner
            .entries
            .iter()
            .map(|entry| entry.count)
            .collect::<Vec<_>>()
            .into_pyarray(py)
    }

    fn export_state(&self) -> PyVocabularyState {
        PyVocabularyState {
            inner: self.inner.export_state(),
        }
    }

    #[classmethod]
    fn restore_state(_cls: &Bound<'_, PyType>, state: &PyVocabularyState) -> PyResult<Self> {
        Ok(Self {
            inner: Arc::new(Vocabulary::restore(&state.inner).map_err(status_error)?),
        })
    }
}

#[pymethods]
impl PyVocabularyState {
    #[getter]
    fn hash_capacity(&self) -> usize {
        self.inner.hash_capacity
    }

    #[getter]
    fn retained_token_count(&self) -> u64 {
        self.inner.retained_token_count
    }

    #[getter]
    fn size(&self) -> usize {
        self.inner.entries.len()
    }

    fn digest(&self) -> PyResult<String> {
        Ok(Vocabulary::restore(&self.inner)
            .map_err(status_error)?
            .digest())
    }

    fn arrays<'py>(&self, py: Python<'py>) -> VocabularyArrays<'py> {
        vocabulary_arrays(py, &self.inner)
    }

    #[classmethod]
    #[pyo3(signature = (
        token_bytes,
        token_offsets,
        counts,
        huffman_offsets,
        huffman_paths,
        huffman_bits,
        hash_capacity,
        retained_token_count
    ))]
    #[allow(clippy::too_many_arguments)]
    fn from_arrays(
        _cls: &Bound<'_, PyType>,
        token_bytes: PyReadonlyArray1<'_, u8>,
        token_offsets: PyReadonlyArray1<'_, u64>,
        counts: PyReadonlyArray1<'_, u64>,
        huffman_offsets: PyReadonlyArray1<'_, u64>,
        huffman_paths: PyReadonlyArray1<'_, u64>,
        huffman_bits: PyReadonlyArray1<'_, u8>,
        hash_capacity: usize,
        retained_token_count: u64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: vocabulary_state_from_arrays(
                &token_bytes,
                &token_offsets,
                &counts,
                &huffman_offsets,
                &huffman_paths,
                &huffman_bits,
                hash_capacity,
                retained_token_count,
            )?,
        })
    }
}

#[pyclass(name = "TrainingConfig", frozen)]
pub struct PyTrainingConfig {
    inner: TrainingConfig,
}

#[pymethods]
impl PyTrainingConfig {
    #[new]
    #[pyo3(signature = (
        model_kind="cbow",
        objective_kind="negative_sampling",
        embedding_dimension=100,
        window_radius=5,
        epochs=5,
        thread_count=12,
        learning_rate_update_interval=10_000,
        observation_interval=0,
        initial_learning_rate=0.05,
        subsampling_threshold=1e-3,
        negative_sample_count=5,
        root_seed=1,
        rng_algorithm="lcg",
        negative_table_size=100_000_000,
        sigmoid_table_size=1000,
        sigmoid_max=6.0,
        hs_out_of_range_policy="skip"
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        model_kind: &str,
        objective_kind: &str,
        embedding_dimension: usize,
        window_radius: usize,
        epochs: usize,
        thread_count: usize,
        learning_rate_update_interval: usize,
        observation_interval: usize,
        initial_learning_rate: f32,
        subsampling_threshold: f32,
        negative_sample_count: usize,
        root_seed: u64,
        rng_algorithm: &str,
        negative_table_size: usize,
        sigmoid_table_size: usize,
        sigmoid_max: f32,
        hs_out_of_range_policy: &str,
    ) -> PyResult<Self> {
        let inner = TrainingConfig {
            model_kind: parse_model_kind(model_kind)?,
            objective_kind: parse_objective_kind(objective_kind)?,
            embedding_dimension,
            window_radius,
            epochs,
            thread_count,
            learning_rate_update_interval,
            observation_interval,
            initial_learning_rate,
            subsampling_threshold,
            negative_sample_count,
            root_seed,
            rng_algorithm: parse_rng_algorithm(rng_algorithm)?,
            negative_table_size,
            sigmoid_table_size,
            sigmoid_max,
            hs_out_of_range_policy: hs_policy(hs_out_of_range_policy)?,
        };
        if inner.validate() != Status::Ok {
            return Err(status_error(Status::InvalidArgument));
        }
        Ok(Self { inner })
    }
}

#[pyclass(name = "Model")]
pub struct PyModel {
    inner: Arc<Model>,
}

#[pymethods]
impl PyModel {
    #[classmethod]
    fn create(
        _cls: &Bound<'_, PyType>,
        vocabulary: &PyVocabulary,
        config: &PyTrainingConfig,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: Arc::new(
                Model::create(
                    &vocabulary.inner,
                    config.inner.embedding_dimension,
                    config.inner.root_seed,
                    config.inner.rng_algorithm,
                )
                .map_err(status_error)?,
            ),
        })
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size
    }

    #[getter]
    fn embedding_dimension(&self) -> usize {
        self.inner.embedding_dimension
    }

    fn input_embeddings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        self.snapshot(py, EmbeddingKind::Input)
    }

    fn output_embeddings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        self.snapshot(py, EmbeddingKind::Output)
    }

    fn restore_input_embeddings(&self, values: PyReadonlyArray2<'_, f32>) -> PyResult<()> {
        self.restore(EmbeddingKind::Input, values)
    }

    fn restore_output_embeddings(&self, values: PyReadonlyArray2<'_, f32>) -> PyResult<()> {
        self.restore(EmbeddingKind::Output, values)
    }
}

impl PyModel {
    fn snapshot<'py>(
        &self,
        py: Python<'py>,
        kind: EmbeddingKind,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let mut values = vec![0.0; self.inner.vocab_size * self.inner.embedding_dimension];
        if self.inner.snapshot_into(kind, &mut values) != Status::Ok {
            return Err(status_error(Status::InvalidState));
        }
        array2_from_values(
            py,
            &values,
            self.inner.vocab_size,
            self.inner.embedding_dimension,
        )
    }

    fn restore(&self, kind: EmbeddingKind, values: PyReadonlyArray2<'_, f32>) -> PyResult<()> {
        let shape = values.shape();
        if shape != [self.inner.vocab_size, self.inner.embedding_dimension] {
            return Err(status_error(Status::InvalidArgument));
        }
        if !values.is_c_contiguous() {
            return Err(PyValueError::new_err(
                "embedding array must be C-contiguous and float32",
            ));
        }
        let values = values.as_slice().map_err(|_| {
            PyValueError::new_err("embedding array must be C-contiguous and float32")
        })?;
        if self.inner.restore_from(kind, values) != Status::Ok {
            return Err(status_error(Status::InvalidState));
        }
        Ok(())
    }
}

#[pyclass(name = "EpochReport", frozen)]
pub struct PyEpochReport {
    #[pyo3(get)]
    epoch: usize,
    #[pyo3(get)]
    epoch_tokens: u64,
    #[pyo3(get)]
    processed_tokens: u64,
    #[pyo3(get)]
    learning_rate: f32,
    #[pyo3(get)]
    elapsed_seconds: f64,
    #[pyo3(get)]
    tokens_per_second: f64,
    #[pyo3(get)]
    objective_loss_sum: f64,
    #[pyo3(get)]
    objective_loss_count: u64,
    observations: Vec<Observation>,
}

#[pymethods]
impl PyEpochReport {
    #[getter]
    fn objective_loss(&self) -> Option<f64> {
        (self.objective_loss_count > 0)
            .then(|| self.objective_loss_sum / self.objective_loss_count as f64)
    }

    fn observations(&self) -> Vec<PyObservation> {
        self.observations
            .iter()
            .cloned()
            .map(PyObservation::from)
            .collect()
    }
}

#[pyclass(name = "Observation", frozen)]
pub struct PyObservation {
    #[pyo3(get)]
    epoch: usize,
    #[pyo3(get)]
    processed_tokens: u64,
    #[pyo3(get)]
    learning_rate: f32,
    #[pyo3(get)]
    objective_loss_sum: f64,
    #[pyo3(get)]
    objective_loss_count: u64,
    #[pyo3(get)]
    elapsed_seconds: f64,
    #[pyo3(get)]
    tokens_per_second: f64,
}

#[pymethods]
impl PyObservation {
    #[getter]
    fn objective_loss(&self) -> f64 {
        self.objective_loss_sum / self.objective_loss_count as f64
    }
}

impl From<Observation> for PyObservation {
    fn from(value: Observation) -> Self {
        Self {
            epoch: value.epoch,
            processed_tokens: value.processed_tokens,
            learning_rate: value.learning_rate,
            objective_loss_sum: value.objective_loss_sum,
            objective_loss_count: value.objective_loss_count,
            elapsed_seconds: value.elapsed_seconds,
            tokens_per_second: value.tokens_per_second,
        }
    }
}

impl From<EpochReport> for PyEpochReport {
    fn from(report: EpochReport) -> Self {
        Self {
            epoch: report.epoch,
            epoch_tokens: report.epoch_tokens,
            processed_tokens: report.processed_tokens,
            learning_rate: report.learning_rate,
            elapsed_seconds: report.elapsed_seconds,
            tokens_per_second: report.tokens_per_second,
            objective_loss_sum: report.objective_loss_sum,
            objective_loss_count: report.objective_loss_count,
            observations: report.observations,
        }
    }
}

#[pyclass(name = "TrainingState", frozen)]
pub struct PyTrainingState {
    inner: TrainingState,
    vocab_size: usize,
    embedding_dimension: usize,
}

#[pyclass(name = "WorkerState", frozen)]
#[derive(Clone)]
pub struct PyWorkerState {
    inner: WorkerState,
}

#[pymethods]
impl PyWorkerState {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        worker_id: usize,
        local_token_count: u64,
        last_learning_rate_update_count: u64,
        learning_rate: f32,
        window_rng_state: u64,
        subsampling_rng_state: u64,
        negative_rng_state: u64,
        objective_count: u64,
    ) -> PyResult<Self> {
        if !learning_rate.is_finite() || learning_rate < 0.0 {
            return Err(status_error(Status::CorruptData));
        }
        Ok(Self {
            inner: WorkerState {
                worker_id,
                local_token_count,
                last_learning_rate_update_count,
                learning_rate,
                window_rng_state,
                subsampling_rng_state,
                negative_rng_state,
                objective_count,
            },
        })
    }

    #[getter]
    fn worker_id(&self) -> usize {
        self.inner.worker_id
    }
    #[getter]
    fn local_token_count(&self) -> u64 {
        self.inner.local_token_count
    }
    #[getter]
    fn last_learning_rate_update_count(&self) -> u64 {
        self.inner.last_learning_rate_update_count
    }
    #[getter]
    fn learning_rate(&self) -> f32 {
        self.inner.learning_rate
    }
    #[getter]
    fn window_rng_state(&self) -> u64 {
        self.inner.window_rng_state
    }
    #[getter]
    fn subsampling_rng_state(&self) -> u64 {
        self.inner.subsampling_rng_state
    }
    #[getter]
    fn negative_rng_state(&self) -> u64 {
        self.inner.negative_rng_state
    }
    #[getter]
    fn objective_count(&self) -> u64 {
        self.inner.objective_count
    }
}

#[pymethods]
impl PyTrainingState {
    #[classmethod]
    #[allow(clippy::too_many_arguments)]
    fn from_parts(
        _cls: &Bound<'_, PyType>,
        schema_version: u32,
        config_digest: String,
        vocabulary_digest: String,
        corpus_digest: String,
        completed_epochs: usize,
        processed_tokens: u64,
        vocabulary_state: &PyVocabularyState,
        input_embeddings: PyReadonlyArray2<'_, f32>,
        output_embeddings: PyReadonlyArray2<'_, f32>,
        workers: Vec<PyWorkerState>,
    ) -> PyResult<Self> {
        if schema_version != crate::trainer::TRAINING_STATE_SCHEMA_VERSION
            || Vocabulary::restore(&vocabulary_state.inner)
                .map_err(status_error)?
                .digest()
                != vocabulary_digest
        {
            return Err(status_error(Status::IdentityMismatch));
        }
        let input_shape = input_embeddings.shape();
        let output_shape = output_embeddings.shape();
        if input_shape.len() != 2 || output_shape != input_shape {
            return Err(status_error(Status::CorruptData));
        }
        let vocab_size = vocabulary_state.inner.entries.len();
        if input_shape[0] != vocab_size || input_shape[1] == 0 {
            return Err(status_error(Status::CorruptData));
        }
        if !input_embeddings.is_c_contiguous() || !output_embeddings.is_c_contiguous() {
            return Err(PyValueError::new_err(
                "embedding arrays must be C-contiguous and float32",
            ));
        }
        let input_values = input_embeddings.as_slice().map_err(|_| {
            PyValueError::new_err("embedding arrays must be C-contiguous and float32")
        })?;
        let output_values = output_embeddings.as_slice().map_err(|_| {
            PyValueError::new_err("embedding arrays must be C-contiguous and float32")
        })?;
        if input_values
            .iter()
            .chain(output_values)
            .any(|value| !value.is_finite())
        {
            return Err(status_error(Status::CorruptData));
        }
        Ok(Self {
            inner: TrainingState {
                descriptor: crate::trainer::StateDescriptor {
                    schema_version,
                    config_digest,
                    vocabulary_digest,
                    corpus_digest,
                },
                completed_epochs,
                processed_tokens,
                vocabulary: vocabulary_state.inner.clone(),
                input_embeddings: input_values.to_vec(),
                output_embeddings: output_values.to_vec(),
                workers: workers.into_iter().map(|worker| worker.inner).collect(),
            },
            vocab_size,
            embedding_dimension: input_shape[1],
        })
    }

    #[getter]
    fn schema_version(&self) -> u32 {
        self.inner.descriptor.schema_version
    }
    #[getter]
    fn config_digest(&self) -> String {
        self.inner.descriptor.config_digest.clone()
    }
    #[getter]
    fn vocabulary_digest(&self) -> String {
        self.inner.descriptor.vocabulary_digest.clone()
    }
    #[getter]
    fn corpus_digest(&self) -> String {
        self.inner.descriptor.corpus_digest.clone()
    }
    #[getter]
    fn completed_epochs(&self) -> usize {
        self.inner.completed_epochs
    }

    #[getter]
    fn processed_tokens(&self) -> u64 {
        self.inner.processed_tokens
    }

    fn input_embeddings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        array2_from_values(
            py,
            &self.inner.input_embeddings,
            self.vocab_size,
            self.embedding_dimension,
        )
    }

    fn output_embeddings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        array2_from_values(
            py,
            &self.inner.output_embeddings,
            self.vocab_size,
            self.embedding_dimension,
        )
    }

    fn vocabulary_state(&self) -> PyVocabularyState {
        PyVocabularyState {
            inner: self.inner.vocabulary.clone(),
        }
    }

    fn workers(&self) -> Vec<PyWorkerState> {
        self.inner
            .workers
            .iter()
            .cloned()
            .map(|inner| PyWorkerState { inner })
            .collect()
    }
}

#[pyclass(name = "TrainingSession")]
pub struct PyTrainingSession {
    inner: TrainingSession,
    vocab_size: usize,
    embedding_dimension: usize,
}

#[pymethods]
impl PyTrainingSession {
    #[new]
    fn new(
        corpus: &PyCorpus,
        vocabulary: &PyVocabulary,
        model: &PyModel,
        config: &PyTrainingConfig,
    ) -> PyResult<Self> {
        let trainer = Arc::new(
            Trainer::create(
                Arc::clone(&corpus.inner),
                Arc::clone(&vocabulary.inner),
                Arc::clone(&model.inner),
                &config.inner,
            )
            .map_err(status_error)?,
        );
        Ok(Self {
            inner: TrainingSession::create(trainer).map_err(status_error)?,
            vocab_size: model.inner.vocab_size,
            embedding_dimension: model.inner.embedding_dimension,
        })
    }

    #[classmethod]
    fn restore(
        _cls: &Bound<'_, PyType>,
        corpus: &PyCorpus,
        vocabulary: &PyVocabulary,
        model: &PyModel,
        config: &PyTrainingConfig,
        state: &PyTrainingState,
    ) -> PyResult<Self> {
        let trainer = Arc::new(
            Trainer::create(
                Arc::clone(&corpus.inner),
                Arc::clone(&vocabulary.inner),
                Arc::clone(&model.inner),
                &config.inner,
            )
            .map_err(status_error)?,
        );
        Ok(Self {
            inner: TrainingSession::restore(trainer, &state.inner).map_err(status_error)?,
            vocab_size: model.inner.vocab_size,
            embedding_dimension: model.inner.embedding_dimension,
        })
    }

    #[getter]
    fn completed_epochs(&self) -> usize {
        self.inner.completed_epochs()
    }

    #[getter]
    fn is_complete(&self) -> bool {
        self.inner.is_complete()
    }

    #[pyo3(signature = (callback=None))]
    fn train_epoch(
        &mut self,
        py: Python<'_>,
        callback: Option<Py<PyAny>>,
    ) -> PyResult<PyEpochReport> {
        let report = py
            .allow_threads(|| self.inner.train_epoch())
            .map_err(status_error)?;
        let report = PyEpochReport::from(report);
        if let Some(callback) = callback {
            let callback_report = Py::new(
                py,
                PyEpochReport {
                    epoch: report.epoch,
                    epoch_tokens: report.epoch_tokens,
                    processed_tokens: report.processed_tokens,
                    learning_rate: report.learning_rate,
                    elapsed_seconds: report.elapsed_seconds,
                    tokens_per_second: report.tokens_per_second,
                    objective_loss_sum: report.objective_loss_sum,
                    objective_loss_count: report.objective_loss_count,
                    observations: report.observations.clone(),
                },
            )?;
            callback.call1(py, (callback_report,))?;
        }
        Ok(report)
    }

    fn export_state(&self) -> PyResult<PyTrainingState> {
        Ok(PyTrainingState {
            inner: self.inner.export_state().map_err(status_error)?,
            vocab_size: self.vocab_size,
            embedding_dimension: self.embedding_dimension,
        })
    }
}

#[pymodule]
fn w2v(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCorpus>()?;
    m.add_class::<PyVocabularyConfig>()?;
    m.add_class::<PyVocabulary>()?;
    m.add_class::<PyVocabularyState>()?;
    m.add_class::<PyTrainingConfig>()?;
    m.add_class::<PyModel>()?;
    m.add_class::<PyEpochReport>()?;
    m.add_class::<PyObservation>()?;
    m.add_class::<PyTrainingState>()?;
    m.add_class::<PyWorkerState>()?;
    m.add_class::<PyTrainingSession>()?;
    Ok(())
}

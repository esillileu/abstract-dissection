use super::*;
use crate::nnlm::{
    HiddenActivation, NnlmAdaGradState, NnlmDownpourTrainingSession, NnlmDownpourTrainingState,
    NnlmModel, NnlmReplicaState, NnlmTrainingConfig, NnlmTrainingSession, NnlmTrainingState,
};

#[pyclass(name = "NnlmTrainingConfig", frozen)]
pub struct PyNnlmConfig {
    inner: NnlmTrainingConfig,
}

#[pymethods]
impl PyNnlmConfig {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (*, embedding_dimension, history_length, hidden_dimension, hidden_activation, epochs, thread_count, initial_learning_rate, root_seed, objective_kind="hierarchical_softmax", observation_interval=0, rng_algorithm="lcg"))]
    fn new(
        embedding_dimension: usize,
        history_length: usize,
        hidden_dimension: usize,
        hidden_activation: &str,
        epochs: usize,
        thread_count: usize,
        initial_learning_rate: f32,
        root_seed: u64,
        objective_kind: &str,
        observation_interval: usize,
        rng_algorithm: &str,
    ) -> PyResult<Self> {
        let activation = match hidden_activation {
            "tanh" => HiddenActivation::Tanh,
            "sigmoid" => HiddenActivation::Sigmoid,
            _ => {
                return Err(PyValueError::new_err(
                    "hidden_activation must be tanh or sigmoid",
                ));
            }
        };
        let inner = NnlmTrainingConfig {
            embedding_dimension,
            history_length,
            hidden_dimension,
            hidden_activation: activation,
            epochs,
            thread_count,
            initial_learning_rate,
            root_seed,
            objective_kind: parse_objective_kind(objective_kind)?,
            observation_interval,
            rng_algorithm: parse_rng_algorithm(rng_algorithm)?,
        };
        if inner.validate() != Status::Ok {
            return Err(status_error(Status::InvalidArgument));
        }
        Ok(Self { inner })
    }
}

#[pyclass(name = "NnlmTrainingState", frozen)]
pub struct PyNnlmState {
    inner: NnlmTrainingState,
    downpour: Option<(NnlmAdaGradState, Vec<NnlmReplicaState>)>,
}

#[pymethods]
impl PyNnlmState {
    #[getter]
    fn optimizer_kind(&self) -> &'static str {
        if self.downpour.is_some() {
            "adagrad"
        } else {
            "sgd"
        }
    }

    fn replicas(&self) -> Vec<PyNnlmReplicaState> {
        self.downpour
            .as_ref()
            .map_or_else(Vec::new, |(_, replicas)| {
                replicas
                    .iter()
                    .cloned()
                    .map(|inner| PyNnlmReplicaState { inner })
                    .collect()
            })
    }
    fn adagrad_arrays<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        let (state, _) = self
            .downpour
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("SGD has no AdaGrad state"))?;
        let model = &self.inner.model;
        pyo3::types::PyTuple::new(
            py,
            [
                array2_from_values(
                    py,
                    &state.input,
                    model.vocab_size,
                    model.embedding_dimension,
                )?
                .into_any(),
                array2_from_values(
                    py,
                    &state.hidden,
                    model.hidden_dimension,
                    model.history_length * model.embedding_dimension,
                )?
                .into_any(),
                state.bias.clone().into_pyarray(py).into_any(),
                array2_from_values(
                    py,
                    &state.output,
                    model.vocab_size - 1,
                    model.hidden_dimension,
                )?
                .into_any(),
            ],
        )
    }
    #[classmethod]
    fn from_downpour_parts(
        _cls: &Bound<'_, PyType>,
        training_state: &PyNnlmState,
        input: PyReadonlyArray2<f32>,
        hidden: PyReadonlyArray2<f32>,
        bias: PyReadonlyArray1<f32>,
        output: PyReadonlyArray2<f32>,
        replicas: Vec<PyNnlmReplicaState>,
    ) -> PyResult<Self> {
        let model = &training_state.inner.model;
        if training_state.downpour.is_some()
            || input.shape() != [model.vocab_size, model.embedding_dimension]
            || hidden.shape()
                != [
                    model.hidden_dimension,
                    model.history_length * model.embedding_dimension,
                ]
            || bias.shape() != [model.hidden_dimension]
            || output.shape() != [model.vocab_size - 1, model.hidden_dimension]
        {
            return Err(status_error(Status::InvalidArgument));
        }
        let adagrad = NnlmAdaGradState {
            input: input.as_slice()?.to_vec(),
            hidden: hidden.as_slice()?.to_vec(),
            bias: bias.as_slice()?.to_vec(),
            output: output.as_slice()?.to_vec(),
        };
        if adagrad
            .input
            .iter()
            .chain(&adagrad.hidden)
            .chain(&adagrad.bias)
            .chain(&adagrad.output)
            .any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(status_error(Status::CorruptData));
        }
        Ok(Self {
            inner: training_state.inner.clone(),
            downpour: Some((adagrad, replicas.into_iter().map(|r| r.inner).collect())),
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
    #[getter]
    fn history_length(&self) -> usize {
        self.inner.model.history_length
    }
    fn vocabulary_state(&self) -> PyVocabularyState {
        PyVocabularyState {
            inner: self.inner.vocabulary.clone(),
        }
    }
    fn input_embeddings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let model = &self.inner.model;
        array2_from_values(
            py,
            &model.input_embeddings,
            model.vocab_size,
            model.embedding_dimension,
        )
    }
    fn hidden_weights<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let model = &self.inner.model;
        array2_from_values(
            py,
            &model.hidden_weights,
            model.hidden_dimension,
            model.history_length * model.embedding_dimension,
        )
    }
    fn hidden_bias<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.model.hidden_bias.clone().into_pyarray(py)
    }
    fn output_weights<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let model = &self.inner.model;
        array2_from_values(
            py,
            &model.output_weights,
            model.vocab_size - 1,
            model.hidden_dimension,
        )
    }
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
        history_length: usize,
        vocabulary: &PyVocabularyState,
        input_embeddings: PyReadonlyArray2<f32>,
        hidden_weights: PyReadonlyArray2<f32>,
        hidden_bias: PyReadonlyArray1<f32>,
        output_weights: PyReadonlyArray2<f32>,
    ) -> PyResult<Self> {
        let v = vocabulary.inner.entries.len();
        let d = input_embeddings.shape()[1];
        let h = hidden_bias.len();
        let p = history_length
            .checked_mul(d)
            .ok_or_else(|| status_error(Status::InvalidArgument))?;
        if v < 2
            || d == 0
            || h == 0
            || history_length == 0
            || input_embeddings.shape() != [v, d]
            || hidden_weights.shape() != [h, p]
            || output_weights.shape() != [v - 1, h]
        {
            return Err(status_error(Status::InvalidArgument));
        }
        let model = NnlmModel {
            vocab_size: v,
            embedding_dimension: d,
            hidden_dimension: h,
            history_length,
            input_embeddings: input_embeddings.as_slice()?.to_vec(),
            hidden_weights: hidden_weights.as_slice()?.to_vec(),
            hidden_bias: hidden_bias.as_slice()?.to_vec(),
            output_weights: output_weights.as_slice()?.to_vec(),
        };
        if model
            .input_embeddings
            .iter()
            .chain(&model.hidden_weights)
            .chain(&model.hidden_bias)
            .chain(&model.output_weights)
            .any(|v| !v.is_finite())
        {
            return Err(status_error(Status::CorruptData));
        }
        if Vocabulary::restore(&vocabulary.inner)
            .map_err(status_error)?
            .digest()
            != vocabulary_digest
        {
            return Err(status_error(Status::IdentityMismatch));
        }
        Ok(Self {
            downpour: None,
            inner: NnlmTrainingState {
                descriptor: crate::StateDescriptor {
                    schema_version,
                    config_digest,
                    vocabulary_digest,
                    corpus_digest,
                },
                vocabulary: vocabulary.inner.clone(),
                model,
                completed_epochs,
                processed_tokens,
            },
        })
    }
}

#[pyclass(name = "NnlmTrainingSession")]
pub struct PyNnlmSession {
    inner: NnlmTrainingSession,
}

#[pymethods]
impl PyNnlmSession {
    #[new]
    #[pyo3(signature = (corpus, vocabulary, config, corpus_digest=None))]
    fn new(
        corpus: &PyCorpus,
        vocabulary: &PyVocabulary,
        config: &PyNnlmConfig,
        corpus_digest: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: NnlmTrainingSession::create(
                corpus.inner.clone(),
                vocabulary.inner.clone(),
                config.inner.clone(),
                corpus_digest,
            )
            .map_err(status_error)?,
        })
    }
    #[classmethod]
    #[pyo3(signature = (corpus, vocabulary, config, state, corpus_digest=None))]
    fn restore(
        _cls: &Bound<'_, PyType>,
        corpus: &PyCorpus,
        vocabulary: &PyVocabulary,
        config: &PyNnlmConfig,
        state: &PyNnlmState,
        corpus_digest: Option<String>,
    ) -> PyResult<Self> {
        if state.downpour.is_some() {
            return Err(PyValueError::new_err(
                "AdaGrad state requires a Downpour session",
            ));
        }
        Ok(Self {
            inner: NnlmTrainingSession::restore(
                corpus.inner.clone(),
                vocabulary.inner.clone(),
                config.inner.clone(),
                &state.inner,
                corpus_digest,
            )
            .map_err(status_error)?,
        })
    }
    #[getter]
    fn config_digest(&self) -> String {
        self.inner.state.descriptor.config_digest.clone()
    }
    #[getter]
    fn processed_tokens(&self) -> u64 {
        self.inner.state.processed_tokens
    }
    #[getter]
    fn completed_epochs(&self) -> usize {
        self.inner.state.completed_epochs
    }
    #[getter]
    fn is_complete(&self) -> bool {
        self.inner.is_complete()
    }
    fn export_state(&self) -> PyNnlmState {
        PyNnlmState {
            downpour: None,
            inner: self.inner.state.clone(),
        }
    }
    fn train_epoch(&mut self, py: Python<'_>) -> PyResult<PyEpochReport> {
        let report = py
            .allow_threads(|| self.inner.train_epoch())
            .map_err(status_error)?;
        Ok(PyEpochReport {
            epoch: report.epoch,
            epoch_tokens: report.epoch_tokens,
            processed_tokens: report.processed_tokens,
            learning_rate: report.learning_rate,
            elapsed_seconds: report.elapsed_seconds,
            tokens_per_second: report.tokens_per_second,
            objective_loss_sum: report.objective_loss_sum,
            objective_loss_count: report.objective_loss_count,
            observations: report.observations,
        })
    }
}

#[pyclass(name = "NnlmReplicaState", frozen)]
#[derive(Clone)]
pub struct PyNnlmReplicaState {
    inner: NnlmReplicaState,
}
#[pymethods]
impl PyNnlmReplicaState {
    #[new]
    fn new(
        replica_id: usize,
        start_byte: usize,
        end_byte: usize,
        processed_tokens: u64,
        objective_count: u64,
        batch_count: u64,
    ) -> PyResult<Self> {
        if start_byte > end_byte {
            return Err(status_error(Status::InvalidArgument));
        }
        Ok(Self {
            inner: NnlmReplicaState {
                replica_id,
                start_byte,
                end_byte,
                processed_tokens,
                objective_count,
                batch_count,
            },
        })
    }
    #[getter]
    fn replica_id(&self) -> usize {
        self.inner.replica_id
    }
    #[getter]
    fn start_byte(&self) -> usize {
        self.inner.start_byte
    }
    #[getter]
    fn end_byte(&self) -> usize {
        self.inner.end_byte
    }
    #[getter]
    fn processed_tokens(&self) -> u64 {
        self.inner.processed_tokens
    }
    #[getter]
    fn objective_count(&self) -> u64 {
        self.inner.objective_count
    }
    #[getter]
    fn batch_count(&self) -> u64 {
        self.inner.batch_count
    }
}

#[pyclass(name = "NnlmDownpourTrainingSession")]
pub struct PyNnlmDownpourSession {
    inner: NnlmDownpourTrainingSession,
}
#[pymethods]
impl PyNnlmDownpourSession {
    #[new]
    #[pyo3(signature = (corpus, vocabulary, config, downpour, corpus_digest=None))]
    fn new(
        corpus: &PyCorpus,
        vocabulary: &PyVocabulary,
        config: &PyNnlmConfig,
        downpour: &PyDownpourConfig,
        corpus_digest: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: NnlmDownpourTrainingSession::create(
                corpus.inner.clone(),
                vocabulary.inner.clone(),
                config.inner.clone(),
                downpour.inner.clone(),
                corpus_digest,
            )
            .map_err(status_error)?,
        })
    }
    #[classmethod]
    #[pyo3(signature = (corpus, vocabulary, config, downpour, state, corpus_digest=None))]
    fn restore(
        _cls: &Bound<'_, PyType>,
        corpus: &PyCorpus,
        vocabulary: &PyVocabulary,
        config: &PyNnlmConfig,
        downpour: &PyDownpourConfig,
        state: &PyNnlmState,
        corpus_digest: Option<String>,
    ) -> PyResult<Self> {
        let (adagrad, replicas) = state
            .downpour
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Downpour requires AdaGrad and replica state"))?;
        Ok(Self {
            inner: NnlmDownpourTrainingSession::restore(
                corpus.inner.clone(),
                vocabulary.inner.clone(),
                config.inner.clone(),
                downpour.inner.clone(),
                &NnlmDownpourTrainingState {
                    training: state.inner.clone(),
                    adagrad: adagrad.clone(),
                    replicas: replicas.clone(),
                },
                corpus_digest,
            )
            .map_err(status_error)?,
        })
    }
    #[getter]
    fn config_digest(&self) -> String {
        self.inner.config_digest().to_owned()
    }
    #[getter]
    fn processed_tokens(&self) -> u64 {
        self.inner.processed_tokens()
    }
    #[getter]
    fn completed_epochs(&self) -> usize {
        self.inner.completed_epochs()
    }
    #[getter]
    fn is_complete(&self) -> bool {
        self.inner.is_complete()
    }
    fn export_state(&self) -> PyResult<PyNnlmState> {
        let state = self.inner.export_state().map_err(status_error)?;
        Ok(PyNnlmState {
            inner: state.training,
            downpour: Some((state.adagrad, state.replicas)),
        })
    }
    fn train_epoch(&mut self, py: Python<'_>) -> PyResult<PyEpochReport> {
        let report = py
            .allow_threads(|| self.inner.train_epoch())
            .map_err(status_error)?;
        Ok(PyEpochReport {
            epoch: report.epoch,
            epoch_tokens: report.epoch_tokens,
            processed_tokens: report.processed_tokens,
            learning_rate: report.learning_rate,
            elapsed_seconds: report.elapsed_seconds,
            tokens_per_second: report.tokens_per_second,
            objective_loss_sum: report.objective_loss_sum,
            objective_loss_count: report.objective_loss_count,
            observations: report.observations,
        })
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyNnlmConfig>()?;
    module.add_class::<PyNnlmState>()?;
    module.add_class::<PyNnlmSession>()?;
    module.add_class::<PyNnlmReplicaState>()?;
    module.add_class::<PyNnlmDownpourSession>()?;
    Ok(())
}

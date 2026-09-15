#include "atomic_float.h"
#include "rng.h"
#include "w2v/model.h"

#include <stdlib.h>

static void report_status(Status *status, Status value)
{
    if (status != NULL)
    {
        *status = value;
    }
}

Model *model_create(
    const Vocabulary *vocab,
    size_t embedding_dimension,
    uint64_t root_seed,
    RngAlgorithm rng_algorithm,
    Status *status)
{
    report_status(status, STATUS_INVALID_ARGUMENT);
    if (vocab == NULL || vocab->size == 0 ||
        embedding_dimension == 0 ||
        (rng_algorithm != RNG_LCG && rng_algorithm != RNG_XORSHIFT) ||
        vocab->size > SIZE_MAX / embedding_dimension)
    {
        return NULL;
    }

    size_t element_count = vocab->size * embedding_dimension;
    if (element_count > SIZE_MAX / sizeof(_Atomic uint32_t))
    {
        return NULL;
    }

    Model *model = calloc(1, sizeof(*model));
    if (model == NULL)
    {
        report_status(status, STATUS_OUT_OF_MEMORY);
        return NULL;
    }

    model->input_embeddings = malloc(
        element_count * sizeof(*model->input_embeddings));
    model->output_embeddings = malloc(
        element_count * sizeof(*model->output_embeddings));
    if (model->input_embeddings == NULL || model->output_embeddings == NULL)
    {
        model_destroy(&model);
        report_status(status, STATUS_OUT_OF_MEMORY);
        return NULL;
    }

    model->vocab_size = vocab->size;
    model->embedding_dimension = embedding_dimension;

    Rng initialization_rng;
    uint64_t initialization_seed = derive_seed(root_seed, 0, RNG_MODEL);

    rng_init(&initialization_rng, initialization_seed, rng_algorithm);
    for (size_t element_index = 0;
         element_index < element_count;
         element_index++)
    {
        real random_value = rng_uniform(&initialization_rng);
        real initialized_value =
            (random_value - 0.5f) / (real)embedding_dimension;

        atomic_float_store(
            &model->input_embeddings[element_index],
            initialized_value);
        atomic_float_store(&model->output_embeddings[element_index], 0);
    }

    report_status(status, STATUS_OK);
    return model;
}

void model_destroy(Model **model)
{
    if (model == NULL || *model == NULL)
    {
        return;
    }

    free((*model)->input_embeddings);
    free((*model)->output_embeddings);
    free(*model);
    *model = NULL;
}

Status model_snapshot(
    const Model *model,
    EmbeddingKind kind,
    real *destination,
    size_t element_count)
{
    if (model == NULL || destination == NULL)
    {
        return STATUS_INVALID_ARGUMENT;
    }
    if (kind != INPUT_EMBEDDING && kind != OUTPUT_EMBEDDING)
    {
        return STATUS_INVALID_ARGUMENT;
    }

    size_t required_count =
        model->vocab_size * model->embedding_dimension;
    if (element_count != required_count)
    {
        return STATUS_INVALID_ARGUMENT;
    }

    const _Atomic uint32_t *source = model->input_embeddings;
    if (kind == OUTPUT_EMBEDDING)
    {
        source = model->output_embeddings;
    }
    for (size_t element_index = 0;
         element_index < element_count;
         element_index++)
    {
        destination[element_index] = atomic_float_load(
            &source[element_index]);
    }
    return STATUS_OK;
}

#ifndef W2V_MODEL_H
#define W2V_MODEL_H

#include "config.h"
#include "vocab.h"

#include <stdatomic.h>

typedef enum
{
    INPUT_EMBEDDING = 0,
    OUTPUT_EMBEDDING = 1
} EmbeddingKind;

typedef struct
{
    _Atomic uint32_t *input_embeddings;
    _Atomic uint32_t *output_embeddings;
    size_t vocab_size;
    size_t embedding_dimension;
} Model;

typedef struct
{
    real *values;
    size_t size;
    real max;
} SigmoidTable;

Model *model_create(
    const Vocabulary *vocab,
    size_t embedding_dimension,
    uint64_t root_seed,
    RngAlgorithm rng_algorithm,
    Status *status);
void model_destroy(Model **model);
Status model_snapshot(
    const Model *model,
    EmbeddingKind kind,
    real *destination,
    size_t element_count);

#endif

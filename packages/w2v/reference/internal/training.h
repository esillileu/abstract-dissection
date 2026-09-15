#ifndef W2V_INTERNAL_TRAINING_H
#define W2V_INTERNAL_TRAINING_H

#include "rng.h"
#include "w2v/trainer.h"

#include <stdio.h>

typedef struct
{
    FILE *file;
    size_t worker_id;
    size_t shard_start;
    size_t sentence[MAX_SENTENCE_LENGTH];
    size_t sentence_length;
    size_t sentence_position;
    real *hidden;
    real *hidden_gradient;
    uint64_t local_token_count;
    uint64_t epoch_token_count;
    uint64_t last_learning_rate_update_count;
    real learning_rate;
    Rng window_rng;
    Rng subsampling_rng;
    Rng negative_rng;
} Worker;

typedef struct
{
    size_t target_token;
    real learning_rate;
    Trainer *trainer;
    Worker *worker;
} ModelStep;

typedef struct
{
    Trainer *trainer;
    size_t worker_id;
    Status status;
} ThreadArguments;

Status worker_initialize(Worker *worker, const Trainer *trainer, size_t worker_id);
void worker_free(Worker *worker);
Status worker_seek_to_shard(FILE *file, size_t shard_start);
Status worker_fill_sentence(Worker *worker, Trainer *trainer, int *finished);
void worker_update_learning_rate(Worker *worker, const Trainer *trainer);
int context_position(
    const Worker *worker,
    size_t radius,
    size_t offset,
    size_t *position);
size_t context_radius(ModelStep *step);
void objective_train(
    ModelStep *step,
    const real *hidden,
    real *hidden_gradient);
real objective_score(
    const real *hidden,
    const _Atomic uint32_t *output_row,
    size_t dimension);
void objective_apply_update(
    const real *hidden,
    real *hidden_gradient,
    _Atomic uint32_t *output_row,
    size_t dimension,
    real gradient_scale);
void hierarchical_softmax_train(
    ModelStep *step,
    const real *hidden,
    real *hidden_gradient);
void negative_sampling_train(
    ModelStep *step,
    const real *hidden,
    real *hidden_gradient);
void cbow_train(ModelStep *step);
void skip_gram_train(ModelStep *step);
void *worker_run(void *opaque_arguments);

#endif

#include "tokenizer.h"
#include "training.h"

#include <limits.h>
#include <math.h>
#include <stdlib.h>

Status worker_seek_to_shard(FILE *file, size_t shard_start)
{
    if (shard_start > (size_t)LONG_MAX)
    {
        return STATUS_IO_ERROR;
    }
    return fseek(file, (long)shard_start, SEEK_SET) == 0
               ? STATUS_OK
               : STATUS_IO_ERROR;
}

void worker_free(Worker *worker)
{
    if (worker->file != NULL)
    {
        fclose(worker->file);
    }
    free(worker->hidden);
    free(worker->hidden_gradient);
    *worker = (Worker){0};
}

Status worker_initialize(
    Worker *worker,
    const Trainer *trainer,
    size_t worker_id)
{
    if (worker == NULL || trainer == NULL ||
        worker_id >= trainer->config.thread_count)
    {
        return STATUS_INVALID_ARGUMENT;
    }

    *worker = (Worker){0};
    worker->worker_id = worker_id;
    worker->learning_rate = trainer->config.initial_learning_rate;
    worker->shard_start =
        trainer->corpus->byte_size /
        trainer->config.thread_count * worker_id;

    size_t dimension = trainer->config.embedding_dimension;

    worker->hidden = calloc(dimension, sizeof(*worker->hidden));
    worker->hidden_gradient = calloc(
        dimension,
        sizeof(*worker->hidden_gradient));
    if (worker->hidden == NULL || worker->hidden_gradient == NULL)
    {
        worker_free(worker);
        return STATUS_OUT_OF_MEMORY;
    }

    worker->file = fopen(trainer->corpus->path, "rb");
    if (worker->file == NULL)
    {
        worker_free(worker);
        return STATUS_IO_ERROR;
    }

    Status status = worker_seek_to_shard(
        worker->file,
        worker->shard_start);
    if (status != STATUS_OK)
    {
        worker_free(worker);
        return status;
    }

    rng_init(
        &worker->window_rng,
        derive_seed(trainer->config.root_seed, worker_id, RNG_WINDOW),
        trainer->config.rng_algorithm);
    rng_init(
        &worker->subsampling_rng,
        derive_seed(trainer->config.root_seed, worker_id, RNG_SUBSAMPLE),
        trainer->config.rng_algorithm);
    rng_init(
        &worker->negative_rng,
        derive_seed(trainer->config.root_seed, worker_id, RNG_NEGATIVE),
        trainer->config.rng_algorithm);
    return STATUS_OK;
}

static int token_should_be_discarded(
    Worker *worker,
    const Trainer *trainer,
    size_t token)
{
    if (trainer->config.subsampling_threshold <= 0)
    {
        return 0;
    }

    double frequency =
        (double)trainer->vocab->entries[token].count /
        (double)trainer->vocab->retained_token_count;
    double threshold = trainer->config.subsampling_threshold;
    double keep_probability =
        (sqrt(frequency / threshold) + 1.0) * threshold / frequency;
    double random_value = rng_uniform(&worker->subsampling_rng);

    return keep_probability < random_value;
}

Status worker_fill_sentence(
    Worker *worker,
    Trainer *trainer,
    int *finished)
{
    worker->sentence_length = 0;
    worker->sentence_position = 0;
    *finished = 0;

    char token_text[MAX_TOKEN_LENGTH];
    int at_eof = 0;
    while (worker->sentence_length < MAX_SENTENCE_LENGTH)
    {
        Status status = token_read(worker->file, token_text, &at_eof);
        if (status != STATUS_OK)
        {
            return status;
        }
        if (at_eof)
        {
            *finished = 1;
            break;
        }

        size_t token = vocab_find(trainer->vocab, token_text);
        if (token == VOCABULARY_NOT_FOUND)
        {
            continue;
        }

        worker->local_token_count++;
        worker->epoch_token_count++;
        atomic_fetch_add_explicit(
            &trainer->processed_tokens,
            1,
            memory_order_relaxed);
        if (token == 0)
        {
            break;
        }
        if (token_should_be_discarded(worker, trainer, token))
        {
            continue;
        }

        worker->sentence[worker->sentence_length] = token;
        worker->sentence_length++;
    }
    return STATUS_OK;
}

void worker_update_learning_rate(Worker *worker, const Trainer *trainer)
{
    uint64_t tokens_since_update =
        worker->local_token_count -
        worker->last_learning_rate_update_count;
    if (tokens_since_update < trainer->config.learning_rate_update_interval)
    {
        return;
    }

    uint64_t processed_tokens = atomic_load_explicit(
        &trainer->processed_tokens,
        memory_order_relaxed);
    double total_tokens =
        (double)trainer->vocab->retained_token_count *
        trainer->config.epochs;
    real learning_rate = trainer->config.initial_learning_rate *
                         (real)(1.0 -
                                (double)processed_tokens /
                                    (total_tokens + 1.0));
    real minimum = trainer->config.initial_learning_rate * 0.0001f;

    worker->learning_rate =
        learning_rate < minimum ? minimum : learning_rate;
    worker->last_learning_rate_update_count = worker->local_token_count;
}

static void train_sentence(Trainer *trainer, Worker *worker)
{
    while (worker->sentence_position < worker->sentence_length)
    {
        worker_update_learning_rate(worker, trainer);
        ModelStep step = {
            .target_token = worker->sentence[worker->sentence_position],
            .learning_rate = worker->learning_rate,
            .trainer = trainer,
            .worker = worker,
        };

        if (trainer->config.model_kind == MODEL_CBOW)
        {
            cbow_train(&step);
        }
        else
        {
            skip_gram_train(&step);
        }
        worker->sentence_position++;
    }
}

void *worker_run(void *opaque_arguments)
{
    ThreadArguments *arguments = opaque_arguments;
    Trainer *trainer = arguments->trainer;
    Worker worker;

    arguments->status = worker_initialize(
        &worker,
        trainer,
        arguments->worker_id);
    if (arguments->status != STATUS_OK)
    {
        return NULL;
    }

    for (size_t epoch = 0;
         epoch < trainer->config.epochs && arguments->status == STATUS_OK;
         epoch++)
    {
        worker.epoch_token_count = 0;
        if (epoch > 0)
        {
            clearerr(worker.file);
            arguments->status = worker_seek_to_shard(
                worker.file,
                worker.shard_start);
        }

        int finished = 0;
        while (arguments->status == STATUS_OK && !finished)
        {
            arguments->status = worker_fill_sentence(
                &worker,
                trainer,
                &finished);
            uint64_t worker_token_limit =
                trainer->vocab->retained_token_count /
                trainer->config.thread_count;
            if (worker.epoch_token_count > worker_token_limit)
            {
                finished = 1;
            }
            if (arguments->status == STATUS_OK)
            {
                if (!finished)
                {
                    train_sentence(trainer, &worker);
                }
            }
        }
    }

    worker_free(&worker);
    return NULL;
}

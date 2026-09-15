#include "negative_sampler.h"
#include "sigmoid_table.h"
#include "training.h"

#include <pthread.h>
#include <stdlib.h>

static void report_status(Status *status, Status value)
{
    if (status != NULL)
    {
        *status = value;
    }
}

static int arguments_are_valid(
    const Corpus *corpus,
    const Vocabulary *vocab,
    const Model *model,
    const TrainingConfig *config)
{
    if (corpus == NULL || corpus->path == NULL || vocab == NULL ||
        model == NULL || training_config_validate(config) != STATUS_OK)
    {
        return 0;
    }
    if (model->vocab_size != vocab->size ||
        model->embedding_dimension != config->embedding_dimension)
    {
        return 0;
    }
    if (config->objective_kind == OBJECTIVE_NEGATIVE_SAMPLING &&
        vocab->size < 2)
    {
        return 0;
    }
    return 1;
}

Trainer *trainer_create(
    const Corpus *corpus,
    const Vocabulary *vocab,
    Model *model,
    const TrainingConfig *config,
    Status *status)
{
    report_status(status, STATUS_INVALID_ARGUMENT);
    if (!arguments_are_valid(corpus, vocab, model, config))
    {
        return NULL;
    }

    Trainer *trainer = calloc(1, sizeof(*trainer));
    if (trainer == NULL)
    {
        report_status(status, STATUS_OUT_OF_MEMORY);
        return NULL;
    }

    trainer->corpus = corpus;
    trainer->vocab = vocab;
    trainer->model = model;
    trainer->config = *config;
    atomic_init(&trainer->processed_tokens, 0);

    Status result = sigmoid_table_initialize(
        &trainer->sigmoid_table,
        config->sigmoid_table_size,
        config->sigmoid_max);
    if (result == STATUS_OK &&
        config->objective_kind == OBJECTIVE_NEGATIVE_SAMPLING)
    {
        result = negative_sampler_initialize(
            &trainer->negative_sampler,
            vocab,
            config->negative_table_size);
    }
    if (result != STATUS_OK)
    {
        trainer_destroy(&trainer);
    }

    report_status(status, result);
    return trainer;
}

void trainer_destroy(Trainer **trainer)
{
    if (trainer == NULL || *trainer == NULL)
    {
        return;
    }

    negative_sampler_free(&(*trainer)->negative_sampler);
    sigmoid_table_free(&(*trainer)->sigmoid_table);
    free(*trainer);
    *trainer = NULL;
}

uint64_t trainer_processed_tokens(const Trainer *trainer)
{
    if (trainer == NULL)
    {
        return 0;
    }
    return atomic_load_explicit(
        &trainer->processed_tokens,
        memory_order_relaxed);
}

Status trainer_train(Trainer *trainer)
{
    if (trainer == NULL)
    {
        return STATUS_INVALID_ARGUMENT;
    }

    size_t thread_count = trainer->config.thread_count;
    if (thread_count > SIZE_MAX / sizeof(pthread_t) ||
        thread_count > SIZE_MAX / sizeof(ThreadArguments))
    {
        return STATUS_INVALID_ARGUMENT;
    }

    pthread_t *threads = calloc(thread_count, sizeof(*threads));
    ThreadArguments *arguments = calloc(
        thread_count,
        sizeof(*arguments));
    if (threads == NULL || arguments == NULL)
    {
        free(threads);
        free(arguments);
        return STATUS_OUT_OF_MEMORY;
    }

    atomic_store_explicit(
        &trainer->processed_tokens,
        0,
        memory_order_relaxed);

    size_t created_thread_count = 0;
    Status result = STATUS_OK;
    for (size_t thread_index = 0;
         thread_index < thread_count;
         thread_index++)
    {
        arguments[thread_index] = (ThreadArguments){
            .trainer = trainer,
            .worker_id = thread_index,
            .status = STATUS_OK,
        };
        int create_result = pthread_create(
            &threads[thread_index],
            NULL,
            worker_run,
            &arguments[thread_index]);
        if (create_result != 0)
        {
            result = STATUS_THREAD_ERROR;
            break;
        }
        created_thread_count++;
    }

    for (size_t thread_index = 0;
         thread_index < created_thread_count;
         thread_index++)
    {
        if (pthread_join(threads[thread_index], NULL) != 0)
        {
            result = STATUS_THREAD_ERROR;
        }
    }
    for (size_t thread_index = 0;
         thread_index < created_thread_count;
         thread_index++)
    {
        if (result == STATUS_OK &&
            arguments[thread_index].status != STATUS_OK)
        {
            result = arguments[thread_index].status;
        }
    }

    free(threads);
    free(arguments);
    return result;
}

#include "w2v/config.h"

#include <math.h>
#include <stdint.h>

void vocab_config_defaults(VocabularyConfig *config)
{
    if (config == NULL)
    {
        return;
    }

    *config = (VocabularyConfig){
        .initial_capacity = 1000,
        .hash_capacity = 3000001,
        .min_count = 5,
    };
}

void training_config_defaults(TrainingConfig *config)
{
    if (config == NULL)
    {
        return;
    }

    *config = (TrainingConfig){
        .model_kind = MODEL_CBOW,
        .objective_kind = OBJECTIVE_NEGATIVE_SAMPLING,
        .embedding_dimension = 100,
        .window_radius = 5,
        .epochs = 5,
        .thread_count = 12,
        .learning_rate_update_interval = 10000,
        .initial_learning_rate = 0.05f,
        .subsampling_threshold = 1e-3f,
        .negative_sample_count = 5,
        .root_seed = 1,
        .rng_algorithm = RNG_LCG,
        .negative_table_size = 1000000,
        .sigmoid_table_size = 1000,
        .sigmoid_max = 6.0f,
        .hs_out_of_range_policy = HS_OUT_OF_RANGE_SKIP,
    };
}

Status vocab_config_validate(const VocabularyConfig *config)
{
    if (config == NULL || config->initial_capacity == 0 ||
        config->hash_capacity < 2 || config->min_count == 0)
    {
        return STATUS_INVALID_ARGUMENT;
    }

    if (config->initial_capacity > SIZE_MAX / sizeof(void *) ||
        config->hash_capacity > SIZE_MAX / sizeof(size_t))
    {
        return STATUS_INVALID_ARGUMENT;
    }

    return STATUS_OK;
}

static int model_kind_is_valid(ModelKind model_kind)
{
    return model_kind == MODEL_CBOW || model_kind == MODEL_SKIP_GRAM;
}

static int objective_kind_is_valid(ObjectiveKind objective_kind)
{
    return objective_kind == OBJECTIVE_HIERARCHICAL_SOFTMAX ||
           objective_kind == OBJECTIVE_NEGATIVE_SAMPLING;
}

static int hs_policy_is_valid(HsOutOfRangePolicy policy)
{
    return policy == HS_OUT_OF_RANGE_SKIP ||
           policy == HS_OUT_OF_RANGE_USE_BOUNDARY_VALUE;
}

static int rng_algorithm_is_valid(RngAlgorithm algorithm)
{
    return algorithm == RNG_LCG || algorithm == RNG_XORSHIFT;
}

Status training_config_validate(const TrainingConfig *config)
{
    if (config == NULL || !model_kind_is_valid(config->model_kind) ||
        !objective_kind_is_valid(config->objective_kind) ||
        !hs_policy_is_valid(config->hs_out_of_range_policy) ||
        !rng_algorithm_is_valid(config->rng_algorithm))
    {
        return STATUS_INVALID_ARGUMENT;
    }

    if (config->embedding_dimension == 0 || config->window_radius == 0 ||
        config->window_radius > SIZE_MAX / 2 ||
        config->epochs == 0 || config->thread_count == 0 ||
        config->learning_rate_update_interval == 0)
    {
        return STATUS_INVALID_ARGUMENT;
    }

    if (!isfinite(config->initial_learning_rate) ||
        config->initial_learning_rate <= 0 ||
        !isfinite(config->subsampling_threshold) ||
        config->subsampling_threshold < 0 ||
        !isfinite(config->sigmoid_max) || config->sigmoid_max <= 0)
    {
        return STATUS_INVALID_ARGUMENT;
    }

    if (config->sigmoid_table_size < 2 ||
        config->embedding_dimension > SIZE_MAX / sizeof(real))
    {
        return STATUS_INVALID_ARGUMENT;
    }

    if (config->objective_kind == OBJECTIVE_NEGATIVE_SAMPLING &&
        (config->negative_sample_count == 0 ||
         config->negative_table_size == 0))
    {
        return STATUS_INVALID_ARGUMENT;
    }

    return STATUS_OK;
}

const char *status_string(Status status)
{
    static const char *names[] = {
        "ok",
        "invalid argument",
        "out of memory",
        "I/O error",
        "corrupt data",
        "thread error",
    };

    if (status < STATUS_OK || status > STATUS_THREAD_ERROR)
    {
        return "unknown status";
    }

    return names[status];
}

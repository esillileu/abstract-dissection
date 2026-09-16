#ifndef W2V_CONFIG_H
#define W2V_CONFIG_H

#include <stddef.h>
#include <stdint.h>

#define MAX_TOKEN_LENGTH 100
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

typedef float real;

typedef enum
{
    STATUS_OK = 0,
    STATUS_INVALID_ARGUMENT,
    STATUS_OUT_OF_MEMORY,
    STATUS_IO_ERROR,
    STATUS_CORRUPT_DATA,
    STATUS_THREAD_ERROR
} Status;

typedef enum
{
    MODEL_CBOW = 0,
    MODEL_SKIP_GRAM = 1
} ModelKind;

typedef enum
{
    OBJECTIVE_HIERARCHICAL_SOFTMAX = 0,
    OBJECTIVE_NEGATIVE_SAMPLING = 1
} ObjectiveKind;

typedef enum
{
    HS_OUT_OF_RANGE_SKIP = 0,
    HS_OUT_OF_RANGE_USE_BOUNDARY_VALUE = 1
} HsOutOfRangePolicy;

typedef enum
{
    RNG_LCG = 0,
    RNG_XORSHIFT = 1
} RngAlgorithm;

typedef struct
{
    size_t initial_capacity;
    size_t hash_capacity;
    uint64_t min_count;
} VocabularyConfig;

typedef struct
{
    ModelKind model_kind;
    ObjectiveKind objective_kind;
    size_t embedding_dimension;
    size_t window_radius;
    size_t epochs;
    size_t thread_count;
    size_t learning_rate_update_interval;
    real initial_learning_rate;
    real subsampling_threshold;
    size_t negative_sample_count;
    uint64_t root_seed;
    RngAlgorithm rng_algorithm;
    size_t negative_table_size;
    size_t sigmoid_table_size;
    real sigmoid_max;
    HsOutOfRangePolicy hs_out_of_range_policy;
} TrainingConfig;

void vocab_config_defaults(VocabularyConfig *config);
void training_config_defaults(TrainingConfig *config);
void training_config_defaults_for_model(TrainingConfig *config, ModelKind model_kind);
Status vocab_config_validate(const VocabularyConfig *config);
Status training_config_validate(const TrainingConfig *config);
const char *status_string(Status status);

#endif

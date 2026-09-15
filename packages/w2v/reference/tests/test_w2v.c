#include "atomic_float.h"
#include "negative_sampler.h"
#include "rng.h"
#include "sigmoid_table.h"
#include "tokenizer.h"
#include "training.h"
#include "w2v/w2v.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const char *fixture_path = "tests/corpus.tmp";

static void write_fixture(void)
{
    FILE *file = fopen(fixture_path, "wb");

    assert(file != NULL);
    assert(fputs(
               "alpha beta alpha gamma\n"
               "beta alpha delta\n"
               "gamma beta alpha\n",
               file) >= 0);
    assert(fclose(file) == 0);
}

static uint64_t hash_float_bits(const real *values, size_t count)
{
    uint64_t hash = UINT64_C(1469598103934665603);

    for (size_t index = 0; index < count; index++)
    {
        uint32_t bits;

        memcpy(&bits, &values[index], sizeof(bits));
        for (size_t byte_index = 0; byte_index < sizeof(bits); byte_index++)
        {
            unsigned int byte = (bits >> (byte_index * 8)) & 0xffU;

            hash ^= byte;
            hash *= UINT64_C(1099511628211);
        }
    }
    return hash;
}

static void test_config_and_rng(void)
{
    VocabularyConfig vocab_config;
    TrainingConfig training_config;

    vocab_config_defaults(&vocab_config);
    training_config_defaults(&training_config);
    assert(vocab_config_validate(&vocab_config) == STATUS_OK);
    assert(training_config_validate(&training_config) == STATUS_OK);
    assert(training_config.rng_algorithm == RNG_LCG);
    assert(vocab_config.hash_capacity == 30000000);
    assert(MAX_CODE_LENGTH == 40);
    assert(training_config.initial_learning_rate == 0.05f);
    assert(training_config.negative_table_size == 100000000);

    training_config_defaults_for_model(&training_config, MODEL_SKIP_GRAM);
    assert(training_config.model_kind == MODEL_SKIP_GRAM);
    assert(training_config.initial_learning_rate == 0.025f);
    assert(training_config_validate(&training_config) == STATUS_OK);
    training_config_defaults_for_model(&training_config, MODEL_CBOW);
    assert(training_config.initial_learning_rate == 0.05f);

    training_config.rng_algorithm = RNG_XORSHIFT;
    assert(training_config_validate(&training_config) == STATUS_OK);
    training_config.rng_algorithm = (RngAlgorithm)99;
    assert(training_config_validate(&training_config) == STATUS_INVALID_ARGUMENT);
    training_config.rng_algorithm = RNG_LCG;

    training_config.window_radius = 0;
    assert(training_config_validate(&training_config) == STATUS_INVALID_ARGUMENT);
    training_config.window_radius = 1;
    training_config.model_kind = (ModelKind)99;
    assert(training_config_validate(&training_config) == STATUS_INVALID_ARGUMENT);
    training_config.model_kind = MODEL_CBOW;
    training_config.initial_learning_rate = NAN;
    assert(training_config_validate(&training_config) == STATUS_INVALID_ARGUMENT);
    training_config.initial_learning_rate = 0.05f;
    training_config.learning_rate_update_interval = 0;
    assert(training_config_validate(&training_config) == STATUS_INVALID_ARGUMENT);
    training_config.learning_rate_update_interval = 10000;
    training_config.hs_out_of_range_policy = (HsOutOfRangePolicy)99;
    assert(training_config_validate(&training_config) == STATUS_INVALID_ARGUMENT);

    Rng first_rng;
    Rng matching_rng;
    Rng different_rng;

    rng_init(&first_rng, derive_seed(7, 2, RNG_WINDOW), RNG_LCG);
    rng_init(&matching_rng, derive_seed(7, 2, RNG_WINDOW), RNG_LCG);
    rng_init(&different_rng, derive_seed(7, 2, RNG_NEGATIVE), RNG_LCG);
    assert(rng_next(&first_rng) == rng_next(&matching_rng));
    assert(rng_next(&matching_rng) != rng_next(&different_rng));

    static const uint64_t expected_lcg[] = {
        UINT64_C(25214903928),
        UINT64_C(8602081314781131043),
        UINT64_C(4749291277619109362),
        UINT64_C(15888805192744905749),
    };
    static const uint64_t expected_xorshift[] = {
        UINT64_C(5180492295206395165),
        UINT64_C(12380297144915551517),
        UINT64_C(13389498078930870103),
        UINT64_C(5599127315341312413),
    };
    Rng sample_rng;

    rng_init(&sample_rng, 1, RNG_LCG);
    for (size_t index = 0; index < 4; index++)
    {
        assert(rng_next(&sample_rng) == expected_lcg[index]);
    }
    rng_init(&sample_rng, 1, RNG_XORSHIFT);
    for (size_t index = 0; index < 4; index++)
    {
        assert(rng_next(&sample_rng) == expected_xorshift[index]);
    }
}

static void test_tokenizer_boundaries(void)
{
    FILE *file = tmpfile();

    assert(file != NULL);
    for (size_t index = 0; index < MAX_TOKEN_LENGTH + 20; index++)
    {
        assert(fputc('x', file) != EOF);
    }
    assert(fputs("  short\r\n\tlast", file) >= 0);
    rewind(file);

    char token[MAX_TOKEN_LENGTH];
    int at_eof = 0;

    assert(token_read(file, token, &at_eof) == STATUS_OK);
    assert(strlen(token) == MAX_TOKEN_LENGTH - 1);
    assert(token_read(file, token, &at_eof) == STATUS_OK);
    assert(strcmp(token, "short") == 0);
    assert(token_read(file, token, &at_eof) == STATUS_OK);
    assert(strcmp(token, "</s>") == 0);
    assert(token_read(file, token, &at_eof) == STATUS_OK);
    assert(strcmp(token, "last") == 0);
    assert(at_eof == 1);
    assert(fclose(file) == 0);
}

static void test_context_order(void)
{
    Worker worker = {0};
    worker.sentence_length = 5;
    worker.sentence_position = 2;

    static const size_t expected_positions[] = {0, 1, 3, 4};
    size_t observed_count = 0;
    for (size_t offset = 0; offset <= 4; offset++)
    {
        size_t position;
        if (context_position(&worker, 2, offset, &position))
        {
            assert(position == expected_positions[observed_count]);
            observed_count++;
        }
    }
    assert(observed_count == 4);
}

/* Fixed one-step oracle: the expected values follow the upstream CBOW and
 * skip-gram equations with a constant sigmoid value. In negative sampling,
 * a duplicate negative is consumed but intentionally makes no update. */
static void test_fixed_one_step(
    const Corpus *corpus,
    const Vocabulary *vocab,
    ModelKind kind,
    ObjectiveKind objective)
{
    TrainingConfig config;
    training_config_defaults_for_model(&config, kind);
    config.embedding_dimension = 2;
    config.objective_kind = objective;
    config.window_radius = 1;
    config.negative_sample_count = 1;
    config.negative_table_size = 7;
    config.sigmoid_table_size = 3;
    config.subsampling_threshold = 0;
    config.initial_learning_rate = 0.05f;

    Status status;
    Model *model = model_create(vocab, 2, 1, RNG_LCG, &status);
    assert(model != NULL && status == STATUS_OK);
    Trainer *trainer = trainer_create(corpus, vocab, model, &config, &status);
    assert(trainer != NULL && status == STATUS_OK);
    for (size_t index = 0; index < trainer->sigmoid_table.size; index++)
    {
        trainer->sigmoid_table.values[index] = 0.5f;
    }
    for (size_t index = 0; index < trainer->negative_sampler.size; index++)
    {
        trainer->negative_sampler.table[index] = 2;
    }

    for (size_t index = 0; index < model->vocab_size * 2; index++)
    {
        atomic_float_store(&model->input_embeddings[index], 0);
        atomic_float_store(&model->output_embeddings[index], 0);
    }
    atomic_float_store(&model->input_embeddings[2], 0.25f);
    atomic_float_store(&model->input_embeddings[3], -0.5f);
    atomic_float_store(&model->input_embeddings[6], 0.75f);
    atomic_float_store(&model->input_embeddings[7], 0.5f);
    atomic_float_store(&model->output_embeddings[4], 0.2f);
    atomic_float_store(&model->output_embeddings[5], -0.4f);
    if (objective == OBJECTIVE_HIERARCHICAL_SOFTMAX)
    {
        atomic_float_store(&model->output_embeddings[6], 0.2f);
        atomic_float_store(&model->output_embeddings[7], -0.4f);
        atomic_float_store(&model->output_embeddings[4], 0);
        atomic_float_store(&model->output_embeddings[5], 0);
    }

    Worker worker = {0};
    worker.sentence[0] = 1;
    worker.sentence[1] = 2;
    worker.sentence[2] = 3;
    worker.sentence_length = 3;
    worker.sentence_position = 1;
    rng_init(&worker.window_rng, 1, RNG_LCG);
    rng_init(&worker.negative_rng, 1, RNG_LCG);
    worker.hidden = calloc(2, sizeof(real));
    worker.hidden_gradient = calloc(2, sizeof(real));
    assert(worker.hidden != NULL && worker.hidden_gradient != NULL);
    ModelStep step = {
        .target_token = 2,
        .learning_rate = 0.05f,
        .trainer = trainer,
        .worker = &worker,
    };

    if (kind == MODEL_CBOW)
    {
        cbow_train(&step);
        assert(atomic_float_load(&model->input_embeddings[2]) == 0.255f);
        assert(atomic_float_load(&model->input_embeddings[3]) == -0.51f);
        assert(atomic_float_load(&model->input_embeddings[6]) == 0.755f);
        assert(atomic_float_load(&model->input_embeddings[7]) == 0.49f);
        size_t first = objective == OBJECTIVE_HIERARCHICAL_SOFTMAX ? 6 : 4;
        assert(atomic_float_load(&model->output_embeddings[first]) == 0.2125f);
        assert(atomic_float_load(&model->output_embeddings[first + 1]) == -0.4f);
        if (objective == OBJECTIVE_HIERARCHICAL_SOFTMAX)
        {
            assert(atomic_float_load(&model->output_embeddings[2]) == -0.0125f);
        }
    }
    else
    {
        skip_gram_train(&step);
        assert(atomic_float_load(&model->input_embeddings[2]) == 0.255f);
        assert(atomic_float_load(&model->input_embeddings[3]) == -0.51f);
        if (objective == OBJECTIVE_HIERARCHICAL_SOFTMAX)
        {
            assert(atomic_float_load(&model->input_embeddings[6]) == 0.7553125f);
            assert(atomic_float_load(&model->input_embeddings[7]) == 0.489375f);
            assert(atomic_float_load(&model->output_embeddings[6]) == 0.225f);
            assert(atomic_float_load(&model->output_embeddings[7]) == -0.4f);
            assert(atomic_float_load(&model->output_embeddings[2]) == -0.025f);
            assert(atomic_float_load(&model->output_embeddings[3]) == 0);
        }
        else
        {
            assert(atomic_float_load(&model->input_embeddings[6]) == 0.75515625f);
            assert(atomic_float_load(&model->input_embeddings[7]) == 0.4896875f);
            assert(atomic_float_load(&model->output_embeddings[4]) == 0.225f);
            assert(atomic_float_load(&model->output_embeddings[5]) == -0.4f);
        }
    }
    assert(worker.window_rng.state == UINT64_C(25214903928));
    if (objective == OBJECTIVE_NEGATIVE_SAMPLING)
    {
        assert(worker.negative_rng.state != 1);
    }
    else
    {
        assert(worker.negative_rng.state == 1);
    }
    free(worker.hidden);
    free(worker.hidden_gradient);
    trainer_destroy(&trainer);
    model_destroy(&model);
}

static void test_prune_at_seventy_percent(void)
{
    const char *path = "tests/prune.tmp";
    FILE *file = fopen(path, "wb");
    assert(file != NULL);
    assert(fputs("a b c d e f g h\n", file) >= 0);
    assert(fclose(file) == 0);

    Status status;
    Corpus *corpus = corpus_create(path, &status);
    assert(corpus != NULL && status == STATUS_OK);
    VocabularyConfig config;
    vocab_config_defaults(&config);
    config.initial_capacity = 2;
    config.hash_capacity = 10;
    config.min_count = 1;
    Vocabulary *vocab = vocab_build(corpus, &config, &status);
    assert(vocab != NULL && status == STATUS_OK);
    assert(vocab->size == 2);
    assert(strcmp(vocab->entries[0].token, "</s>") == 0);
    assert(strcmp(vocab->entries[1].token, "h") == 0);
    vocab_destroy(&vocab);
    corpus_destroy(&corpus);
    assert(remove(path) == 0);
}

static void test_duplicate_negative_is_not_redrawn(
    const Corpus *corpus,
    const Vocabulary *vocab)
{
    TrainingConfig config;
    training_config_defaults(&config);
    config.embedding_dimension = 2;
    config.negative_sample_count = 3;
    config.negative_table_size = 7;

    Status status;
    Model *model = model_create(
        vocab, 2, config.root_seed, config.rng_algorithm, &status);
    assert(model != NULL && status == STATUS_OK);
    Trainer *trainer = trainer_create(corpus, vocab, model, &config, &status);
    assert(trainer != NULL && status == STATUS_OK);

    size_t positive_token = 1;
    for (size_t index = 0; index < trainer->negative_sampler.size; index++)
    {
        trainer->negative_sampler.table[index] = positive_token;
    }

    Worker worker = {0};
    rng_init(&worker.negative_rng, 1, RNG_LCG);
    ModelStep step = {
        .target_token = positive_token,
        .learning_rate = config.initial_learning_rate,
        .trainer = trainer,
        .worker = &worker,
    };
    real hidden[] = {0.25f, -0.5f};
    real gradient[] = {0, 0};

    negative_sampling_train(&step, hidden, gradient);
    assert(worker.negative_rng.state != 1);
    assert(atomic_float_load(
               &model->output_embeddings[positive_token * 2]) != 0);
    for (size_t index = 0; index < model->vocab_size; index++)
    {
        if (index != positive_token)
        {
            assert(atomic_float_load(
                       &model->output_embeddings[index * 2]) == 0);
        }
    }

    trainer_destroy(&trainer);
    model_destroy(&model);
}

static void test_learning_rate_interval(
    const Corpus *corpus,
    const Vocabulary *vocab)
{
    TrainingConfig config;
    training_config_defaults(&config);
    config.embedding_dimension = 2;
    config.learning_rate_update_interval = 3;
    config.negative_table_size = 7;

    Status status;
    Model *model = model_create(
        vocab, 2, config.root_seed, config.rng_algorithm, &status);
    assert(model != NULL && status == STATUS_OK);
    Trainer *trainer = trainer_create(corpus, vocab, model, &config, &status);
    assert(trainer != NULL && status == STATUS_OK);

    Worker worker = {0};
    worker.learning_rate = config.initial_learning_rate;
    worker.local_token_count = 2;
    atomic_store_explicit(&trainer->processed_tokens, 2, memory_order_relaxed);
    worker_update_learning_rate(&worker, trainer);
    assert(worker.learning_rate == config.initial_learning_rate);

    worker.local_token_count = 3;
    atomic_store_explicit(&trainer->processed_tokens, 3, memory_order_relaxed);
    worker_update_learning_rate(&worker, trainer);
    assert(worker.learning_rate == config.initial_learning_rate);
    assert(worker.last_learning_rate_update_count == 0);

    worker.local_token_count = 4;
    atomic_store_explicit(&trainer->processed_tokens, 4, memory_order_relaxed);
    worker_update_learning_rate(&worker, trainer);
    real updated_rate = worker.learning_rate;
    assert(updated_rate < config.initial_learning_rate);
    assert(worker.last_learning_rate_update_count == 4);

    worker.local_token_count = 7;
    atomic_store_explicit(&trainer->processed_tokens, 7, memory_order_relaxed);
    worker_update_learning_rate(&worker, trainer);
    assert(worker.learning_rate == updated_rate);

    worker.local_token_count = 8;
    atomic_store_explicit(&trainer->processed_tokens, 8, memory_order_relaxed);
    worker_update_learning_rate(&worker, trainer);
    assert(worker.learning_rate < updated_rate);

    trainer_destroy(&trainer);
    model_destroy(&model);
}

static void test_hs_boundary_policy(
    const Corpus *corpus,
    const Vocabulary *vocab,
    HsOutOfRangePolicy policy)
{
    TrainingConfig config;
    training_config_defaults(&config);
    config.objective_kind = OBJECTIVE_HIERARCHICAL_SOFTMAX;
    config.embedding_dimension = 2;
    config.hs_out_of_range_policy = policy;

    Status status;
    Model *model = model_create(
        vocab, 2, config.root_seed, config.rng_algorithm, &status);
    assert(model != NULL && status == STATUS_OK);
    Trainer *trainer = trainer_create(corpus, vocab, model, &config, &status);
    assert(trainer != NULL && status == STATUS_OK);

    size_t target_token = 1;
    const VocabularyEntry *entry = &vocab->entries[target_token];
    real sign = entry->huffman_bits[0] == 0 ? -1.0f : 1.0f;
    real boundary_value = sign * config.sigmoid_max;
    size_t output_coordinate = entry->huffman_path[0] * 2;
    atomic_float_store(
        &model->output_embeddings[output_coordinate],
        boundary_value);

    Worker worker = {0};
    ModelStep step = {
        .target_token = target_token,
        .learning_rate = config.initial_learning_rate,
        .trainer = trainer,
        .worker = &worker,
    };
    real hidden[] = {1.0f, 0};
    real gradient[] = {0, 0};

    hierarchical_softmax_train(&step, hidden, gradient);
    real updated_value = atomic_float_load(
        &model->output_embeddings[output_coordinate]);
    if (policy == HS_OUT_OF_RANGE_SKIP)
    {
        assert(updated_value == boundary_value);
    }
    else
    {
        assert(updated_value != boundary_value);
    }

    trainer_destroy(&trainer);
    model_destroy(&model);
}

static void test_vocab(const Vocabulary *vocab)
{
    static const char *expected_tokens[] = {
        "</s>", "alpha", "beta", "gamma", "delta"};
    static const uint64_t expected_counts[] = {3, 4, 3, 2, 1};
    static const size_t expected_lengths[] = {2, 2, 2, 3, 3};
    static const size_t expected_paths[][3] = {
        {3, 2, 0}, {3, 2, 0}, {3, 1, 0},
        {3, 1, 0}, {3, 1, 0}};
    static const unsigned char expected_bits[][3] = {
        {1, 1, 0}, {1, 0, 0}, {0, 1, 0},
        {0, 0, 1}, {0, 0, 0}};

    assert(vocab->size == 5);
    for (size_t index = 0; index < 5; index++)
    {
        const char *token = vocab->entries[index].token;

        assert(strcmp(token, expected_tokens[index]) == 0);
        assert(vocab->entries[index].count == expected_counts[index]);
        assert(vocab->entries[index].huffman_length == expected_lengths[index]);
        for (size_t path = 0; path < expected_lengths[index]; path++)
        {
            assert(vocab->entries[index].huffman_path[path] ==
                   expected_paths[index][path]);
            assert(vocab->entries[index].huffman_bits[path] ==
                   expected_bits[index][path]);
        }
    }

    NegativeSampler sampler = {0};
    size_t alpha_hits = 0;

    assert(negative_sampler_initialize(&sampler, vocab, 1000) == STATUS_OK);
    for (size_t index = 0; index < sampler.size; index++)
    {
        if (sampler.table[index] == 1)
        {
            alpha_hits++;
        }
    }
    assert(alpha_hits == 281);
    negative_sampler_free(&sampler);

    SigmoidTable sigmoid = {0};

    assert(sigmoid_table_initialize(&sigmoid, 101, 6.0f) == STATUS_OK);
    assert(sigmoid_table_lookup(&sigmoid, -6.0f) == 0);
    assert(sigmoid_table_lookup(&sigmoid, 6.0f) == 1);
    assert(sigmoid_table_lookup(&sigmoid, 0) == 0x1.b483bep-2f);
    sigmoid_table_free(&sigmoid);
}

typedef struct
{
    ModelKind model_kind;
    ObjectiveKind objective_kind;
    RngAlgorithm rng_algorithm;
    uint64_t input_hash;
    uint64_t output_hash;
} GoldenCase;

static void test_golden_case(
    const Corpus *corpus,
    const Vocabulary *vocab,
    const GoldenCase *golden)
{
    TrainingConfig config;

    training_config_defaults_for_model(&config, golden->model_kind);
    config.objective_kind = golden->objective_kind;
    config.rng_algorithm = golden->rng_algorithm;
    config.embedding_dimension = 8;
    config.window_radius = 2;
    config.epochs = 2;
    config.thread_count = 1;
    config.subsampling_threshold = 0;
    config.negative_sample_count = 2;
    config.negative_table_size = 257;
    config.sigmoid_table_size = 101;

    Status status;
    Model *model = model_create(
        vocab,
        config.embedding_dimension,
        config.root_seed,
        config.rng_algorithm,
        &status);

    assert(model != NULL && status == STATUS_OK);
    size_t element_count = model->vocab_size *
                           model->embedding_dimension;
    real *snapshot = malloc(element_count * sizeof(*snapshot));

    assert(snapshot != NULL);
    assert(model_snapshot(
               model,
               INPUT_EMBEDDING,
               snapshot,
               element_count) == STATUS_OK);
    uint64_t initial_hash = golden->rng_algorithm == RNG_LCG
                                ? UINT64_C(0x1481c956b48bc5f8)
                                : UINT64_C(0x2e41636ac144f2d5);
    assert(hash_float_bits(snapshot, element_count) == initial_hash);

    Trainer *trainer = trainer_create(
        corpus,
        vocab,
        model,
        &config,
        &status);

    assert(trainer != NULL && status == STATUS_OK);
    assert(trainer_train(trainer) == STATUS_OK);
    assert(trainer_processed_tokens(trainer) == 26);
    assert(model_snapshot(
               model,
               INPUT_EMBEDDING,
               snapshot,
               element_count) == STATUS_OK);
    assert(hash_float_bits(snapshot, element_count) == golden->input_hash);
    assert(model_snapshot(
               model,
               OUTPUT_EMBEDDING,
               snapshot,
               element_count) == STATUS_OK);
    assert(hash_float_bits(snapshot, element_count) == golden->output_hash);

    trainer_destroy(&trainer);
    trainer_destroy(&trainer);
    model_destroy(&model);
    model_destroy(&model);
    free(snapshot);
}

static void test_parallel_training(
    const Corpus *corpus,
    const Vocabulary *vocab)
{
    TrainingConfig config;

    training_config_defaults(&config);
    config.embedding_dimension = 16;
    config.window_radius = 2;
    config.epochs = 8;
    config.thread_count = 4;
    config.subsampling_threshold = 0;
    config.negative_sample_count = 4;
    config.negative_table_size = 257;
    config.sigmoid_table_size = 101;

    Status status;
    Model *model = model_create(
        vocab,
        config.embedding_dimension,
        config.root_seed,
        config.rng_algorithm,
        &status);
    Trainer *trainer = trainer_create(
        corpus,
        vocab,
        model,
        &config,
        &status);

    assert(model != NULL && trainer != NULL && status == STATUS_OK);
    assert(trainer_train(trainer) == STATUS_OK);
    assert(trainer_processed_tokens(trainer) > 0);

    size_t element_count = model->vocab_size *
                           model->embedding_dimension;
    real *snapshot = malloc(element_count * sizeof(*snapshot));

    assert(snapshot != NULL);
    assert(model_snapshot(
               model,
               INPUT_EMBEDDING,
               snapshot,
               element_count) == STATUS_OK);
    for (size_t index = 0; index < element_count; index++)
    {
        assert(isfinite(snapshot[index]));
    }

    free(snapshot);
    trainer_destroy(&trainer);
    model_destroy(&model);
}

static void test_error_paths(const Vocabulary *vocab)
{
    Status status = STATUS_OK;
    Corpus *missing = corpus_create(
        "tests/does-not-exist",
        &status);

    assert(missing == NULL && status == STATUS_IO_ERROR);
    corpus_destroy(&missing);

    Model *model = model_create(vocab, SIZE_MAX, 1, RNG_LCG, &status);

    assert(model == NULL && status == STATUS_INVALID_ARGUMENT);
}

int main(void)
{
    static const GoldenCase golden_cases[] = {
        {MODEL_CBOW, OBJECTIVE_HIERARCHICAL_SOFTMAX, RNG_LCG,
         UINT64_C(0x514f61880ff65c1f), UINT64_C(0xef981cde17cb9eb2)},
        {MODEL_CBOW, OBJECTIVE_NEGATIVE_SAMPLING, RNG_LCG,
         UINT64_C(0x998a6cbb98dc7f6c), UINT64_C(0x7b15e9239642b63f)},
        {MODEL_SKIP_GRAM, OBJECTIVE_HIERARCHICAL_SOFTMAX, RNG_LCG,
         UINT64_C(0xdf8d0e477392fa07), UINT64_C(0x9a0b52d6d6b31bd7)},
        {MODEL_SKIP_GRAM, OBJECTIVE_NEGATIVE_SAMPLING, RNG_LCG,
         UINT64_C(0xd9c808eed06c7053), UINT64_C(0xfa8c51e6c86f0efc)},
        {MODEL_CBOW, OBJECTIVE_NEGATIVE_SAMPLING, RNG_XORSHIFT,
         UINT64_C(0xcbe69a7cc0673513), UINT64_C(0x722439a66b1227d7)},
    };

    write_fixture();
    test_config_and_rng();
    test_tokenizer_boundaries();
    test_context_order();
    test_prune_at_seventy_percent();

    Status status;
    Corpus *corpus = corpus_create(fixture_path, &status);

    assert(corpus != NULL && status == STATUS_OK);
    assert(corpus->byte_size > 0);

    VocabularyConfig config;

    vocab_config_defaults(&config);
    config.initial_capacity = 2;
    config.hash_capacity = 17;
    config.min_count = 1;
    Vocabulary *vocab = vocab_build(
        corpus,
        &config,
        &status);

    assert(vocab != NULL && status == STATUS_OK);
    test_vocab(vocab);
    test_fixed_one_step(corpus, vocab, MODEL_CBOW, OBJECTIVE_NEGATIVE_SAMPLING);
    test_fixed_one_step(corpus, vocab, MODEL_SKIP_GRAM, OBJECTIVE_NEGATIVE_SAMPLING);
    test_fixed_one_step(corpus, vocab, MODEL_CBOW, OBJECTIVE_HIERARCHICAL_SOFTMAX);
    test_fixed_one_step(corpus, vocab, MODEL_SKIP_GRAM, OBJECTIVE_HIERARCHICAL_SOFTMAX);
    test_duplicate_negative_is_not_redrawn(corpus, vocab);
    test_learning_rate_interval(corpus, vocab);
    test_hs_boundary_policy(corpus, vocab, HS_OUT_OF_RANGE_SKIP);
    test_hs_boundary_policy(
        corpus,
        vocab,
        HS_OUT_OF_RANGE_USE_BOUNDARY_VALUE);
    for (size_t index = 0;
         index < sizeof(golden_cases) / sizeof(golden_cases[0]);
         index++)
    {
        test_golden_case(corpus, vocab, &golden_cases[index]);
    }
    test_parallel_training(corpus, vocab);
    test_error_paths(vocab);

    vocab_destroy(&vocab);
    vocab_destroy(&vocab);
    corpus_destroy(&corpus);
    corpus_destroy(&corpus);
    assert(unlink(fixture_path) == 0);
    puts("w2v tests passed");
    return 0;
}

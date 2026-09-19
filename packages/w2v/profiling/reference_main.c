#include "w2v/w2v.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *program)
{
    fprintf(stderr, "usage: %s CORPUS cbow|skipgram hs|negative THREADS EPOCHS [NEGATIVE_TABLE_SIZE]\n", program);
}

static size_t parse_size(const char *value)
{
    char *end = NULL;
    errno = 0;
    unsigned long long parsed = strtoull(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed == 0 || parsed > SIZE_MAX)
    {
        return 0;
    }
    return (size_t)parsed;
}

int main(int argc, char **argv)
{
    if (argc < 6 || argc > 7)
    {
        usage(argv[0]);
        return 2;
    }
    ModelKind model_kind;
    if (strcmp(argv[2], "cbow") == 0)
    {
        model_kind = MODEL_CBOW;
    }
    else if (strcmp(argv[2], "skipgram") == 0)
    {
        model_kind = MODEL_SKIP_GRAM;
    }
    else
    {
        usage(argv[0]);
        return 2;
    }
    ObjectiveKind objective_kind;
    if (strcmp(argv[3], "hs") == 0)
    {
        objective_kind = OBJECTIVE_HIERARCHICAL_SOFTMAX;
    }
    else if (strcmp(argv[3], "negative") == 0)
    {
        objective_kind = OBJECTIVE_NEGATIVE_SAMPLING;
    }
    else
    {
        usage(argv[0]);
        return 2;
    }
    size_t threads = parse_size(argv[4]);
    size_t epochs = parse_size(argv[5]);
    size_t table_size = argc == 7 ? parse_size(argv[6]) : 100000000;
    if (threads == 0 || epochs == 0 || table_size == 0)
    {
        usage(argv[0]);
        return 2;
    }

    Status status = STATUS_OK;
    Corpus *corpus = corpus_create(argv[1], &status);
    VocabularyConfig vocab_config;
    vocab_config_defaults(&vocab_config);
    Vocabulary *vocab = corpus == NULL ? NULL : vocab_build(corpus, &vocab_config, &status);
    TrainingConfig config;
    training_config_defaults_for_model(&config, model_kind);
    config.objective_kind = objective_kind;
    config.embedding_dimension = 100;
    config.window_radius = 5;
    config.epochs = epochs;
    config.thread_count = threads;
    config.subsampling_threshold = 0.0f;
    config.negative_sample_count = 5;
    config.root_seed = 1;
    config.rng_algorithm = RNG_LCG;
    config.negative_table_size = table_size;
    Model *model = vocab == NULL ? NULL : model_create(vocab, config.embedding_dimension, config.root_seed, config.rng_algorithm, &status);
    Trainer *trainer = model == NULL ? NULL : trainer_create(corpus, vocab, model, &config, &status);
    if (trainer != NULL)
    {
        status = trainer_train(trainer);
    }
    if (status == STATUS_OK)
    {
        printf("processed_tokens=%llu vocab_size=%zu\n", (unsigned long long)trainer_processed_tokens(trainer), vocab->size);
    }
    else
    {
        fprintf(stderr, "reference runner: %s\n", status_string(status));
    }
    trainer_destroy(&trainer);
    model_destroy(&model);
    vocab_destroy(&vocab);
    corpus_destroy(&corpus);
    return status == STATUS_OK ? 0 : 1;
}

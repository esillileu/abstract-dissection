#ifndef W2V_TRAINER_H
#define W2V_TRAINER_H

#include "corpus.h"
#include "model.h"
#include "vocab.h"

#include <stdatomic.h>

typedef struct
{
    const Corpus *corpus;
    const Vocabulary *vocab;
    Model *model;
    NegativeSampler negative_sampler;
    SigmoidTable sigmoid_table;
    TrainingConfig config;
    _Atomic uint64_t processed_tokens;
} Trainer;

Trainer *trainer_create(
    const Corpus *corpus,
    const Vocabulary *vocab,
    Model *model,
    const TrainingConfig *config,
    Status *status);
void trainer_destroy(Trainer **trainer);
Status trainer_train(Trainer *trainer);
uint64_t trainer_processed_tokens(const Trainer *trainer);

#endif

#ifndef W2V_INTERNAL_NEGATIVE_SAMPLER_H
#define W2V_INTERNAL_NEGATIVE_SAMPLER_H

#include "rng.h"
#include "w2v/vocab.h"

Status negative_sampler_initialize(
    NegativeSampler *sampler,
    const Vocabulary *vocab,
    size_t table_size);
void negative_sampler_free(NegativeSampler *sampler);
size_t negative_sampler_draw(
    const NegativeSampler *sampler,
    Rng *rng,
    uint64_t *random_value);

#endif

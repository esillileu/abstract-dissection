#ifndef W2V_INTERNAL_RNG_H
#define W2V_INTERNAL_RNG_H

#include "w2v/config.h"

typedef struct
{
    uint64_t state;
    RngAlgorithm algorithm;
} Rng;

typedef enum
{
    RNG_MODEL = 1,
    RNG_WINDOW = 2,
    RNG_SUBSAMPLE = 3,
    RNG_NEGATIVE = 4
} RngPurpose;

uint64_t derive_seed(
    uint64_t root_seed,
    size_t worker_id,
    RngPurpose purpose);
void rng_init(Rng *rng, uint64_t seed, RngAlgorithm algorithm);
uint64_t rng_next(Rng *rng);
real rng_uniform(Rng *rng);

#endif

#include "rng.h"

static uint64_t mix_seed(uint64_t value)
{
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

uint64_t derive_seed(
    uint64_t root_seed,
    size_t worker_id,
    RngPurpose purpose)
{
    uint64_t worker_seed = mix_seed((uint64_t)worker_id + 1);
    uint64_t purpose_seed = mix_seed((uint64_t)purpose);

    return mix_seed(root_seed ^ worker_seed ^ purpose_seed);
}

void rng_init(Rng *rng, uint64_t seed, RngAlgorithm algorithm)
{
    rng->algorithm = algorithm;
    if (algorithm == RNG_XORSHIFT && seed == 0)
    {
        rng->state = UINT64_C(0x6a09e667f3bcc909);
        return;
    }

    rng->state = seed;
}

uint64_t rng_next(Rng *rng)
{
    if (rng->algorithm == RNG_LCG)
    {
        rng->state = rng->state * UINT64_C(25214903917) + 11;
        return rng->state;
    }

    uint64_t value = rng->state;

    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    rng->state = value;
    return value * UINT64_C(2685821657736338717);
}

real rng_uniform(Rng *rng)
{
    if (rng->algorithm == RNG_LCG)
    {
        uint64_t random_bits = rng_next(rng) & UINT64_C(0xffff);
        return (real)random_bits / 65536.0f;
    }

    uint64_t random_bits = rng_next(rng) >> 40;

    return (real)(random_bits * (1.0 / 16777216.0));
}

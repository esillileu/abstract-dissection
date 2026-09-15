#include "negative_sampler.h"

#include <math.h>
#include <stdlib.h>

Status negative_sampler_initialize(
    NegativeSampler *sampler,
    const Vocabulary *vocab,
    size_t table_size)
{
    if (sampler == NULL || vocab == NULL || vocab->size == 0 ||
        table_size == 0 || table_size > SIZE_MAX / sizeof(size_t))
    {
        return STATUS_INVALID_ARGUMENT;
    }

    *sampler = (NegativeSampler){0};
    sampler->table = malloc(table_size * sizeof(*sampler->table));
    if (sampler->table == NULL)
    {
        return STATUS_OUT_OF_MEMORY;
    }
    sampler->size = table_size;

    double total_weight = 0;
    for (size_t entry_index = 0;
         entry_index < vocab->size;
         entry_index++)
    {
        total_weight += pow(
            (double)vocab->entries[entry_index].count,
            0.75);
    }
    if (!(total_weight > 0))
    {
        negative_sampler_free(sampler);
        return STATUS_CORRUPT_DATA;
    }

    size_t entry_index = 0;
    double cumulative_probability =
        pow((double)vocab->entries[0].count, 0.75) / total_weight;
    for (size_t table_index = 0;
         table_index < table_size;
         table_index++)
    {
        double quantile = (double)table_index / (double)table_size;
        sampler->table[table_index] = entry_index;
        if (quantile > cumulative_probability &&
            entry_index + 1 < vocab->size)
        {
            entry_index++;
            cumulative_probability +=
                pow((double)vocab->entries[entry_index].count, 0.75) /
                total_weight;
        }
    }
    return STATUS_OK;
}

void negative_sampler_free(NegativeSampler *sampler)
{
    if (sampler == NULL)
    {
        return;
    }

    free(sampler->table);
    *sampler = (NegativeSampler){0};
}

size_t negative_sampler_draw(
    const NegativeSampler *sampler,
    Rng *rng,
    uint64_t *random_value)
{
    *random_value = rng_next(rng);
    size_t table_index = (*random_value >> 16) % sampler->size;

    return sampler->table[table_index];
}

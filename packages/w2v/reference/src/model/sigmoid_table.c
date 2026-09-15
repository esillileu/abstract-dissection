#include "sigmoid_table.h"

#include <math.h>
#include <stdlib.h>

Status sigmoid_table_initialize(
    SigmoidTable *table,
    size_t table_size,
    real maximum)
{
    if (table == NULL || table_size < 2 || !isfinite(maximum) ||
        maximum <= 0 || table_size > SIZE_MAX / sizeof(real))
    {
        return STATUS_INVALID_ARGUMENT;
    }

    *table = (SigmoidTable){0};
    table->values = malloc(table_size * sizeof(*table->values));
    if (table->values == NULL)
    {
        return STATUS_OUT_OF_MEMORY;
    }

    table->size = table_size;
    table->max = maximum;
    for (size_t table_index = 0; table_index < table_size; table_index++)
    {
        real scaled_index = (real)table_index / (real)table_size;
        real input = (scaled_index * 2 - 1) * maximum;
        real exponential = (real)exp((double)input);
        table->values[table_index] =
            exponential / (exponential + 1);
    }
    return STATUS_OK;
}

void sigmoid_table_free(SigmoidTable *table)
{
    if (table == NULL)
    {
        return;
    }

    free(table->values);
    *table = (SigmoidTable){0};
}

real sigmoid_table_lookup(const SigmoidTable *table, real value)
{
    if (value <= -table->max)
    {
        return 0;
    }
    if (value >= table->max)
    {
        return 1;
    }

    size_t index_scale = (size_t)(
        (real)table->size / table->max / 2);
    size_t table_index = (size_t)(
        (value + table->max) * (real)index_scale);
    if (table_index >= table->size)
    {
        table_index = table->size - 1;
    }
    return table->values[table_index];
}

#include "atomic_float.h"
#include "training.h"

real objective_score(
    const real *hidden,
    const _Atomic uint32_t *output_row,
    size_t dimension)
{
    real score = 0;

    for (size_t coordinate = 0; coordinate < dimension; coordinate++)
    {
        real output_value = atomic_float_load(&output_row[coordinate]);

        score += hidden[coordinate] * output_value;
    }
    return score;
}

void objective_apply_update(
    const real *hidden,
    real *hidden_gradient,
    _Atomic uint32_t *output_row,
    size_t dimension,
    real gradient_scale)
{
    for (size_t coordinate = 0; coordinate < dimension; coordinate++)
    {
        real output_value = atomic_float_load(&output_row[coordinate]);

        hidden_gradient[coordinate] += gradient_scale * output_value;
    }
    for (size_t coordinate = 0; coordinate < dimension; coordinate++)
    {
        real delta = gradient_scale * hidden[coordinate];

        atomic_float_add(&output_row[coordinate], delta);
    }
}

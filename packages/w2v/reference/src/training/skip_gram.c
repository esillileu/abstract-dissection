#include "atomic_float.h"
#include "training.h"

#include <string.h>

static void copy_input(
    const Model *model,
    size_t token,
    real *destination)
{
    size_t dimension = model->embedding_dimension;
    const _Atomic uint32_t *input_row =
        model->input_embeddings + token * dimension;

    for (size_t coordinate = 0; coordinate < dimension; coordinate++)
    {
        destination[coordinate] = atomic_float_load(&input_row[coordinate]);
    }
}

static void update_input(
    Model *model,
    size_t token,
    const real *gradient)
{
    size_t dimension = model->embedding_dimension;
    _Atomic uint32_t *input_row =
        model->input_embeddings + token * dimension;

    for (size_t coordinate = 0; coordinate < dimension; coordinate++)
    {
        atomic_float_add(&input_row[coordinate], gradient[coordinate]);
    }
}

void skip_gram_train(ModelStep *step)
{
    Model *model = step->trainer->model;
    Worker *worker = step->worker;
    size_t dimension = model->embedding_dimension;
    size_t radius = context_radius(step);

    for (size_t offset = 0; offset <= radius * 2; offset++)
    {
        size_t position;
        if (context_position(worker, radius, offset, &position))
        {
            size_t token = worker->sentence[position];

            copy_input(model, token, worker->hidden);
            memset(
                worker->hidden_gradient,
                0,
                dimension * sizeof(*worker->hidden_gradient));
            objective_train(step, worker->hidden, worker->hidden_gradient);
            update_input(model, token, worker->hidden_gradient);
        }
    }
}

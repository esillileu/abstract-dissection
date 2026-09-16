#include "atomic_float.h"
#include "training.h"

#include <string.h>

static void accumulate_input(
    const Model *model,
    size_t token,
    real *hidden)
{
    size_t dimension = model->embedding_dimension;
    const _Atomic uint32_t *input_row =
        model->input_embeddings + token * dimension;

    for (size_t coordinate = 0; coordinate < dimension; coordinate++)
    {
        hidden[coordinate] += atomic_float_load(&input_row[coordinate]);
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

void cbow_train(ModelStep *step)
{
    Model *model = step->trainer->model;
    Worker *worker = step->worker;
    size_t dimension = model->embedding_dimension;
    size_t radius = context_radius(step);
    size_t context_count = 0;

    memset(worker->hidden, 0, dimension * sizeof(*worker->hidden));
    memset(
        worker->hidden_gradient,
        0,
        dimension * sizeof(*worker->hidden_gradient));

    for (size_t offset = 0; offset <= radius * 2; offset++)
    {
        size_t position;
        if (context_position(worker, radius, offset, &position))
        {
            size_t token = worker->sentence[position];

            accumulate_input(model, token, worker->hidden);
            context_count++;
        }
    }
    if (context_count == 0)
    {
        return;
    }

    for (size_t coordinate = 0; coordinate < dimension; coordinate++)
    {
        worker->hidden[coordinate] /= (real)context_count;
    }
    objective_train(step, worker->hidden, worker->hidden_gradient);

    for (size_t offset = 0; offset <= radius * 2; offset++)
    {
        size_t position;
        if (context_position(worker, radius, offset, &position))
        {
            size_t token = worker->sentence[position];

            update_input(model, token, worker->hidden_gradient);
        }
    }
}

#include "training.h"

int context_position(
    const Worker *worker,
    size_t radius,
    size_t offset,
    size_t *position)
{
    if (offset == radius)
    {
        return 0;
    }

    if (offset < radius)
    {
        size_t left_distance = radius - offset;
        if (worker->sentence_position < left_distance)
        {
            return 0;
        }
        *position = worker->sentence_position - left_distance;
        return 1;
    }

    size_t right_distance = offset - radius;
    if (right_distance >= worker->sentence_length - worker->sentence_position)
    {
        return 0;
    }
    *position = worker->sentence_position + right_distance;
    return 1;
}

size_t context_radius(ModelStep *step)
{
    size_t radius = step->trainer->config.window_radius;
    size_t shrink = rng_next(&step->worker->window_rng) % radius;

    return radius - shrink;
}

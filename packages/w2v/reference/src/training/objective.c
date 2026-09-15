#include "training.h"

void objective_train(
    ModelStep *step,
    const real *hidden,
    real *hidden_gradient)
{
    if (step->trainer->config.objective_kind ==
        OBJECTIVE_HIERARCHICAL_SOFTMAX)
    {
        hierarchical_softmax_train(step, hidden, hidden_gradient);
        return;
    }

    negative_sampling_train(step, hidden, hidden_gradient);
}

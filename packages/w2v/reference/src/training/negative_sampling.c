#include "negative_sampler.h"
#include "sigmoid_table.h"
#include "training.h"

void negative_sampling_train(
    ModelStep *step,
    const real *hidden,
    real *hidden_gradient)
{
    Trainer *trainer = step->trainer;
    size_t dimension = trainer->model->embedding_dimension;
    for (size_t sample_index = 0;
         sample_index <= trainer->config.negative_sample_count;
         sample_index++)
    {
        size_t sampled_token;
        real label;

        if (sample_index == 0)
        {
            sampled_token = step->target_token;
            label = 1;
        }
        else
        {
            uint64_t random_value;
            sampled_token = negative_sampler_draw(
                &trainer->negative_sampler,
                &step->worker->negative_rng,
                &random_value);
            if (sampled_token == 0)
            {
                sampled_token =
                    random_value % (trainer->vocab->size - 1) + 1;
            }
            if (sampled_token == step->target_token)
            {
                continue;
            }
            label = 0;
        }

        _Atomic uint32_t *output_row =
            trainer->model->output_embeddings + sampled_token * dimension;
        real score = objective_score(hidden, output_row, dimension);
        real prediction = sigmoid_table_lookup(
            &trainer->sigmoid_table,
            score);
        real gradient_scale =
            (label - prediction) * step->learning_rate;

        objective_apply_update(
            hidden,
            hidden_gradient,
            output_row,
            dimension,
            gradient_scale);
    }
}

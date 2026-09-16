#include "sigmoid_table.h"
#include "training.h"

void hierarchical_softmax_train(
    ModelStep *step,
    const real *hidden,
    real *hidden_gradient)
{
    Trainer *trainer = step->trainer;
    size_t dimension = trainer->model->embedding_dimension;
    const VocabularyEntry *entry =
        &trainer->vocab->entries[step->target_token];

    for (size_t path_index = 0;
         path_index < entry->huffman_length;
         path_index++)
    {
        size_t output_index = entry->huffman_path[path_index];
        _Atomic uint32_t *output_row =
            trainer->model->output_embeddings + output_index * dimension;
        real score = objective_score(hidden, output_row, dimension);
        if (trainer->config.hs_out_of_range_policy ==
                HS_OUT_OF_RANGE_SKIP &&
            (score <= -trainer->sigmoid_table.max ||
             score >= trainer->sigmoid_table.max))
        {
            continue;
        }
        real prediction = sigmoid_table_lookup(
            &trainer->sigmoid_table,
            score);
        real target = (real)entry->huffman_bits[path_index];
        real gradient_scale =
            (1.0f - target - prediction) * step->learning_rate;

        objective_apply_update(
            hidden,
            hidden_gradient,
            output_row,
            dimension,
            gradient_scale);
    }
}

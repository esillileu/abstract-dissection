use super::{ModelStep, context_position, context_radius, objective};
use crate::atomic_float;

pub fn train(step: &mut ModelStep<'_, '_, '_>) {
    let model = step.trainer.model;
    let dimension = model.embedding_dimension;
    let radius = context_radius(step.window_rng, step.trainer.config.window_radius);
    let center_token = step.target_token;
    let start = center_token * dimension;
    let input_row = &model.input_embeddings[start..start + dimension];

    for offset in 0..=radius * 2 {
        if let Some(position) =
            context_position(step.sentence.len(), step.sentence_position, radius, offset)
        {
            let context_token = step.sentence[position];
            for (coordinate, value) in input_row.iter().enumerate() {
                step.hidden[coordinate] = atomic_float::load(value);
            }
            step.hidden_gradient.fill(0.0);
            objective::train(
                step.trainer,
                context_token,
                step.learning_rate,
                step.negative_rng,
                step.hidden,
                step.hidden_gradient,
            );
            for (coordinate, value) in input_row.iter().enumerate() {
                atomic_float::add(value, step.hidden_gradient[coordinate]);
            }
        }
    }
}

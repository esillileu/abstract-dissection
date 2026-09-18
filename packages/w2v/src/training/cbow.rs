use super::{ModelStep, context_position, context_radius, objective};
use crate::{atomic_float, config::Real};

pub fn train(step: &mut ModelStep<'_, '_>) {
    let model = &step.trainer.model;
    let dimension = model.embedding_dimension;
    let radius = context_radius(step.window_rng, step.trainer.config.window_radius);
    let mut context_count = 0usize;
    step.hidden.fill(0.0);
    step.hidden_gradient.fill(0.0);

    for offset in 0..=radius * 2 {
        if let Some(position) =
            context_position(step.sentence.len(), step.sentence_position, radius, offset)
        {
            let token = step.sentence[position];
            let start = token * dimension;
            let input_row = &model.input_embeddings[start..start + dimension];
            for (coordinate, value) in input_row.iter().enumerate() {
                step.hidden[coordinate] += atomic_float::load(value);
            }
            context_count += 1;
        }
    }
    if context_count == 0 {
        return;
    }
    for coordinate in 0..dimension {
        step.hidden[coordinate] /= context_count as Real;
    }
    objective::train(
        step.trainer,
        step.target_token,
        step.learning_rate,
        step.negative_rng,
        step.hidden,
        step.hidden_gradient,
    );
    for offset in 0..=radius * 2 {
        if let Some(position) =
            context_position(step.sentence.len(), step.sentence_position, radius, offset)
        {
            let token = step.sentence[position];
            let start = token * dimension;
            let input_row = &model.input_embeddings[start..start + dimension];
            for (coordinate, value) in input_row.iter().enumerate() {
                atomic_float::add(value, step.hidden_gradient[coordinate]);
            }
        }
    }
}

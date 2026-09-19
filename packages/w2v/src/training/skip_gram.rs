use super::{ModelStep, ObjectiveLoss, context_position, context_radius_with_policy, objective};
use crate::{atomic_float, config::Status};

#[inline]
pub fn train(step: &mut ModelStep<'_, '_>) -> Result<ObjectiveLoss, Status> {
    let model = &step.trainer.model;
    let dimension = model.embedding_dimension;
    let radius = context_radius_with_policy(
        step.window_rng,
        step.trainer.config.window_radius,
        step.trainer.config.context_policy,
    );
    let center_token = step.target_token;
    let start = center_token * dimension;
    let input_row = &model.input_embeddings[start..start + dimension];

    let mut loss = ObjectiveLoss::default();
    for offset in 0..=radius * 2 {
        if let Some(position) =
            context_position(step.sentence.len(), step.sentence_position, radius, offset)
        {
            let context_token = step.sentence[position];
            for (coordinate, value) in input_row.iter().enumerate() {
                step.hidden[coordinate] = atomic_float::load(value);
            }
            step.hidden_gradient.fill(0.0);
            loss.add(objective::train(
                step.trainer,
                context_token,
                step.learning_rate,
                step.negative_rng,
                step.hidden,
                step.hidden_gradient,
                step.observe_objective,
            )?);
            for (coordinate, value) in input_row.iter().enumerate() {
                atomic_float::add(value, step.hidden_gradient[coordinate]);
            }
        }
    }
    Ok(loss)
}

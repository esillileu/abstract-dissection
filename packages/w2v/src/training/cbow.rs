use super::{
    ModelStep, ObjectiveLoss, ObjectiveScratch, context_position, context_radius_with_policy,
    hierarchical_softmax, negative_sampling, shared_add,
};
use crate::{
    atomic_float,
    config::{ObjectiveKind, Real, Status},
    simd,
};

#[inline]
pub fn train(step: &mut ModelStep<'_, '_>) -> Result<ObjectiveLoss, Status> {
    let model = &step.trainer.model;
    let dimension = model.embedding_dimension;
    let radius = context_radius_with_policy(
        step.window_rng,
        step.trainer.config.window_radius,
        step.trainer.config.context_policy,
    );
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
        return Ok(ObjectiveLoss::default());
    }
    simd::divide_in_place(step.hidden, context_count as Real);
    let loss = match step.trainer.config.objective_kind {
        ObjectiveKind::HierarchicalSoftmax => hierarchical_softmax::train_with_scratch(
            step.trainer,
            step.target_token,
            step.learning_rate,
            step.hidden,
            ObjectiveScratch {
                hidden_gradient: step.hidden_gradient,
                output_snapshot: step.output_snapshot,
            },
            step.observe_objective,
        ),
        ObjectiveKind::NegativeSampling => negative_sampling::train_with_scratch(
            step.trainer,
            step.target_token,
            step.learning_rate,
            step.negative_rng,
            step.hidden,
            ObjectiveScratch {
                hidden_gradient: step.hidden_gradient,
                output_snapshot: step.output_snapshot,
            },
            step.observe_objective,
        ),
    }?;
    for offset in 0..=radius * 2 {
        if let Some(position) =
            context_position(step.sentence.len(), step.sentence_position, radius, offset)
        {
            let token = step.sentence[position];
            let start = token * dimension;
            let input_row = &model.input_embeddings[start..start + dimension];
            for (coordinate, value) in input_row.iter().enumerate() {
                shared_add(
                    value,
                    step.hidden_gradient[coordinate],
                    step.trainer.config.update_strategy,
                );
            }
        }
    }
    Ok(loss)
}

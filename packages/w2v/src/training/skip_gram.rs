use super::{
    ModelStep, ObjectiveLoss, ObjectiveScratch, context_position, context_radius_with_policy,
    hierarchical_softmax, negative_sampling, shared_add,
};
use crate::{
    atomic_float,
    config::{ObjectiveKind, Status},
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
            loss.add(match step.trainer.config.objective_kind {
                ObjectiveKind::HierarchicalSoftmax => hierarchical_softmax::train_with_scratch(
                    step.trainer,
                    context_token,
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
                    context_token,
                    step.learning_rate,
                    step.negative_rng,
                    step.hidden,
                    ObjectiveScratch {
                        hidden_gradient: step.hidden_gradient,
                        output_snapshot: step.output_snapshot,
                    },
                    step.observe_objective,
                ),
            }?);
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

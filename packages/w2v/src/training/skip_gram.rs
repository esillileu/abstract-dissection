use super::{
    ModelStep, ObjectiveLoss, ObjectiveScratch, backend::ParameterBackend,
    backend::SharedModelBackend, context_position, context_radius_with_policy,
    hierarchical_softmax, negative_sampling,
};
use crate::config::{ObjectiveKind, Status};

#[inline]
pub fn train(step: &mut ModelStep<'_, '_>) -> Result<ObjectiveLoss, Status> {
    let mut backend =
        SharedModelBackend::new(&step.trainer.model, step.trainer.config.update_strategy);
    train_with_backend(step, &mut backend)
}

#[inline]
pub fn train_with_backend<B: ParameterBackend>(
    step: &mut ModelStep<'_, '_>,
    backend: &mut B,
) -> Result<ObjectiveLoss, Status> {
    let radius = context_radius_with_policy(
        step.window_rng,
        step.trainer.config.window_radius,
        step.trainer.config.context_policy,
    );
    let center_token = step.target_token;
    let mut loss = ObjectiveLoss::default();
    for offset in 0..=radius * 2 {
        if let Some(position) =
            context_position(step.sentence.len(), step.sentence_position, radius, offset)
        {
            let context_token = step.sentence[position];
            backend.load_input_row(center_token, step.hidden);
            step.hidden_gradient.fill(0.0);
            loss.add(match step.trainer.config.objective_kind {
                ObjectiveKind::HierarchicalSoftmax => hierarchical_softmax::train_with_backend(
                    step.trainer,
                    backend,
                    context_token,
                    step.learning_rate,
                    step.hidden,
                    ObjectiveScratch {
                        hidden_gradient: step.hidden_gradient,
                        output_snapshot: step.output_snapshot,
                    },
                    step.observe_objective,
                ),
                ObjectiveKind::NegativeSampling => negative_sampling::train_with_backend(
                    step.trainer,
                    backend,
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
            backend.apply_input_gradient(center_token, step.hidden_gradient, step.learning_rate);
        }
    }
    Ok(loss)
}

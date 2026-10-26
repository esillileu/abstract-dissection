use super::{
    ModelStep, ObjectiveLoss, ObjectiveScratch, backend::ParameterBackend,
    backend::SharedModelBackend, context_radius_with_policy, hierarchical_softmax, nce,
    negative_sampling,
};
use crate::config::{ContextPolicy, ObjectiveKind, Status};

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
    let radius = if step.trainer.config.context_policy == ContextPolicy::EntireSentence {
        step.sentence.len()
    } else {
        context_radius_with_policy(
            step.window_rng,
            step.trainer.config.window_radius,
            step.trainer.config.context_policy,
        )
    };
    let center_token = step.target_token;
    let mut loss = ObjectiveLoss::default();
    let start = step.sentence_position.saturating_sub(radius);
    let end = step
        .sentence_position
        .saturating_add(radius)
        .saturating_add(1)
        .min(step.sentence.len());
    for position in start..end {
        if position != step.sentence_position {
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
                ObjectiveKind::Nce => nce::train_with_backend(
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

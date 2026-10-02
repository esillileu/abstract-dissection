use super::{
    ModelStep, ObjectiveLoss, ObjectiveScratch, backend::ParameterBackend,
    backend::SharedModelBackend, context_position, context_radius_with_policy,
    hierarchical_softmax, negative_sampling,
};
use crate::{
    config::{ObjectiveKind, Real, Status},
    simd,
};

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
    let mut context_count = 0usize;
    step.hidden.fill(0.0);
    step.hidden_gradient.fill(0.0);

    for offset in 0..=radius * 2 {
        if let Some(position) =
            context_position(step.sentence.len(), step.sentence_position, radius, offset)
        {
            let token = step.sentence[position];
            backend.accumulate_input_row(token, step.hidden);
            context_count += 1;
        }
    }
    if context_count == 0 {
        return Ok(ObjectiveLoss::default());
    }
    simd::divide_in_place(step.hidden, context_count as Real);
    let loss = match step.trainer.config.objective_kind {
        ObjectiveKind::HierarchicalSoftmax => hierarchical_softmax::train_with_backend(
            step.trainer,
            backend,
            step.target_token,
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
            backend.apply_input_gradient(token, step.hidden_gradient, step.learning_rate);
        }
    }
    Ok(loss)
}

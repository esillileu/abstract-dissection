use super::{
    ObjectiveLoss, ObjectiveScratch, backend::ParameterBackend, backend::SharedModelBackend,
};
use crate::{
    config::{Real, Status},
    random::Rng,
    simd,
    trainer::Trainer,
};

pub fn train(
    trainer: &Trainer,
    target_token: usize,
    learning_rate: Real,
    negative_rng: &mut Rng,
    hidden: &[Real],
    hidden_gradient: &mut [Real],
    observe: bool,
) -> Result<ObjectiveLoss, Status> {
    let mut output_snapshot = vec![0.0; hidden.len()];
    let mut backend = SharedModelBackend::new(&trainer.model, trainer.config.update_strategy);
    train_with_backend(
        trainer,
        &mut backend,
        target_token,
        learning_rate,
        negative_rng,
        hidden,
        ObjectiveScratch {
            hidden_gradient,
            output_snapshot: &mut output_snapshot,
        },
        observe,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn train_with_backend<B: ParameterBackend>(
    trainer: &Trainer,
    backend: &mut B,
    target_token: usize,
    learning_rate: Real,
    negative_rng: &mut Rng,
    hidden: &[Real],
    scratch: ObjectiveScratch<'_>,
    observe: bool,
) -> Result<ObjectiveLoss, Status> {
    let mut loss = ObjectiveLoss::default();
    for sample_index in 0..=trainer.config.negative_sample_count {
        let (sampled_token, label) = if sample_index == 0 {
            (target_token, 1.0)
        } else {
            let (mut sampled_token, random_value) = trainer.negative_sampler.draw(negative_rng);
            if sampled_token == 0 {
                sampled_token =
                    (random_value % (trainer.vocab.entries.len() as u64 - 1) + 1) as usize;
            }
            if sampled_token == target_token {
                continue;
            }
            (sampled_token, 0.0)
        };
        backend.load_output_row(sampled_token, scratch.output_snapshot);
        let score = simd::dot(hidden, scratch.output_snapshot);
        let prediction = trainer.sigmoid_table.lookup(score);
        if !prediction.is_finite() {
            return Err(Status::InvalidState);
        }
        if observe {
            let probability = if label == 1.0 {
                prediction
            } else {
                1.0 - prediction
            };
            let value = -(probability as f64).max(f64::EPSILON).ln();
            if !value.is_finite() {
                return Err(Status::InvalidState);
            }
            loss.sum += value;
            loss.count += 1;
        }
        let raw_error = label - prediction;
        backend.apply_output_gradient(
            sampled_token,
            hidden,
            scratch.output_snapshot,
            raw_error,
            learning_rate,
            scratch.hidden_gradient,
        );
    }
    Ok(loss)
}

use super::{
    ObjectiveLoss, ObjectiveScratch, backend::ParameterBackend, backend::SharedModelBackend,
};
use crate::{
    config::{HsOutOfRangePolicy, Real, Status},
    simd,
    trainer::Trainer,
};

pub fn train(
    trainer: &Trainer,
    target_token: usize,
    learning_rate: Real,
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
        hidden,
        ObjectiveScratch {
            hidden_gradient,
            output_snapshot: &mut output_snapshot,
        },
        observe,
    )
}

pub fn train_with_backend<B: ParameterBackend>(
    trainer: &Trainer,
    backend: &mut B,
    target_token: usize,
    learning_rate: Real,
    hidden: &[Real],
    scratch: ObjectiveScratch<'_>,
    observe: bool,
) -> Result<ObjectiveLoss, Status> {
    let entry = &trainer.vocab.entries[target_token];
    let mut loss = ObjectiveLoss::default();
    for (&output_index, &bit) in entry.huffman_path.iter().zip(&entry.huffman_bits) {
        backend.load_output_row(output_index, scratch.output_snapshot);
        let score = simd::dot(hidden, scratch.output_snapshot);
        if trainer.config.hs_out_of_range_policy == HsOutOfRangePolicy::Skip
            && (score <= -trainer.sigmoid_table.max || score >= trainer.sigmoid_table.max)
        {
            continue;
        }
        let prediction = trainer.sigmoid_table.lookup(score);
        let target = bit as Real;
        if !prediction.is_finite() {
            return Err(Status::InvalidState);
        }
        if observe {
            let probability = if bit == 0 {
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
        let raw_error = 1.0 - target - prediction;
        backend.apply_output_gradient(
            output_index,
            hidden,
            scratch.output_snapshot,
            raw_error,
            learning_rate,
            scratch.hidden_gradient,
        );
    }
    Ok(loss)
}

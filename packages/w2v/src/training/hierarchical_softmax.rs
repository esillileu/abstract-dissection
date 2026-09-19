use super::{ObjectiveLoss, objective_apply_update_fast, objective_score};
use crate::{
    config::{HsOutOfRangePolicy, Real, Status},
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
    let dimension = trainer.model.embedding_dimension;
    let entry = &trainer.vocab.entries[target_token];
    let mut loss = ObjectiveLoss::default();
    for (&output_index, &bit) in entry.huffman_path.iter().zip(&entry.huffman_bits) {
        let start = output_index * dimension;
        let output_row = &trainer.model.output_embeddings[start..start + dimension];
        let score = objective_score(hidden, output_row);
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
        let gradient_scale = (1.0 - target - prediction) * learning_rate;
        objective_apply_update_fast(
            hidden,
            hidden_gradient,
            output_row,
            gradient_scale,
            trainer.config.update_strategy,
        );
    }
    Ok(loss)
}

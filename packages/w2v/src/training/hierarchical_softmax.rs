use super::{objective_apply_update, objective_score};
use crate::{
    config::{HsOutOfRangePolicy, Real},
    trainer::Trainer,
};

pub fn train(
    trainer: &Trainer,
    target_token: usize,
    learning_rate: Real,
    hidden: &[Real],
    hidden_gradient: &mut [Real],
) {
    let dimension = trainer.model.embedding_dimension;
    let entry = &trainer.vocab.entries[target_token];
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
        let gradient_scale = (1.0 - target - prediction) * learning_rate;
        objective_apply_update(hidden, hidden_gradient, output_row, gradient_scale);
    }
}

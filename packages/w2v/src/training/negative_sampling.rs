use super::{objective_apply_update, objective_score};
use crate::{config::Real, random::Rng, trainer::Trainer};

pub fn train(
    trainer: &Trainer<'_>,
    target_token: usize,
    learning_rate: Real,
    negative_rng: &mut Rng,
    hidden: &[Real],
    hidden_gradient: &mut [Real],
) {
    let dimension = trainer.model.embedding_dimension;
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
        let start = sampled_token * dimension;
        let output_row = &trainer.model.output_embeddings[start..start + dimension];
        let score = objective_score(hidden, output_row);
        let prediction = trainer.sigmoid_table.lookup(score);
        let gradient_scale = (label - prediction) * learning_rate;
        objective_apply_update(hidden, hidden_gradient, output_row, gradient_scale);
    }
}

//! Binary NCE with fixed Z(h)=1. See the W2V2 README for the derivation.
use super::{ObjectiveLoss, ObjectiveScratch, ParameterBackend};
use crate::{
    config::{Real, Status},
    random::Rng,
    simd,
    trainer::Trainer,
};

/// Returns negative log binary likelihood and its score derivative.
/// z = u_w^T h - log(k q(w)); Z(h) is fixed to one.
pub fn binary_loss_gradient(
    score: f64,
    noise_probability: f64,
    k: usize,
    data: bool,
) -> Result<(f64, f64), Status> {
    if !score.is_finite()
        || !noise_probability.is_finite()
        || noise_probability <= 0.0
        || noise_probability > 1.0
        || k == 0
    {
        return Err(Status::InvalidArgument);
    }
    let z = score - (k as f64 * noise_probability).ln();
    let signed = if data { -z } else { z };
    let loss = signed.max(0.0) + (-signed.abs()).exp().ln_1p();
    // Differentiate softplus directly, avoiding sigmoid(z)-1 cancellation.
    let gradient_logit = if data { -z } else { z };
    let magnitude = if gradient_logit >= 0.0 {
        1.0 / (1.0 + (-gradient_logit).exp())
    } else {
        let e = gradient_logit.exp();
        e / (1.0 + e)
    };
    Ok((loss, if data { -magnitude } else { magnitude }))
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
    // Snapshot each touched row once and defer all writes until every binary
    // term has been evaluated. This is local step consistency, not a global
    // snapshot against concurrent workers. Preserve first-occurrence order.
    let mut rows: Vec<(usize, Vec<Real>, f64)> = Vec::new();
    for index in 0..=trainer.config.negative_sample_count {
        let data = index == 0;
        let token = if data {
            target_token
        } else {
            trainer.nce_noise_sampler.draw(negative_rng)
        };
        // Keep repeated samples and collisions with the positive word: all k
        // draws belong to the same unconditional noise distribution.
        let row_index = match rows.iter().position(|row| row.0 == token) {
            Some(index) => index,
            None => {
                backend.load_output_row(token, scratch.output_snapshot);
                rows.push((token, scratch.output_snapshot.to_vec(), 0.0));
                rows.len() - 1
            }
        };
        let (_, snapshot, error) = &mut rows[row_index];
        let score = simd::dot(hidden, snapshot) as f64;
        let (value, derivative) = binary_loss_gradient(
            score,
            trainer.nce_noise_sampler.probability(token),
            trainer.config.negative_sample_count,
            data,
        )?;
        if observe {
            loss.sum += value;
            loss.count += 1;
        }
        *error -= derivative;
    }
    for (token, mut snapshot, error) in rows {
        backend.apply_output_gradient(
            token,
            hidden,
            &mut snapshot,
            error as Real,
            learning_rate,
            scratch.hidden_gradient,
        );
    }
    Ok(loss)
}

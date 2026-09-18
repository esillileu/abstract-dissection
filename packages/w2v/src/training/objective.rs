use super::{hierarchical_softmax, negative_sampling};
use crate::{
    config::{ObjectiveKind, Real},
    random::Rng,
    trainer::Trainer,
};

pub fn train(
    trainer: &Trainer,
    target_token: usize,
    learning_rate: Real,
    negative_rng: &mut Rng,
    hidden: &[Real],
    hidden_gradient: &mut [Real],
) {
    if trainer.config.objective_kind == ObjectiveKind::HierarchicalSoftmax {
        hierarchical_softmax::train(
            trainer,
            target_token,
            learning_rate,
            hidden,
            hidden_gradient,
        );
    } else {
        negative_sampling::train(
            trainer,
            target_token,
            learning_rate,
            negative_rng,
            hidden,
            hidden_gradient,
        );
    }
}

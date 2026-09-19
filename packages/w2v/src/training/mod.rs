//! Context traversal and objective arithmetic from the modular C oracle.
use crate::{
    atomic_float,
    config::{ContextPolicy, Real, Status, UpdateStrategy},
    random::Rng,
    trainer::Trainer,
};
use std::sync::atomic::AtomicU32;

mod cbow;
mod hierarchical_softmax;
mod negative_sampling;
mod objective;
mod skip_gram;
pub mod worker;

pub use cbow::train as cbow_train;
pub use hierarchical_softmax::train as hierarchical_softmax_train;
pub use negative_sampling::train as negative_sampling_train;
pub use objective::train as objective_train;
pub use skip_gram::train as skip_gram_train;

#[inline(always)]
pub(crate) fn shared_add(destination: &AtomicU32, delta: Real, strategy: UpdateStrategy) {
    match strategy {
        UpdateStrategy::AtomicCas => atomic_float::add(destination, delta),
        UpdateStrategy::Hogwild => atomic_float::add_hogwild(destination, delta),
    }
}

/// The per-target contract needed by both model kinds. Stage 6 owns the
/// sentence and scratch buffers and creates a step for each trained target.
pub struct ModelStep<'t, 'w> {
    pub trainer: &'t Trainer,
    pub target_token: usize,
    pub learning_rate: Real,
    pub sentence: &'w [usize],
    pub sentence_position: usize,
    pub hidden: &'w mut [Real],
    pub hidden_gradient: &'w mut [Real],
    pub window_rng: &'w mut Rng,
    pub negative_rng: &'w mut Rng,
    pub observe_objective: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ObjectiveLoss {
    pub sum: f64,
    pub count: u64,
}

impl ObjectiveLoss {
    pub fn add(&mut self, other: Self) {
        self.sum += other.sum;
        self.count += other.count;
    }
}

pub fn context_position(
    sentence_length: usize,
    sentence_position: usize,
    radius: usize,
    offset: usize,
) -> Option<usize> {
    if offset == radius {
        return None;
    }
    if offset < radius {
        let left_distance = radius - offset;
        if sentence_position < left_distance {
            None
        } else {
            Some(sentence_position - left_distance)
        }
    } else {
        let right_distance = offset - radius;
        if right_distance >= sentence_length - sentence_position {
            None
        } else {
            Some(sentence_position + right_distance)
        }
    }
}

pub fn context_radius(window_rng: &mut Rng, window_radius: usize) -> usize {
    context_radius_with_policy(window_rng, window_radius, ContextPolicy::Dynamic)
}

pub fn context_radius_with_policy(
    window_rng: &mut Rng,
    window_radius: usize,
    policy: ContextPolicy,
) -> usize {
    match policy {
        ContextPolicy::Fixed => window_radius,
        ContextPolicy::Dynamic => {
            let shrink = window_rng.next_u64() % window_radius as u64;
            window_radius - shrink as usize
        }
    }
}

#[inline(always)]
pub fn objective_score(hidden: &[Real], output_row: &[AtomicU32]) -> Real {
    assert_eq!(hidden.len(), output_row.len());
    let mut score = 0.0;
    for coordinate in 0..hidden.len() {
        let output_value = atomic_float::load(&output_row[coordinate]);
        score += hidden[coordinate] * output_value;
    }
    score
}

pub fn objective_apply_update_checked(
    hidden: &[Real],
    hidden_gradient: &mut [Real],
    output_row: &[AtomicU32],
    gradient_scale: Real,
) -> Status {
    assert_eq!(hidden.len(), output_row.len());
    assert_eq!(hidden_gradient.len(), hidden.len());
    if !gradient_scale.is_finite() || hidden.iter().any(|value| !value.is_finite()) {
        return Status::InvalidState;
    }
    for coordinate in 0..hidden.len() {
        let output_value = atomic_float::load(&output_row[coordinate]);
        let next = hidden_gradient[coordinate] + gradient_scale * output_value;
        let delta = gradient_scale * hidden[coordinate];
        if !output_value.is_finite()
            || !next.is_finite()
            || !delta.is_finite()
            || !(output_value + delta).is_finite()
        {
            return Status::InvalidState;
        }
    }
    for coordinate in 0..hidden.len() {
        let output_value = atomic_float::load(&output_row[coordinate]);
        hidden_gradient[coordinate] += gradient_scale * output_value;
    }
    for coordinate in 0..hidden.len() {
        let delta = gradient_scale * hidden[coordinate];
        atomic_float::add(&output_row[coordinate], delta);
    }
    Status::Ok
}

/// Applies a validated objective update without repeating state validation for
/// every coordinate. Each coordinate's gradient reads its output value before
/// that same coordinate is updated.
pub(crate) fn objective_apply_update_fast(
    hidden: &[Real],
    hidden_gradient: &mut [Real],
    output_row: &[AtomicU32],
    gradient_scale: Real,
    update_strategy: UpdateStrategy,
) {
    match update_strategy {
        UpdateStrategy::AtomicCas => {
            objective_apply_update_cas(hidden, hidden_gradient, output_row, gradient_scale)
        }
        UpdateStrategy::Hogwild => {
            objective_apply_update_hogwild(hidden, hidden_gradient, output_row, gradient_scale)
        }
    }
}

#[inline(always)]
fn objective_apply_update_cas(
    hidden: &[Real],
    hidden_gradient: &mut [Real],
    output_row: &[AtomicU32],
    gradient_scale: Real,
) {
    for coordinate in 0..hidden.len() {
        let output_value = atomic_float::load(&output_row[coordinate]);
        hidden_gradient[coordinate] += gradient_scale * output_value;
        let delta = gradient_scale * hidden[coordinate];
        atomic_float::add(&output_row[coordinate], delta);
    }
}

#[inline(always)]
fn objective_apply_update_hogwild(
    hidden: &[Real],
    hidden_gradient: &mut [Real],
    output_row: &[AtomicU32],
    gradient_scale: Real,
) {
    for coordinate in 0..hidden.len() {
        let output_value = atomic_float::load(&output_row[coordinate]);
        hidden_gradient[coordinate] += gradient_scale * output_value;
        let delta = gradient_scale * hidden[coordinate];
        atomic_float::add_hogwild(&output_row[coordinate], delta);
    }
}

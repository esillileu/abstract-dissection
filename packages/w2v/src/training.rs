//! Context traversal and objective arithmetic from the modular C oracle.
use crate::{atomic_float, config::Real, random::Rng};
use std::sync::atomic::AtomicU32;

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
    let shrink = window_rng.next_u64() % window_radius as u64;
    window_radius - shrink as usize
}

pub fn objective_score(hidden: &[Real], output_row: &[AtomicU32]) -> Real {
    assert_eq!(hidden.len(), output_row.len());
    let mut score = 0.0;
    for coordinate in 0..hidden.len() {
        let output_value = atomic_float::load(&output_row[coordinate]);
        score += hidden[coordinate] * output_value;
    }
    score
}

pub fn objective_apply_update(
    hidden: &[Real],
    hidden_gradient: &mut [Real],
    output_row: &[AtomicU32],
    gradient_scale: Real,
) {
    assert_eq!(hidden.len(), output_row.len());
    assert_eq!(hidden_gradient.len(), hidden.len());
    for coordinate in 0..hidden.len() {
        let output_value = atomic_float::load(&output_row[coordinate]);
        hidden_gradient[coordinate] += gradient_scale * output_value;
    }
    for coordinate in 0..hidden.len() {
        let delta = gradient_scale * hidden[coordinate];
        atomic_float::add(&output_row[coordinate], delta);
    }
}

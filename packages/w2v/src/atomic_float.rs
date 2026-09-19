//! Race-free float bit-pattern operations from `reference/src/model/atomic_float.c`.
use crate::config::Real;
use std::sync::atomic::{AtomicU32, Ordering};

#[inline(always)]
pub fn load(value: &AtomicU32) -> Real {
    Real::from_bits(value.load(Ordering::Relaxed))
}

#[inline(always)]
pub fn store(destination: &AtomicU32, value: Real) {
    destination.store(value.to_bits(), Ordering::Relaxed);
}

#[inline(always)]
pub fn add(destination: &AtomicU32, delta: Real) {
    let mut expected_bits = destination.load(Ordering::Relaxed);
    loop {
        let desired_bits = (Real::from_bits(expected_bits) + delta).to_bits();
        match destination.compare_exchange_weak(
            expected_bits,
            desired_bits,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => return,
            Err(actual_bits) => expected_bits = actual_bits,
        }
    }
}

#[inline(always)]
pub fn add_hogwild(destination: &AtomicU32, delta: Real) {
    let value = load(destination);
    store(destination, value + delta);
}

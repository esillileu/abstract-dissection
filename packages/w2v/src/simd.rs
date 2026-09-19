//! Small vector kernels. Shared embeddings are loaded atomically before SIMD
//! arithmetic is applied to local values.

use crate::{atomic_float, config::Real};
use std::sync::{OnceLock, atomic::AtomicU32};

#[inline]
pub fn shared_dot(left: &[Real], right: &[AtomicU32]) -> Real {
    assert_eq!(left.len(), right.len());
    if is_available() {
        // SAFETY: the feature check is performed once per call. Shared values
        // are copied with atomic loads before the SIMD multiplication.
        unsafe { shared_dot_avx2(left, right) }
    } else {
        shared_dot_scalar(left, right)
    }
}

#[inline]
pub fn divide_in_place(values: &mut [Real], divisor: Real) {
    if is_available() {
        // SAFETY: the feature check is performed once per call and the AVX2
        // routine only accesses the caller-owned mutable slice.
        unsafe { divide_avx2(values, divisor) }
    } else {
        divide_scalar(values, divisor);
    }
}

#[inline]
fn is_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    })
}

#[inline(always)]
fn divide_scalar(values: &mut [Real], divisor: Real) {
    for value in values {
        *value /= divisor;
    }
}

#[inline(always)]
fn shared_dot_scalar(left: &[Real], right: &[AtomicU32]) -> Real {
    let mut result = 0.0;
    for (left, right) in left.iter().zip(right) {
        result += left * atomic_float::load(right);
    }
    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn shared_dot_avx2(left: &[Real], right: &[AtomicU32]) -> Real {
    use std::arch::x86_64::{_mm256_loadu_ps, _mm256_mul_ps, _mm256_storeu_ps};

    let mut result = 0.0;
    let mut index = 0;
    while index + 8 <= left.len() {
        let right_values = [
            atomic_float::load(&right[index]),
            atomic_float::load(&right[index + 1]),
            atomic_float::load(&right[index + 2]),
            atomic_float::load(&right[index + 3]),
            atomic_float::load(&right[index + 4]),
            atomic_float::load(&right[index + 5]),
            atomic_float::load(&right[index + 6]),
            atomic_float::load(&right[index + 7]),
        ];
        let left_vector = unsafe { _mm256_loadu_ps(left.as_ptr().add(index)) };
        let right_vector = unsafe { _mm256_loadu_ps(right_values.as_ptr()) };
        let mut products = [0.0; 8];
        unsafe {
            _mm256_storeu_ps(
                products.as_mut_ptr(),
                _mm256_mul_ps(left_vector, right_vector),
            );
        }
        for product in products {
            result += product;
        }
        index += 8;
    }
    result + shared_dot_scalar(&left[index..], &right[index..])
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn shared_dot_avx2(left: &[Real], right: &[AtomicU32]) -> Real {
    shared_dot_scalar(left, right)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn divide_avx2(values: &mut [Real], divisor: Real) {
    use std::arch::x86_64::{_mm256_div_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps};

    let scalar_divisor = divisor;
    let divisor = _mm256_set1_ps(divisor);
    let mut index = 0;
    while index + 8 <= values.len() {
        let vector = unsafe { _mm256_loadu_ps(values.as_ptr().add(index)) };
        unsafe {
            _mm256_storeu_ps(
                values.as_mut_ptr().add(index),
                _mm256_div_ps(vector, divisor),
            );
        }
        index += 8;
    }
    divide_scalar(&mut values[index..], scalar_divisor);
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn divide_avx2(values: &mut [Real], divisor: Real) {
    divide_scalar(values, divisor);
}

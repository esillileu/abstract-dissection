//! Small local-vector kernels. Shared embeddings remain atomic and are never
//! accessed through these non-atomic routines.

use crate::config::Real;
use std::sync::OnceLock;

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

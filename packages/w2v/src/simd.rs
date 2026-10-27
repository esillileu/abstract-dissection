//! Small vector kernels. Shared embeddings are loaded atomically before SIMD
//! arithmetic is applied to local values.

use crate::{atomic_float, config::Real};
use std::sync::{OnceLock, atomic::AtomicU32};

#[inline]
pub fn shared_dot(left: &[Real], right: &[AtomicU32]) -> Real {
    assert_eq!(left.len(), right.len());
    if left.len() >= 16 && is_available() {
        // SAFETY: the feature check is performed once per call. Shared values
        // are copied with atomic loads before the SIMD multiplication.
        unsafe { shared_dot_avx2(left, right) }
    } else {
        shared_dot_scalar(left, right)
    }
}

#[inline]
pub fn dot(left: &[Real], right: &[Real]) -> Real {
    assert_eq!(left.len(), right.len());
    if left.len() >= 16 && is_available() {
        // SAFETY: the feature check is performed once per call and both slices
        // are caller-owned local values.
        unsafe { dot_avx2(left, right) }
    } else {
        dot_scalar(left, right)
    }
}

#[inline]
pub fn scaled_accumulate(destination: &mut [Real], source: &[Real], scale: Real) {
    assert_eq!(destination.len(), source.len());
    if destination.len() >= 16 && is_available() {
        // SAFETY: the feature check is performed once per call and both slices
        // are caller-owned local values.
        unsafe { scaled_accumulate_avx2(destination, source, scale) }
    } else {
        scaled_accumulate_scalar(destination, source, scale);
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
pub fn load_shared_row(destination: &mut [Real], source: &[AtomicU32]) {
    assert_eq!(destination.len(), source.len());
    if destination.len() >= 16 && is_available() {
        unsafe { load_shared_row_avx2(destination, source) }
    } else {
        for (dst, src) in destination.iter_mut().zip(source) {
            *dst = atomic_float::load(src);
        }
    }
}

#[inline]
#[allow(dead_code)]
pub fn store_shared_row(destination: &[AtomicU32], source: &[Real]) {
    assert_eq!(destination.len(), source.len());
    if destination.len() >= 16 && is_available() {
        unsafe { store_shared_row_avx2(destination, source) }
    } else {
        for (dst, src) in destination.iter().zip(source) {
            atomic_float::store(dst, *src);
        }
    }
}

#[inline]
pub fn add_shared_row(destination: &[AtomicU32], delta: &[Real]) {
    assert_eq!(destination.len(), delta.len());
    if destination.len() >= 16 && is_available() {
        unsafe { add_shared_row_avx2(destination, delta) }
    } else {
        for (dst, d) in destination.iter().zip(delta) {
            atomic_float::add_hogwild(dst, *d);
        }
    }
}

#[inline]
pub fn apply_output_gradient_hogwild(
    output_row: &[AtomicU32],
    output_snapshot: &[Real],
    hidden: &[Real],
    hidden_gradient: &mut [Real],
    gradient_scale: Real,
) {
    assert_eq!(output_row.len(), output_snapshot.len());
    assert_eq!(output_snapshot.len(), hidden.len());
    assert_eq!(hidden.len(), hidden_gradient.len());
    if output_row.len() >= 16 && is_available() {
        unsafe {
            apply_output_gradient_hogwild_avx2(
                output_row,
                output_snapshot,
                hidden,
                hidden_gradient,
                gradient_scale,
            )
        }
    } else {
        apply_output_gradient_hogwild_scalar(
            output_row,
            output_snapshot,
            hidden,
            hidden_gradient,
            gradient_scale,
        );
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

#[inline(always)]
fn dot_scalar(left: &[Real], right: &[Real]) -> Real {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

#[inline(always)]
fn scaled_accumulate_scalar(destination: &mut [Real], source: &[Real], scale: Real) {
    for (destination, source) in destination.iter_mut().zip(source) {
        *destination += source * scale;
    }
}

#[inline(always)]
fn apply_output_gradient_hogwild_scalar(
    output_row: &[AtomicU32],
    output_snapshot: &[Real],
    hidden: &[Real],
    hidden_gradient: &mut [Real],
    gradient_scale: Real,
) {
    for i in 0..output_row.len() {
        hidden_gradient[i] += output_snapshot[i] * gradient_scale;
        atomic_float::store(&output_row[i], output_snapshot[i] + hidden[i] * gradient_scale);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_avx2(left: &[Real], right: &[Real]) -> Real {
    use std::arch::x86_64::{
        _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    };

    let mut accumulator = _mm256_setzero_ps();
    let mut index = 0;
    while index + 8 <= left.len() {
        let left_vector = unsafe { _mm256_loadu_ps(left.as_ptr().add(index)) };
        let right_vector = unsafe { _mm256_loadu_ps(right.as_ptr().add(index)) };
        accumulator = _mm256_fmadd_ps(left_vector, right_vector, accumulator);
        index += 8;
    }
    let mut lanes = [0.0; 8];
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), accumulator) };
    lanes.into_iter().sum::<Real>() + dot_scalar(&left[index..], &right[index..])
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn dot_avx2(left: &[Real], right: &[Real]) -> Real {
    dot_scalar(left, right)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn scaled_accumulate_avx2(destination: &mut [Real], source: &[Real], scale: Real) {
    use std::arch::x86_64::{_mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps};

    let scale_vector = _mm256_set1_ps(scale);
    let mut index = 0;
    while index + 8 <= destination.len() {
        let destination_vector = unsafe { _mm256_loadu_ps(destination.as_ptr().add(index)) };
        let source_vector = unsafe { _mm256_loadu_ps(source.as_ptr().add(index)) };
        unsafe {
            _mm256_storeu_ps(
                destination.as_mut_ptr().add(index),
                _mm256_fmadd_ps(source_vector, scale_vector, destination_vector),
            );
        }
        index += 8;
    }
    scaled_accumulate_scalar(&mut destination[index..], &source[index..], scale);
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn scaled_accumulate_avx2(destination: &mut [Real], source: &[Real], scale: Real) {
    scaled_accumulate_scalar(destination, source, scale);
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn load_shared_row_avx2(destination: &mut [Real], source: &[AtomicU32]) {
    use std::arch::x86_64::{_mm256_loadu_ps, _mm256_storeu_ps};

    let mut index = 0;
    while index + 8 <= destination.len() {
        unsafe {
            let v = _mm256_loadu_ps(source.as_ptr().add(index) as *const f32);
            _mm256_storeu_ps(destination.as_mut_ptr().add(index), v);
        }
        index += 8;
    }
    for (dst, src) in destination[index..].iter_mut().zip(&source[index..]) {
        *dst = atomic_float::load(src);
    }
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn load_shared_row_avx2(destination: &mut [Real], source: &[AtomicU32]) {
    for (dst, src) in destination.iter_mut().zip(source) {
        *dst = atomic_float::load(src);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn store_shared_row_avx2(destination: &[AtomicU32], source: &[Real]) {
    use std::arch::x86_64::{_mm256_loadu_ps, _mm256_storeu_ps};

    let mut index = 0;
    while index + 8 <= destination.len() {
        unsafe {
            let v = _mm256_loadu_ps(source.as_ptr().add(index));
            _mm256_storeu_ps(destination.as_ptr().add(index) as *mut f32, v);
        }
        index += 8;
    }
    for (dst, src) in destination[index..].iter().zip(&source[index..]) {
        atomic_float::store(dst, *src);
    }
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn store_shared_row_avx2(destination: &[AtomicU32], source: &[Real]) {
    for (dst, src) in destination.iter().zip(source) {
        atomic_float::store(dst, *src);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_shared_row_avx2(destination: &[AtomicU32], delta: &[Real]) {
    use std::arch::x86_64::{_mm256_add_ps, _mm256_loadu_ps, _mm256_storeu_ps};

    let mut index = 0;
    while index + 8 <= destination.len() {
        unsafe {
            let cur = _mm256_loadu_ps(destination.as_ptr().add(index) as *const f32);
            let d = _mm256_loadu_ps(delta.as_ptr().add(index));
            let sum = _mm256_add_ps(cur, d);
            _mm256_storeu_ps(destination.as_ptr().add(index) as *mut f32, sum);
        }
        index += 8;
    }
    for (dst, d) in destination[index..].iter().zip(&delta[index..]) {
        atomic_float::add_hogwild(dst, *d);
    }
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn add_shared_row_avx2(destination: &[AtomicU32], delta: &[Real]) {
    for (dst, d) in destination.iter().zip(delta) {
        atomic_float::add_hogwild(dst, *d);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn apply_output_gradient_hogwild_avx2(
    output_row: &[AtomicU32],
    output_snapshot: &[Real],
    hidden: &[Real],
    hidden_gradient: &mut [Real],
    gradient_scale: Real,
) {
    use std::arch::x86_64::{_mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps};

    let scale_vector = _mm256_set1_ps(gradient_scale);
    let mut index = 0;
    while index + 8 <= output_row.len() {
        unsafe {
            let snap = _mm256_loadu_ps(output_snapshot.as_ptr().add(index));
            let hid = _mm256_loadu_ps(hidden.as_ptr().add(index));
            let hg = _mm256_loadu_ps(hidden_gradient.as_ptr().add(index));

            let new_hg = _mm256_fmadd_ps(snap, scale_vector, hg);
            let new_out = _mm256_fmadd_ps(hid, scale_vector, snap);

            _mm256_storeu_ps(hidden_gradient.as_mut_ptr().add(index), new_hg);
            _mm256_storeu_ps(output_row.as_ptr().add(index) as *mut f32, new_out);
        }
        index += 8;
    }
    apply_output_gradient_hogwild_scalar(
        &output_row[index..],
        &output_snapshot[index..],
        &hidden[index..],
        &mut hidden_gradient[index..],
        gradient_scale,
    );
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn apply_output_gradient_hogwild_avx2(
    output_row: &[AtomicU32],
    output_snapshot: &[Real],
    hidden: &[Real],
    hidden_gradient: &mut [Real],
    gradient_scale: Real,
) {
    apply_output_gradient_hogwild_scalar(
        output_row,
        output_snapshot,
        hidden,
        hidden_gradient,
        gradient_scale,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_kernels_match_scalar_math() {
        let left: Vec<Real> = (0..35).map(|index| index as Real * 0.125 - 2.0).collect();
        let right: Vec<Real> = (0..35).map(|index| index as Real * -0.0625 + 1.0).collect();
        let expected_dot: Real = left
            .iter()
            .zip(&right)
            .map(|(left, right)| left * right)
            .sum();
        assert!((dot(&left, &right) - expected_dot).abs() < 1e-5);

        let mut destination = left.clone();
        let mut expected = left.clone();
        for (destination, source) in expected.iter_mut().zip(&right) {
            *destination = source.mul_add(0.25, *destination);
        }
        scaled_accumulate(&mut destination, &right, 0.25);
        assert_eq!(destination, expected);

        let output_row: Vec<AtomicU32> = (0..35)
            .map(|i| AtomicU32::new((i as Real * 0.1).to_bits()))
            .collect();
        let output_snapshot: Vec<Real> = (0..35).map(|i| i as Real * 0.1).collect();
        let hidden: Vec<Real> = (0..35).map(|i| i as Real * -0.05 + 0.5).collect();
        let mut hidden_gradient: Vec<Real> = (0..35).map(|i| i as Real * 0.02).collect();
        let scale = 0.05;

        let mut expected_hg = hidden_gradient.clone();
        scaled_accumulate(&mut expected_hg, &output_snapshot, scale);
        let mut expected_out = output_snapshot.clone();
        scaled_accumulate(&mut expected_out, &hidden, scale);

        apply_output_gradient_hogwild(
            &output_row,
            &output_snapshot,
            &hidden,
            &mut hidden_gradient,
            scale,
        );

        assert_eq!(hidden_gradient, expected_hg);
        for (shared, exp) in output_row.iter().zip(&expected_out) {
            assert_eq!(atomic_float::load(shared), *exp);
        }
    }
}

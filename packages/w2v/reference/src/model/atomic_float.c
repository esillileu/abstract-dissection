#include "atomic_float.h"

#include <string.h>

static uint32_t float_to_bits(real value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static real bits_to_float(uint32_t bits)
{
    real value;

    memcpy(&value, &bits, sizeof(value));
    return value;
}

real atomic_float_load(const _Atomic uint32_t *value)
{
    uint32_t bits = atomic_load_explicit(value, memory_order_relaxed);

    return bits_to_float(bits);
}

void atomic_float_store(_Atomic uint32_t *destination, real value)
{
    atomic_store_explicit(
        destination,
        float_to_bits(value),
        memory_order_relaxed);
}

void atomic_float_add(_Atomic uint32_t *destination, real delta)
{
    uint32_t expected_bits = atomic_load_explicit(
        destination,
        memory_order_relaxed);

    for (;;)
    {
        real current_value = bits_to_float(expected_bits);
        real updated_value = current_value + delta;
        uint32_t desired_bits = float_to_bits(updated_value);

        if (atomic_compare_exchange_weak_explicit(
                destination,
                &expected_bits,
                desired_bits,
                memory_order_relaxed,
                memory_order_relaxed))
        {
            return;
        }
    }
}

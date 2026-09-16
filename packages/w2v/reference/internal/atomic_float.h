#ifndef W2V_INTERNAL_ATOMIC_FLOAT_H
#define W2V_INTERNAL_ATOMIC_FLOAT_H

#include "w2v/config.h"

#include <stdatomic.h>

real atomic_float_load(const _Atomic uint32_t *value);
void atomic_float_store(_Atomic uint32_t *destination, real value);
void atomic_float_add(_Atomic uint32_t *destination, real delta);

#endif

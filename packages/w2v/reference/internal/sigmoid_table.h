#ifndef W2V_INTERNAL_SIGMOID_TABLE_H
#define W2V_INTERNAL_SIGMOID_TABLE_H

#include "w2v/model.h"

Status sigmoid_table_initialize(
    SigmoidTable *table,
    size_t table_size,
    real maximum);
void sigmoid_table_free(SigmoidTable *table);
real sigmoid_table_lookup(const SigmoidTable *table, real value);

#endif

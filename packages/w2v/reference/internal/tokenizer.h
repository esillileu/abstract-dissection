#ifndef W2V_INTERNAL_TOKENIZER_H
#define W2V_INTERNAL_TOKENIZER_H

#include "w2v/config.h"

#include <stdio.h>

Status token_read(
    FILE *file,
    char token[MAX_TOKEN_LENGTH],
    int *at_eof);

#endif

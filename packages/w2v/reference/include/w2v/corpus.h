#ifndef W2V_CORPUS_H
#define W2V_CORPUS_H

#include "config.h"

typedef struct
{
    char *path;
    size_t byte_size;
} Corpus;

Corpus *corpus_create(const char *path, Status *status);
void corpus_destroy(Corpus **corpus);

#endif

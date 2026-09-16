#include "w2v/corpus.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *copy_path(const char *path)
{
    size_t byte_count = strlen(path) + 1;
    char *copy = malloc(byte_count);

    if (copy != NULL)
    {
        memcpy(copy, path, byte_count);
    }
    return copy;
}

Corpus *corpus_create(const char *path, Status *status)
{
    if (path == NULL || *path == '\0')
    {
        if (status != NULL)
        {
            *status = STATUS_INVALID_ARGUMENT;
        }
        return NULL;
    }

    FILE *file = fopen(path, "rb");
    if (file == NULL)
    {
        if (status != NULL)
        {
            *status = STATUS_IO_ERROR;
        }
        return NULL;
    }

    Status result = STATUS_OK;
    long file_size = 0;
    if (fseek(file, 0, SEEK_END) != 0)
    {
        result = STATUS_IO_ERROR;
    }
    else
    {
        file_size = ftell(file);
        if (file_size < 0)
        {
            result = STATUS_IO_ERROR;
        }
    }
    if (fclose(file) != 0)
    {
        result = STATUS_IO_ERROR;
    }

    Corpus *corpus = NULL;
    if (result == STATUS_OK)
    {
        corpus = calloc(1, sizeof(*corpus));
        if (corpus == NULL)
        {
            result = STATUS_OUT_OF_MEMORY;
        }
    }
    if (result == STATUS_OK)
    {
        corpus->path = copy_path(path);
        if (corpus->path == NULL)
        {
            result = STATUS_OUT_OF_MEMORY;
        }
    }
    if (result == STATUS_OK)
    {
        corpus->byte_size = (size_t)file_size;
    }
    else
    {
        corpus_destroy(&corpus);
    }

    if (status != NULL)
    {
        *status = result;
    }
    return corpus;
}

void corpus_destroy(Corpus **corpus)
{
    if (corpus == NULL || *corpus == NULL)
    {
        return;
    }

    free((*corpus)->path);
    free(*corpus);
    *corpus = NULL;
}

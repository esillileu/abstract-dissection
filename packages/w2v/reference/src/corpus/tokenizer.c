#include "tokenizer.h"

#include <string.h>

Status token_read(
    FILE *file,
    char token[MAX_TOKEN_LENGTH],
    int *at_eof)
{
    if (file == NULL || token == NULL || at_eof == NULL)
    {
        return STATUS_INVALID_ARGUMENT;
    }

    *at_eof = 0;
    size_t token_length = 0;
    for (;;)
    {
        int character = fgetc(file);
        if (character == EOF)
        {
            if (ferror(file))
            {
                return STATUS_IO_ERROR;
            }
            *at_eof = 1;
            break;
        }
        if (character == '\r')
        {
            continue;
        }
        if (character == '\n')
        {
            if (token_length > 0)
            {
                ungetc(character, file);
                break;
            }
            memcpy(token, "</s>", 5);
            return STATUS_OK;
        }
        if (character == ' ' || character == '\t')
        {
            if (token_length > 0)
            {
                break;
            }
            continue;
        }
        if (token_length < MAX_TOKEN_LENGTH - 1)
        {
            token[token_length] = (char)character;
            token_length++;
        }
    }

    token[token_length] = '\0';
    return STATUS_OK;
}

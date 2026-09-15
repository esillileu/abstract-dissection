#include "huffman.h"
#include "tokenizer.h"
#include "w2v/vocab.h"

#include <stdlib.h>
#include <string.h>

static char *copy_token(const char *token)
{
    size_t byte_count = strlen(token) + 1;
    char *copy = malloc(byte_count);

    if (copy != NULL)
    {
        memcpy(copy, token, byte_count);
    }
    return copy;
}

static size_t token_hash(const char *token, size_t capacity)
{
    uint64_t hash = 0;

    for (; *token != '\0'; token++)
    {
        hash = hash * 257 + (unsigned char)*token;
    }
    return capacity == 0 ? 0 : (size_t)(hash % capacity);
}

size_t vocab_find(const Vocabulary *vocab, const char *token)
{
    if (vocab == NULL || token == NULL ||
        vocab->hash_slots == NULL || vocab->hash_capacity == 0)
    {
        return VOCABULARY_NOT_FOUND;
    }

    size_t slot = token_hash(token, vocab->hash_capacity);
    for (size_t probe = 0; probe < vocab->hash_capacity; probe++)
    {
        size_t entry_index = vocab->hash_slots[slot];

        if (entry_index == VOCABULARY_NOT_FOUND)
        {
            return VOCABULARY_NOT_FOUND;
        }
        if (entry_index < vocab->size &&
            strcmp(vocab->entries[entry_index].token, token) == 0)
        {
            return entry_index;
        }
        slot = (slot + 1) % vocab->hash_capacity;
    }
    return VOCABULARY_NOT_FOUND;
}

static Status rebuild_hash(Vocabulary *vocab)
{
    for (size_t slot = 0; slot < vocab->hash_capacity; slot++)
    {
        vocab->hash_slots[slot] = VOCABULARY_NOT_FOUND;
    }

    for (size_t entry_index = 0;
         entry_index < vocab->size;
         entry_index++)
    {
        size_t slot = token_hash(
            vocab->entries[entry_index].token,
            vocab->hash_capacity);
        size_t probe_count = 0;

        while (vocab->hash_slots[slot] != VOCABULARY_NOT_FOUND &&
               probe_count < vocab->hash_capacity)
        {
            slot = (slot + 1) % vocab->hash_capacity;
            probe_count++;
        }
        if (probe_count >= vocab->hash_capacity)
        {
            return STATUS_OUT_OF_MEMORY;
        }
        vocab->hash_slots[slot] = entry_index;
    }
    return STATUS_OK;
}

static Status grow_entries(Vocabulary *vocab)
{
    size_t new_capacity = vocab->capacity * 2;

    if (new_capacity <= vocab->capacity ||
        new_capacity > SIZE_MAX / sizeof(*vocab->entries))
    {
        return STATUS_OUT_OF_MEMORY;
    }

    VocabularyEntry *resized_entries = realloc(
        vocab->entries,
        new_capacity * sizeof(*resized_entries));
    if (resized_entries == NULL)
    {
        return STATUS_OUT_OF_MEMORY;
    }

    memset(
        resized_entries + vocab->capacity,
        0,
        (new_capacity - vocab->capacity) * sizeof(*resized_entries));
    vocab->entries = resized_entries;
    vocab->capacity = new_capacity;
    return STATUS_OK;
}

static Status add_token(
    Vocabulary *vocab,
    const char *token,
    size_t *result_index)
{
    if (vocab->size >= vocab->hash_capacity)
    {
        return STATUS_INVALID_ARGUMENT;
    }

    size_t existing_index = vocab_find(vocab, token);
    if (existing_index != VOCABULARY_NOT_FOUND)
    {
        if (result_index != NULL)
        {
            *result_index = existing_index;
        }
        return STATUS_OK;
    }

    if (vocab->size == vocab->capacity)
    {
        Status grow_status = grow_entries(vocab);
        if (grow_status != STATUS_OK)
        {
            return grow_status;
        }
    }

    char *owned_token = copy_token(token);
    if (owned_token == NULL)
    {
        return STATUS_OUT_OF_MEMORY;
    }

    size_t entry_index = vocab->size;
    vocab->size++;
    vocab->entries[entry_index] = (VocabularyEntry){
        .token = owned_token,
    };

    size_t slot = token_hash(token, vocab->hash_capacity);
    while (vocab->hash_slots[slot] != VOCABULARY_NOT_FOUND)
    {
        slot = (slot + 1) % vocab->hash_capacity;
    }
    vocab->hash_slots[slot] = entry_index;
    if (result_index != NULL)
    {
        *result_index = entry_index;
    }
    return STATUS_OK;
}

static Status prune(Vocabulary *vocab, uint64_t threshold)
{
    size_t output_index = 0;

    for (size_t input_index = 0;
         input_index < vocab->size;
         input_index++)
    {
        VocabularyEntry *entry = &vocab->entries[input_index];
        if (input_index == 0 || entry->count > threshold)
        {
            if (output_index != input_index)
            {
                vocab->entries[output_index] = *entry;
            }
            output_index++;
        }
        else
        {
            free(entry->token);
        }
    }
    vocab->size = output_index;
    return rebuild_hash(vocab);
}

static int compare_entries(const void *left, const void *right)
{
    const VocabularyEntry *left_entry = left;
    const VocabularyEntry *right_entry = right;

    if (left_entry->count < right_entry->count)
    {
        return 1;
    }
    if (left_entry->count > right_entry->count)
    {
        return -1;
    }
    return 0;
}

static Status count_corpus(
    Vocabulary *vocab,
    const Corpus *corpus)
{
    size_t entry_index;
    Status status = add_token(vocab, "</s>", &entry_index);
    FILE *file = NULL;

    if (status == STATUS_OK)
    {
        file = fopen(corpus->path, "rb");
        if (file == NULL)
        {
            status = STATUS_IO_ERROR;
        }
    }

    uint64_t prune_threshold = 1;
    char token[MAX_TOKEN_LENGTH];
    int at_eof = 0;
    while (status == STATUS_OK && !at_eof)
    {
        status = token_read(file, token, &at_eof);
        if (status != STATUS_OK || at_eof)
        {
            break;
        }

        entry_index = vocab_find(vocab, token);
        if (entry_index == VOCABULARY_NOT_FOUND)
        {
            status = add_token(vocab, token, &entry_index);
            int hash_is_crowded =
                vocab->size * 10 >= vocab->hash_capacity * 7;
            if (status == STATUS_INVALID_ARGUMENT && hash_is_crowded)
            {
                status = prune(vocab, prune_threshold);
                prune_threshold++;
                if (status == STATUS_OK)
                {
                    status = add_token(vocab, token, &entry_index);
                }
            }
        }
        if (status == STATUS_OK)
        {
            vocab->entries[entry_index].count++;
        }
    }

    if (file != NULL && fclose(file) != 0 && status == STATUS_OK)
    {
        status = STATUS_IO_ERROR;
    }
    return status;
}

static void apply_minimum_count(
    Vocabulary *vocab,
    uint64_t minimum_count)
{
    size_t output_index = 1;

    for (size_t input_index = 1;
         input_index < vocab->size;
         input_index++)
    {
        VocabularyEntry *entry = &vocab->entries[input_index];
        if (entry->count >= minimum_count)
        {
            vocab->entries[output_index] = *entry;
            output_index++;
        }
        else
        {
            free(entry->token);
        }
    }
    vocab->size = output_index;
}

Vocabulary *vocab_build(
    const Corpus *corpus,
    const VocabularyConfig *config,
    Status *status)
{
    if (corpus == NULL || corpus->path == NULL ||
        vocab_config_validate(config) != STATUS_OK)
    {
        if (status != NULL)
        {
            *status = STATUS_INVALID_ARGUMENT;
        }
        return NULL;
    }

    Vocabulary *vocab = calloc(1, sizeof(*vocab));
    if (vocab == NULL)
    {
        if (status != NULL)
        {
            *status = STATUS_OUT_OF_MEMORY;
        }
        return NULL;
    }

    vocab->capacity = config->initial_capacity;
    vocab->hash_capacity = config->hash_capacity;
    vocab->entries = calloc(
        vocab->capacity,
        sizeof(*vocab->entries));
    vocab->hash_slots = malloc(
        vocab->hash_capacity * sizeof(*vocab->hash_slots));

    Status result = STATUS_OK;
    if (vocab->entries == NULL || vocab->hash_slots == NULL)
    {
        result = STATUS_OUT_OF_MEMORY;
    }
    else
    {
        result = rebuild_hash(vocab);
    }
    if (result == STATUS_OK)
    {
        result = count_corpus(vocab, corpus);
    }
    if (result == STATUS_OK && vocab->size > 1)
    {
        qsort(
            vocab->entries + 1,
            vocab->size - 1,
            sizeof(*vocab->entries),
            compare_entries);
    }
    if (result == STATUS_OK)
    {
        apply_minimum_count(vocab, config->min_count);
        vocab->retained_token_count = 0;
        for (size_t index = 0; index < vocab->size; index++)
        {
            vocab->retained_token_count +=
                vocab->entries[index].count;
        }
        result = rebuild_hash(vocab);
    }
    if (result == STATUS_OK)
    {
        result = huffman_assign(vocab);
    }
    if (result != STATUS_OK)
    {
        vocab_destroy(&vocab);
    }
    if (status != NULL)
    {
        *status = result;
    }
    return vocab;
}

void vocab_destroy(Vocabulary **vocab)
{
    if (vocab == NULL || *vocab == NULL)
    {
        return;
    }

    for (size_t index = 0; index < (*vocab)->size; index++)
    {
        free((*vocab)->entries[index].token);
        free((*vocab)->entries[index].huffman_path);
        free((*vocab)->entries[index].huffman_bits);
    }
    free((*vocab)->entries);
    free((*vocab)->hash_slots);
    free(*vocab);
    *vocab = NULL;
}

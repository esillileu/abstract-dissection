#ifndef W2V_VOCABULARY_H
#define W2V_VOCABULARY_H

#include "config.h"
#include "corpus.h"

#define VOCABULARY_NOT_FOUND SIZE_MAX

typedef struct
{
    char *token;
    uint64_t count;
    size_t *huffman_path;
    unsigned char *huffman_bits;
    size_t huffman_length;
} VocabularyEntry;

typedef struct
{
    VocabularyEntry *entries;
    size_t size;
    size_t capacity;
    size_t *hash_slots;
    size_t hash_capacity;
    uint64_t retained_token_count;
} Vocabulary;

typedef struct
{
    size_t *table;
    size_t size;
} NegativeSampler;

Vocabulary *vocab_build(
    const Corpus *corpus,
    const VocabularyConfig *config,
    Status *status);
void vocab_destroy(Vocabulary **vocab);
size_t vocab_find(const Vocabulary *vocab, const char *token);

#endif

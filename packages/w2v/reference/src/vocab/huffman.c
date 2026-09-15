#include "huffman.h"

#include <stdlib.h>

static void free_tree_arrays(
    uint64_t *counts,
    size_t *parents,
    unsigned char *bits)
{
    free(counts);
    free(parents);
    free(bits);
}

Status huffman_assign(Vocabulary *vocab)
{
    if (vocab == NULL || vocab->size == 0)
    {
        return STATUS_INVALID_ARGUMENT;
    }
    if (vocab->size == 1)
    {
        return STATUS_OK;
    }

    size_t leaf_count = vocab->size;
    size_t node_count = 2 * leaf_count - 1;
    uint64_t *counts = malloc(node_count * sizeof(*counts));
    size_t *parents = malloc(node_count * sizeof(*parents));
    unsigned char *bits = calloc(node_count, sizeof(*bits));

    if (counts == NULL || parents == NULL || bits == NULL)
    {
        free_tree_arrays(counts, parents, bits);
        return STATUS_OUT_OF_MEMORY;
    }

    for (size_t index = 0; index < leaf_count; index++)
    {
        counts[index] = vocab->entries[index].count;
    }
    for (size_t index = leaf_count; index < node_count; index++)
    {
        counts[index] = UINT64_C(1000000000000000);
    }

    size_t next_leaf = leaf_count;
    size_t next_internal = leaf_count;
    for (size_t node = leaf_count; node < node_count; node++)
    {
        size_t chosen[2];

        for (size_t child = 0; child < 2; child++)
        {
            int choose_leaf =
                next_leaf > 0 &&
                (next_internal >= node ||
                 counts[next_leaf - 1] < counts[next_internal]);
            if (choose_leaf)
            {
                next_leaf--;
                chosen[child] = next_leaf;
            }
            else
            {
                chosen[child] = next_internal;
                next_internal++;
            }
        }

        counts[node] = counts[chosen[0]] + counts[chosen[1]];
        parents[chosen[0]] = node;
        parents[chosen[1]] = node;
        bits[chosen[1]] = 1;
    }

    for (size_t entry_index = 0;
         entry_index < leaf_count;
         entry_index++)
    {
        size_t reverse_path[MAX_CODE_LENGTH];
        unsigned char reverse_bits[MAX_CODE_LENGTH];
        size_t path_length = 0;
        size_t node = entry_index;

        while (node != node_count - 1)
        {
            if (path_length == MAX_CODE_LENGTH)
            {
                free_tree_arrays(counts, parents, bits);
                return STATUS_CORRUPT_DATA;
            }
            reverse_path[path_length] = parents[node] - leaf_count;
            reverse_bits[path_length] = bits[node];
            path_length++;
            node = parents[node];
        }

        VocabularyEntry *entry = &vocab->entries[entry_index];
        entry->huffman_path = malloc(
            path_length * sizeof(*entry->huffman_path));
        entry->huffman_bits = malloc(
            path_length * sizeof(*entry->huffman_bits));
        if (entry->huffman_path == NULL || entry->huffman_bits == NULL)
        {
            free_tree_arrays(counts, parents, bits);
            return STATUS_OUT_OF_MEMORY;
        }

        entry->huffman_length = path_length;
        for (size_t path_index = 0;
             path_index < path_length;
             path_index++)
        {
            size_t reverse_index = path_length - path_index - 1;

            entry->huffman_path[path_index] = reverse_path[reverse_index];
            entry->huffman_bits[path_index] = reverse_bits[reverse_index];
        }
    }

    free_tree_arrays(counts, parents, bits);
    return STATUS_OK;
}

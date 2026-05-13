/*
 * Memory management utilities
 */

#include "core/mem.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

gxmpp_arena_t *gxmpp_arena_create(size_t capacity)
{
    gxmpp_arena_t *arena = malloc(sizeof(gxmpp_arena_t));
    if (!arena) return NULL;

    arena->base = gxmpp_aligned_alloc(64, capacity);
    if (!arena->base) {
        free(arena);
        return NULL;
    }

    arena->capacity = capacity;
    arena->used = 0;
    return arena;
}

void gxmpp_arena_destroy(gxmpp_arena_t *arena)
{
    if (!arena) return;
    gxmpp_aligned_free(arena->base);
    free(arena);
}

void *gxmpp_arena_alloc(gxmpp_arena_t *arena, size_t size)
{
    /* Align all allocations to 64 bytes for SIMD */
    size_t aligned = (size + 63) & ~(size_t)63;

    if (arena->used + aligned > arena->capacity) {
        fprintf(stderr, "arena: out of memory (need %zu, have %zu)\n",
                aligned, arena->capacity - arena->used);
        return NULL;
    }

    void *ptr = (char *)arena->base + arena->used;
    arena->used += aligned;
    return ptr;
}

void gxmpp_arena_reset(gxmpp_arena_t *arena)
{
    arena->used = 0;
}

void *gxmpp_aligned_alloc(size_t alignment, size_t size)
{
    void *ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0)
        return NULL;
    return ptr;
}

void gxmpp_aligned_free(void *ptr)
{
    free(ptr);
}

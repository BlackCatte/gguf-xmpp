/*
 * Memory management utilities
 */

#ifndef GXMPP_CORE_MEM_H
#define GXMPP_CORE_MEM_H

#include <stddef.h>

/* Arena allocator for per-inference scratch memory */
typedef struct {
    void   *base;
    size_t  capacity;
    size_t  used;
} gxmpp_arena_t;

gxmpp_arena_t *gxmpp_arena_create(size_t capacity);
void           gxmpp_arena_destroy(gxmpp_arena_t *arena);
void          *gxmpp_arena_alloc(gxmpp_arena_t *arena, size_t size);
void           gxmpp_arena_reset(gxmpp_arena_t *arena);

/* Aligned allocation for SIMD tensor ops */
void *gxmpp_aligned_alloc(size_t alignment, size_t size);
void  gxmpp_aligned_free(void *ptr);

#endif /* GXMPP_CORE_MEM_H */

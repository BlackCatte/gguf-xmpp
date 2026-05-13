/*
 * GGUF file format parser
 *
 * Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
 *
 * File layout:
 *   [header]
 *   [metadata key-value pairs]
 *   [tensor info entries]
 *   [padding to alignment]
 *   [tensor data]
 */

#ifndef GXMPP_GGUF_H
#define GXMPP_GGUF_H

#include "core/types.h"

#define GGUF_MAGIC       0x46475547  /* "GGUF" in little-endian */
#define GGUF_VERSION_MIN 2
#define GGUF_VERSION_MAX 3
#define GGUF_DEFAULT_ALIGNMENT 32

/* Parsed GGUF file context */
typedef struct {
    uint32_t         version;
    uint64_t         n_tensors;
    uint64_t         n_kv;
    uint32_t         alignment;

    gxmpp_meta_kv_t *kv;         /* metadata array [n_kv] */
    gxmpp_tensor_t  *tensors;    /* tensor descriptors [n_tensors] */

    int              fd;          /* file descriptor for mmap */
    void            *mmap_base;   /* mmap base pointer */
    size_t           mmap_len;    /* mmap total length */
    uint64_t         data_offset; /* offset of tensor data section */
} gguf_ctx_t;

/* Open and parse a GGUF file. Tensors are memory-mapped. */
gxmpp_result_t gguf_open(const char *path, gguf_ctx_t **ctx);

/* Close and free all resources */
void gguf_close(gguf_ctx_t *ctx);

/* Look up metadata by key. Returns NULL if not found. */
const gxmpp_meta_kv_t *gguf_meta_find(const gguf_ctx_t *ctx, const char *key);

/* Convenience: get string metadata value, or default if missing */
const char *gguf_meta_str(const gguf_ctx_t *ctx, const char *key,
                          const char *fallback);

/* Convenience: get uint32 metadata value, or default if missing */
uint32_t gguf_meta_u32(const gguf_ctx_t *ctx, const char *key,
                       uint32_t fallback);

/* Convenience: get float32 metadata value, or default if missing */
float gguf_meta_f32(const gguf_ctx_t *ctx, const char *key, float fallback);

/* Look up tensor by name. Returns NULL if not found. */
const gxmpp_tensor_t *gguf_tensor_find(const gguf_ctx_t *ctx, const char *name);

/* Get pointer to tensor data (via mmap). Returns NULL on error. */
const void *gguf_tensor_data(const gguf_ctx_t *ctx, const gxmpp_tensor_t *t);

/* Compute byte size for a quantized block */
size_t gguf_type_block_size(gguf_quant_type_t type);
size_t gguf_type_type_size(gguf_quant_type_t type);

#endif /* GXMPP_GGUF_H */

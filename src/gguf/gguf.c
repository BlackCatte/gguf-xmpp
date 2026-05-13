/*
 * GGUF binary format parser
 *
 * Parses the GGUF header, metadata key-value pairs, and tensor info.
 * Tensor data is memory-mapped for zero-copy access.
 */

#include "gguf/gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

/* Block sizes and type sizes for quantization formats */
static const struct {
    size_t block_size;
    size_t type_size;
} quant_info[GGUF_TYPE_COUNT] = {
    [GGUF_TYPE_F32]     = { 1,  4 },
    [GGUF_TYPE_F16]     = { 1,  2 },
    [GGUF_TYPE_Q4_0]    = { 32, 18 },
    [GGUF_TYPE_Q4_1]    = { 32, 20 },
    [GGUF_TYPE_Q5_0]    = { 32, 22 },
    [GGUF_TYPE_Q5_1]    = { 32, 24 },
    [GGUF_TYPE_Q8_0]    = { 32, 34 },
    [GGUF_TYPE_Q8_1]    = { 32, 36 },
    [GGUF_TYPE_Q2_K]    = { 256, 84 },
    [GGUF_TYPE_Q3_K]    = { 256, 110 },
    [GGUF_TYPE_Q4_K]    = { 256, 144 },
    [GGUF_TYPE_Q5_K]    = { 256, 176 },
    [GGUF_TYPE_Q6_K]    = { 256, 210 },
    [GGUF_TYPE_IQ2_XXS] = { 256, 66 },
    [GGUF_TYPE_IQ2_XS]  = { 256, 74 },
    [GGUF_TYPE_IQ3_XXS] = { 256, 98 },
    [GGUF_TYPE_IQ1_S]   = { 256, 50 },
    [GGUF_TYPE_IQ4_NL]  = { 32, 18 },
    [GGUF_TYPE_IQ3_S]   = { 256, 110 },
    [GGUF_TYPE_IQ2_S]   = { 256, 82 },
    [GGUF_TYPE_IQ4_XS]  = { 256, 136 },
};

size_t gguf_type_block_size(gguf_quant_type_t type)
{
    if (type >= GGUF_TYPE_COUNT) return 0;
    return quant_info[type].block_size;
}

size_t gguf_type_type_size(gguf_quant_type_t type)
{
    if (type >= GGUF_TYPE_COUNT) return 0;
    return quant_info[type].type_size;
}

/* Read helpers — all little-endian */

typedef struct {
    const uint8_t *data;
    size_t         len;
    size_t         pos;
} reader_t;

static bool read_bytes(reader_t *r, void *dst, size_t n)
{
    if (r->pos + n > r->len) return false;
    memcpy(dst, r->data + r->pos, n);
    r->pos += n;
    return true;
}

static bool read_u8(reader_t *r, uint8_t *v)   { return read_bytes(r, v, 1); }
static bool read_i8(reader_t *r, int8_t *v)     { return read_bytes(r, v, 1); }
static bool read_u16(reader_t *r, uint16_t *v)  { return read_bytes(r, v, 2); }
static bool read_i16(reader_t *r, int16_t *v)   { return read_bytes(r, v, 2); }
static bool read_u32(reader_t *r, uint32_t *v)  { return read_bytes(r, v, 4); }
static bool read_i32(reader_t *r, int32_t *v)   { return read_bytes(r, v, 4); }
static bool read_u64(reader_t *r, uint64_t *v)  { return read_bytes(r, v, 8); }
static bool read_i64(reader_t *r, int64_t *v)   { return read_bytes(r, v, 8); }
static bool read_f32(reader_t *r, float *v)     { return read_bytes(r, v, 4); }
static bool read_f64(reader_t *r, double *v)    { return read_bytes(r, v, 8); }

static bool read_str(reader_t *r, char **out, uint64_t *out_len)
{
    uint64_t len;
    if (!read_u64(r, &len)) return false;
    if (r->pos + len > r->len) return false;

    char *s = malloc(len + 1);
    if (!s) return false;
    memcpy(s, r->data + r->pos, len);
    s[len] = '\0';
    r->pos += len;

    *out = s;
    if (out_len) *out_len = len;
    return true;
}

static bool read_meta_value(reader_t *r, gguf_meta_type_t type,
                            gxmpp_meta_kv_t *kv);

static bool read_meta_array(reader_t *r, gxmpp_meta_kv_t *kv)
{
    uint32_t elem_type;
    uint64_t count;
    if (!read_u32(r, &elem_type)) return false;
    if (!read_u64(r, &count)) return false;

    kv->value.arr.elem_type = (gguf_meta_type_t)elem_type;
    kv->value.arr.count = count;

    if (count == 0) {
        kv->value.arr.data = NULL;
        return true;
    }

    if (elem_type == GGUF_META_STRING) {
        char **strings = calloc(count, sizeof(char *));
        if (!strings) return false;
        for (uint64_t i = 0; i < count; i++) {
            uint64_t slen;
            if (!read_str(r, &strings[i], &slen)) {
                for (uint64_t j = 0; j < i; j++) free(strings[j]);
                free(strings);
                return false;
            }
        }
        kv->value.arr.data = strings;
    } else {
        /* For non-string arrays, store as raw gxmpp_meta_kv_t array */
        gxmpp_meta_kv_t *elems = calloc(count, sizeof(gxmpp_meta_kv_t));
        if (!elems) return false;
        for (uint64_t i = 0; i < count; i++) {
            elems[i].type = (gguf_meta_type_t)elem_type;
            if (!read_meta_value(r, (gguf_meta_type_t)elem_type, &elems[i])) {
                free(elems);
                return false;
            }
        }
        kv->value.arr.data = elems;
    }
    return true;
}

static bool read_meta_value(reader_t *r, gguf_meta_type_t type,
                            gxmpp_meta_kv_t *kv)
{
    kv->type = type;
    switch (type) {
    case GGUF_META_UINT8:   return read_u8(r, &kv->value.u8);
    case GGUF_META_INT8:    return read_i8(r, &kv->value.i8);
    case GGUF_META_UINT16:  return read_u16(r, &kv->value.u16);
    case GGUF_META_INT16:   return read_i16(r, &kv->value.i16);
    case GGUF_META_UINT32:  return read_u32(r, &kv->value.u32);
    case GGUF_META_INT32:   return read_i32(r, &kv->value.i32);
    case GGUF_META_UINT64:  return read_u64(r, &kv->value.u64);
    case GGUF_META_INT64:   return read_i64(r, &kv->value.i64);
    case GGUF_META_FLOAT32: return read_f32(r, &kv->value.f32);
    case GGUF_META_FLOAT64: return read_f64(r, &kv->value.f64);
    case GGUF_META_BOOL: {
        uint8_t b;
        if (!read_u8(r, &b)) return false;
        kv->value.b = (b != 0);
        return true;
    }
    case GGUF_META_STRING:
        return read_str(r, &kv->value.str.data, &kv->value.str.len);
    case GGUF_META_ARRAY:
        return read_meta_array(r, kv);
    default:
        return false;
    }
}

static uint64_t compute_tensor_size(const gxmpp_tensor_t *t)
{
    uint64_t n_elems = 1;
    for (uint32_t d = 0; d < t->n_dims; d++)
        n_elems *= t->dims[d];

    size_t bs = gguf_type_block_size(t->type);
    size_t ts = gguf_type_type_size(t->type);
    if (bs == 0) return 0;

    return (n_elems / bs) * ts;
}

gxmpp_result_t gguf_open(const char *path, gguf_ctx_t **out)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        perror("gguf_open");
        return GXMPP_ERR_IO;
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        return GXMPP_ERR_IO;
    }

    size_t file_size = (size_t)st.st_size;
    void *mapped = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
        close(fd);
        return GXMPP_ERR_IO;
    }

    reader_t r = { .data = mapped, .len = file_size, .pos = 0 };

    /* Read header */
    uint32_t magic;
    if (!read_u32(&r, &magic) || magic != GGUF_MAGIC) {
        fprintf(stderr, "gguf: invalid magic: 0x%08x\n", magic);
        munmap(mapped, file_size);
        close(fd);
        return GXMPP_ERR_FORMAT;
    }

    uint32_t version;
    if (!read_u32(&r, &version)) goto err;
    if (version < GGUF_VERSION_MIN || version > GGUF_VERSION_MAX) {
        fprintf(stderr, "gguf: unsupported version: %u\n", version);
        goto err;
    }

    uint64_t n_tensors, n_kv;
    if (!read_u64(&r, &n_tensors)) goto err;
    if (!read_u64(&r, &n_kv)) goto err;

    gguf_ctx_t *ctx = calloc(1, sizeof(gguf_ctx_t));
    if (!ctx) goto err;

    ctx->version = version;
    ctx->n_tensors = n_tensors;
    ctx->n_kv = n_kv;
    ctx->fd = fd;
    ctx->mmap_base = mapped;
    ctx->mmap_len = file_size;
    ctx->alignment = GGUF_DEFAULT_ALIGNMENT;

    /* Read metadata KV pairs */
    ctx->kv = calloc(n_kv, sizeof(gxmpp_meta_kv_t));
    if (!ctx->kv) { free(ctx); goto err; }

    for (uint64_t i = 0; i < n_kv; i++) {
        uint64_t key_len;
        if (!read_str(&r, &ctx->kv[i].key, &key_len)) {
            fprintf(stderr, "gguf: failed to read metadata key %lu\n",
                    (unsigned long)i);
            goto err_ctx;
        }

        uint32_t vtype;
        if (!read_u32(&r, &vtype)) goto err_ctx;

        if (!read_meta_value(&r, (gguf_meta_type_t)vtype, &ctx->kv[i])) {
            fprintf(stderr, "gguf: failed to read metadata value for '%s'\n",
                    ctx->kv[i].key);
            goto err_ctx;
        }

        /* Check for custom alignment */
        if (strcmp(ctx->kv[i].key, "general.alignment") == 0 &&
            ctx->kv[i].type == GGUF_META_UINT32) {
            ctx->alignment = ctx->kv[i].value.u32;
        }
    }

    /* Read tensor info */
    ctx->tensors = calloc(n_tensors, sizeof(gxmpp_tensor_t));
    if (!ctx->tensors) goto err_ctx;

    for (uint64_t i = 0; i < n_tensors; i++) {
        gxmpp_tensor_t *t = &ctx->tensors[i];
        uint64_t name_len;
        if (!read_str(&r, &t->name, &name_len)) goto err_ctx;
        if (!read_u32(&r, &t->n_dims)) goto err_ctx;

        memset(t->dims, 0, sizeof(t->dims));
        for (uint32_t d = 0; d < t->n_dims && d < 4; d++) {
            if (!read_u64(&r, &t->dims[d])) goto err_ctx;
        }

        uint32_t qtype;
        if (!read_u32(&r, &qtype)) goto err_ctx;
        t->type = (gguf_quant_type_t)qtype;

        if (!read_u64(&r, &t->offset)) goto err_ctx;
        t->size = compute_tensor_size(t);
    }

    /* Compute data section offset (aligned) */
    uint64_t data_offset = r.pos;
    data_offset = (data_offset + ctx->alignment - 1) & ~((uint64_t)ctx->alignment - 1);
    ctx->data_offset = data_offset;

    /* Set tensor data pointers via mmap */
    for (uint64_t i = 0; i < n_tensors; i++) {
        gxmpp_tensor_t *t = &ctx->tensors[i];
        uint64_t abs_offset = data_offset + t->offset;
        if (abs_offset + t->size > file_size) {
            fprintf(stderr, "gguf: tensor '%s' extends beyond file\n", t->name);
            goto err_ctx;
        }
        t->data = (uint8_t *)mapped + abs_offset;
    }

    *out = ctx;
    return GXMPP_OK;

err_ctx:
    gguf_close(ctx);
    return GXMPP_ERR_FORMAT;
err:
    munmap(mapped, file_size);
    close(fd);
    return GXMPP_ERR_FORMAT;
}

void gguf_close(gguf_ctx_t *ctx)
{
    if (!ctx) return;

    if (ctx->kv) {
        for (uint64_t i = 0; i < ctx->n_kv; i++) {
            free(ctx->kv[i].key);
            if (ctx->kv[i].type == GGUF_META_STRING)
                free(ctx->kv[i].value.str.data);
            if (ctx->kv[i].type == GGUF_META_ARRAY) {
                if (ctx->kv[i].value.arr.elem_type == GGUF_META_STRING) {
                    char **strs = ctx->kv[i].value.arr.data;
                    for (uint64_t j = 0; j < ctx->kv[i].value.arr.count; j++)
                        free(strs[j]);
                }
                free(ctx->kv[i].value.arr.data);
            }
        }
        free(ctx->kv);
    }

    if (ctx->tensors) {
        for (uint64_t i = 0; i < ctx->n_tensors; i++)
            free(ctx->tensors[i].name);
        free(ctx->tensors);
    }

    if (ctx->mmap_base && ctx->mmap_base != MAP_FAILED)
        munmap(ctx->mmap_base, ctx->mmap_len);

    if (ctx->fd >= 0)
        close(ctx->fd);

    free(ctx);
}

const gxmpp_meta_kv_t *gguf_meta_find(const gguf_ctx_t *ctx, const char *key)
{
    for (uint64_t i = 0; i < ctx->n_kv; i++) {
        if (strcmp(ctx->kv[i].key, key) == 0)
            return &ctx->kv[i];
    }
    return NULL;
}

const char *gguf_meta_str(const gguf_ctx_t *ctx, const char *key,
                          const char *fallback)
{
    const gxmpp_meta_kv_t *kv = gguf_meta_find(ctx, key);
    if (kv && kv->type == GGUF_META_STRING)
        return kv->value.str.data;
    return fallback;
}

uint32_t gguf_meta_u32(const gguf_ctx_t *ctx, const char *key,
                       uint32_t fallback)
{
    const gxmpp_meta_kv_t *kv = gguf_meta_find(ctx, key);
    if (kv && kv->type == GGUF_META_UINT32)
        return kv->value.u32;
    return fallback;
}

float gguf_meta_f32(const gguf_ctx_t *ctx, const char *key, float fallback)
{
    const gxmpp_meta_kv_t *kv = gguf_meta_find(ctx, key);
    if (kv && kv->type == GGUF_META_FLOAT32)
        return kv->value.f32;
    return fallback;
}

const gxmpp_tensor_t *gguf_tensor_find(const gguf_ctx_t *ctx, const char *name)
{
    for (uint64_t i = 0; i < ctx->n_tensors; i++) {
        if (strcmp(ctx->tensors[i].name, name) == 0)
            return &ctx->tensors[i];
    }
    return NULL;
}

const void *gguf_tensor_data(const gguf_ctx_t *ctx, const gxmpp_tensor_t *t)
{
    (void)ctx;
    if (!t || !t->data) return NULL;
    return t->data;
}

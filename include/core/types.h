/*
 * gguf-xmpp: Lightweight GGUF inference over XMPP
 * Core type definitions
 */

#ifndef GXMPP_CORE_TYPES_H
#define GXMPP_CORE_TYPES_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/* Quantization types supported by GGUF */
typedef enum {
    GGUF_TYPE_F32     = 0,
    GGUF_TYPE_F16     = 1,
    GGUF_TYPE_Q4_0    = 2,
    GGUF_TYPE_Q4_1    = 3,
    GGUF_TYPE_Q5_0    = 6,
    GGUF_TYPE_Q5_1    = 7,
    GGUF_TYPE_Q8_0    = 8,
    GGUF_TYPE_Q8_1    = 9,
    GGUF_TYPE_Q2_K    = 10,
    GGUF_TYPE_Q3_K    = 11,
    GGUF_TYPE_Q4_K    = 12,
    GGUF_TYPE_Q5_K    = 13,
    GGUF_TYPE_Q6_K    = 14,
    GGUF_TYPE_IQ2_XXS = 15,
    GGUF_TYPE_IQ2_XS  = 16,
    GGUF_TYPE_IQ3_XXS = 17,
    GGUF_TYPE_IQ1_S   = 18,
    GGUF_TYPE_IQ4_NL  = 19,
    GGUF_TYPE_IQ3_S   = 20,
    GGUF_TYPE_IQ2_S   = 21,
    GGUF_TYPE_IQ4_XS  = 22,
    GGUF_TYPE_COUNT
} gguf_quant_type_t;

/* GGUF metadata value types */
typedef enum {
    GGUF_META_UINT8   = 0,
    GGUF_META_INT8    = 1,
    GGUF_META_UINT16  = 2,
    GGUF_META_INT16   = 3,
    GGUF_META_UINT32  = 4,
    GGUF_META_INT32   = 5,
    GGUF_META_FLOAT32 = 6,
    GGUF_META_BOOL    = 7,
    GGUF_META_STRING  = 8,
    GGUF_META_ARRAY   = 9,
    GGUF_META_UINT64  = 10,
    GGUF_META_INT64   = 11,
    GGUF_META_FLOAT64 = 12,
} gguf_meta_type_t;

/* Tensor descriptor */
typedef struct {
    char        *name;
    uint32_t     n_dims;
    uint64_t     dims[4];
    gguf_quant_type_t type;
    uint64_t     offset;    /* offset within data section */
    size_t       size;      /* size in bytes */
    void        *data;      /* pointer to loaded data (mmap or malloc) */
} gxmpp_tensor_t;

/* Metadata key-value pair */
typedef struct {
    char            *key;
    gguf_meta_type_t type;
    union {
        uint8_t     u8;
        int8_t      i8;
        uint16_t    u16;
        int16_t     i16;
        uint32_t    u32;
        int32_t     i32;
        uint64_t    u64;
        int64_t     i64;
        float       f32;
        double      f64;
        bool        b;
        struct {
            char    *data;
            uint64_t len;
        } str;
        struct {
            gguf_meta_type_t elem_type;
            uint64_t         count;
            void            *data;
        } arr;
    } value;
} gxmpp_meta_kv_t;

/* Result codes */
typedef enum {
    GXMPP_OK = 0,
    GXMPP_ERR_IO,
    GXMPP_ERR_FORMAT,
    GXMPP_ERR_MEMORY,
    GXMPP_ERR_UNSUPPORTED,
    GXMPP_ERR_NET,
    GXMPP_ERR_XML,
    GXMPP_ERR_TLS,
    GXMPP_ERR_AUTH,
    GXMPP_ERR_PARAM,
} gxmpp_result_t;

/* Callback for inference token streaming */
typedef void (*gxmpp_token_cb)(const char *token, void *userdata);

/* Callback for incoming XMPP messages */
typedef void (*gxmpp_msg_cb)(const char *from, const char *body,
                             const void *media, size_t media_len,
                             void *userdata);

/* Callback for voice transcription results */
typedef void (*gxmpp_voice_cb)(const char *from, const char *transcript,
                               void *userdata);

#endif /* GXMPP_CORE_TYPES_H */

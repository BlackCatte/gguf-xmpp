/*
 * Vision pipeline
 *
 * Handles image encoding for multimodal GGUF models.
 * Implements a minimal CLIP-style vision encoder that reads
 * image patches and produces embedding vectors for the LLM.
 */

#ifndef GXMPP_VISION_H
#define GXMPP_VISION_H

#include "core/types.h"
#include "gguf/gguf.h"

/* Supported image formats (decoded in-house, no external libs) */
typedef enum {
    GXMPP_IMG_PNG,
    GXMPP_IMG_JPEG,
    GXMPP_IMG_BMP,
    GXMPP_IMG_UNKNOWN,
} gxmpp_img_format_t;

/* Raw decoded image */
typedef struct {
    uint8_t  *pixels;     /* RGB, row-major */
    uint32_t  width;
    uint32_t  height;
    uint32_t  channels;   /* 3 = RGB */
} gxmpp_image_t;

/* Vision encoder context */
typedef struct {
    gguf_ctx_t  *gguf;       /* vision model GGUF (may be same file) */
    uint32_t     patch_size;
    uint32_t     image_size;  /* expected input resolution */
    uint32_t     n_embd;      /* vision embedding dimension */
    uint32_t     n_layers;
    uint32_t     n_heads;
    float       *proj;        /* projection matrix to LLM space */
    uint32_t     proj_dim;    /* LLM embedding dimension */
} gxmpp_vision_ctx_t;

/* Initialize vision encoder from a GGUF context.
 * The GGUF may be a standalone vision model or a combined LLaVA-style file. */
gxmpp_result_t gxmpp_vision_init(const gguf_ctx_t *gguf,
                                 gxmpp_vision_ctx_t **ctx);

/* Free vision context */
void gxmpp_vision_free(gxmpp_vision_ctx_t *ctx);

/* Detect image format from raw bytes */
gxmpp_img_format_t gxmpp_image_detect(const void *data, size_t len);

/* Decode image from raw bytes to pixel buffer */
gxmpp_result_t gxmpp_image_decode(const void *data, size_t len,
                                  gxmpp_image_t **img);

/* Free decoded image */
void gxmpp_image_free(gxmpp_image_t *img);

/* Encode image to embeddings suitable for LLM input.
 * Returns float array [n_patches * proj_dim]. Caller must free. */
float *gxmpp_vision_encode(gxmpp_vision_ctx_t *ctx,
                           const gxmpp_image_t *img,
                           int *n_patches);

#endif /* GXMPP_VISION_H */

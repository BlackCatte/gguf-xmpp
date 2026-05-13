/*
 * Vision pipeline
 *
 * Minimal image decoding (BMP + simple PNG/JPEG detection)
 * and CLIP-style vision encoder for multimodal GGUF models.
 */

#include "vision/vision.h"
#include "core/mem.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern void dequantize(const void *src, float *dst, int n,
                       gguf_quant_type_t type);

/* ---- Image format detection ---- */

gxmpp_img_format_t gxmpp_image_detect(const void *data, size_t len)
{
    if (len < 4) return GXMPP_IMG_UNKNOWN;
    const uint8_t *p = data;

    if (p[0] == 0x89 && p[1] == 'P' && p[2] == 'N' && p[3] == 'G')
        return GXMPP_IMG_PNG;
    if (p[0] == 0xFF && p[1] == 0xD8)
        return GXMPP_IMG_JPEG;
    if (p[0] == 'B' && p[1] == 'M')
        return GXMPP_IMG_BMP;

    return GXMPP_IMG_UNKNOWN;
}

/* ---- BMP decoder (24-bit uncompressed) ---- */

static gxmpp_result_t decode_bmp(const void *data, size_t len,
                                 gxmpp_image_t **out)
{
    const uint8_t *p = data;
    if (len < 54) return GXMPP_ERR_FORMAT;

    uint32_t offset = *(uint32_t *)(p + 10);
    int32_t  width  = *(int32_t  *)(p + 18);
    int32_t  height = *(int32_t  *)(p + 22);
    uint16_t bpp    = *(uint16_t *)(p + 28);

    if (width <= 0 || bpp != 24) return GXMPP_ERR_UNSUPPORTED;
    int flip = (height > 0);
    if (height < 0) height = -height;

    gxmpp_image_t *img = calloc(1, sizeof(gxmpp_image_t));
    if (!img) return GXMPP_ERR_MEMORY;

    img->width = (uint32_t)width;
    img->height = (uint32_t)height;
    img->channels = 3;
    img->pixels = malloc((size_t)width * height * 3);
    if (!img->pixels) { free(img); return GXMPP_ERR_MEMORY; }

    int row_stride = ((width * 3 + 3) / 4) * 4;

    for (int y = 0; y < height; y++) {
        int src_y = flip ? (height - 1 - y) : y;
        const uint8_t *row = p + offset + src_y * row_stride;
        uint8_t *dst = img->pixels + y * width * 3;
        for (int x = 0; x < width; x++) {
            dst[x * 3 + 0] = row[x * 3 + 2]; /* B -> R */
            dst[x * 3 + 1] = row[x * 3 + 1]; /* G */
            dst[x * 3 + 2] = row[x * 3 + 0]; /* R -> B */
        }
    }

    *out = img;
    return GXMPP_OK;
}

gxmpp_result_t gxmpp_image_decode(const void *data, size_t len,
                                  gxmpp_image_t **img)
{
    gxmpp_img_format_t fmt = gxmpp_image_detect(data, len);
    switch (fmt) {
    case GXMPP_IMG_BMP:
        return decode_bmp(data, len, img);
    case GXMPP_IMG_PNG:
    case GXMPP_IMG_JPEG:
        fprintf(stderr, "vision: PNG/JPEG decoding not yet implemented\n");
        return GXMPP_ERR_UNSUPPORTED;
    default:
        return GXMPP_ERR_FORMAT;
    }
}

void gxmpp_image_free(gxmpp_image_t *img)
{
    if (!img) return;
    free(img->pixels);
    free(img);
}

/* ---- Image preprocessing ---- */

static void resize_bilinear(const uint8_t *src, int sw, int sh,
                            float *dst, int dw, int dh)
{
    float sx = (float)sw / dw;
    float sy = (float)sh / dh;

    for (int y = 0; y < dh; y++) {
        for (int x = 0; x < dw; x++) {
            float fx = (x + 0.5f) * sx - 0.5f;
            float fy = (y + 0.5f) * sy - 0.5f;

            int x0 = (int)fx, y0 = (int)fy;
            if (x0 < 0) x0 = 0;
            if (y0 < 0) y0 = 0;
            int x1 = x0 + 1, y1 = y0 + 1;
            if (x1 >= sw) x1 = sw - 1;
            if (y1 >= sh) y1 = sh - 1;

            float dx_frac = fx - x0;
            float dy_frac = fy - y0;

            for (int c = 0; c < 3; c++) {
                float v00 = src[(y0 * sw + x0) * 3 + c] / 255.0f;
                float v01 = src[(y0 * sw + x1) * 3 + c] / 255.0f;
                float v10 = src[(y1 * sw + x0) * 3 + c] / 255.0f;
                float v11 = src[(y1 * sw + x1) * 3 + c] / 255.0f;

                float v = v00 * (1 - dx_frac) * (1 - dy_frac) +
                          v01 * dx_frac * (1 - dy_frac) +
                          v10 * (1 - dx_frac) * dy_frac +
                          v11 * dx_frac * dy_frac;

                dst[(c * dh + y) * dw + x] = v;
            }
        }
    }
}

static void normalize_image(float *img, int n_pixels)
{
    /* ImageNet normalization: (x - mean) / std */
    static const float mean[3] = { 0.48145466f, 0.4578275f, 0.40821073f };
    static const float std[3]  = { 0.26862954f, 0.26130258f, 0.27577711f };

    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < n_pixels; i++) {
            img[c * n_pixels + i] =
                (img[c * n_pixels + i] - mean[c]) / std[c];
        }
    }
}

/* ---- Vision encoder ---- */

gxmpp_result_t gxmpp_vision_init(const gguf_ctx_t *gguf,
                                 gxmpp_vision_ctx_t **out)
{
    gxmpp_vision_ctx_t *ctx = calloc(1, sizeof(gxmpp_vision_ctx_t));
    if (!ctx) return GXMPP_ERR_MEMORY;

    ctx->gguf = (gguf_ctx_t *)gguf;

    /* Try to read vision-specific metadata */
    ctx->patch_size = gguf_meta_u32(gguf, "clip.vision.patch_size", 14);
    ctx->image_size = gguf_meta_u32(gguf, "clip.vision.image_size", 336);
    ctx->n_embd    = gguf_meta_u32(gguf, "clip.vision.embedding_length", 1024);
    ctx->n_layers  = gguf_meta_u32(gguf, "clip.vision.block_count", 24);
    ctx->n_heads   = gguf_meta_u32(gguf, "clip.vision.attention.head_count", 16);

    /* Look for projection matrix */
    const gxmpp_tensor_t *proj_t = gguf_tensor_find(gguf, "mm.0.weight");
    if (proj_t) {
        ctx->proj_dim = (uint32_t)proj_t->dims[0];
        ctx->proj = malloc(proj_t->dims[0] * proj_t->dims[1] * sizeof(float));
        if (ctx->proj) {
            dequantize(proj_t->data, ctx->proj,
                       (int)(proj_t->dims[0] * proj_t->dims[1]),
                       proj_t->type);
        }
    }

    fprintf(stderr, "vision: patch=%u img=%u embd=%u layers=%u\n",
            ctx->patch_size, ctx->image_size, ctx->n_embd, ctx->n_layers);

    *out = ctx;
    return GXMPP_OK;
}

void gxmpp_vision_free(gxmpp_vision_ctx_t *ctx)
{
    if (!ctx) return;
    free(ctx->proj);
    free(ctx);
}

float *gxmpp_vision_encode(gxmpp_vision_ctx_t *ctx,
                           const gxmpp_image_t *img,
                           int *n_patches)
{
    int ps = ctx->patch_size;
    int is = ctx->image_size;
    int np = (is / ps) * (is / ps);

    /* Resize and normalize image */
    float *resized = malloc(3 * is * is * sizeof(float));
    if (!resized) return NULL;

    resize_bilinear(img->pixels, img->width, img->height,
                    resized, is, is);
    normalize_image(resized, is * is);

    /* Extract patch embeddings via convolution with patch embedding weights */
    const gxmpp_tensor_t *patch_embd =
        gguf_tensor_find(ctx->gguf, "v.patch_embd.weight");

    float *embeddings = calloc(np, ctx->n_embd * sizeof(float));
    if (!embeddings) { free(resized); return NULL; }

    if (patch_embd) {
        /* Patch embedding: convolve each patch with the embedding kernel */
        int grid = is / ps;
        float *kernel = malloc(3 * ps * ps * sizeof(float));

        for (int row = 0; row < grid; row++) {
            for (int col = 0; col < grid; col++) {
                int patch_idx = row * grid + col;

                /* For each output dimension */
                for (uint32_t d = 0; d < ctx->n_embd; d++) {
                    /* Extract and dequantize kernel for this output channel */
                    size_t kernel_size = 3 * ps * ps;
                    size_t bs = gguf_type_block_size(patch_embd->type);
                    size_t ts = gguf_type_type_size(patch_embd->type);
                    size_t row_bytes = (kernel_size / bs) * ts;

                    dequantize((uint8_t *)patch_embd->data + d * row_bytes,
                               kernel, (int)kernel_size, patch_embd->type);

                    /* Dot product with image patch */
                    float sum = 0.0f;
                    for (int c = 0; c < 3; c++) {
                        for (int py = 0; py < ps; py++) {
                            for (int px = 0; px < ps; px++) {
                                int iy = row * ps + py;
                                int ix = col * ps + px;
                                float pixel = resized[c * is * is + iy * is + ix];
                                float weight = kernel[c * ps * ps + py * ps + px];
                                sum += pixel * weight;
                            }
                        }
                    }
                    embeddings[patch_idx * ctx->n_embd + d] = sum;
                }
            }
        }
        free(kernel);
    }

    free(resized);
    *n_patches = np;

    /* Project to LLM embedding space if projection matrix exists */
    if (ctx->proj && ctx->proj_dim > 0) {
        float *projected = malloc((size_t)np * ctx->proj_dim * sizeof(float));
        if (projected) {
            for (int p = 0; p < np; p++) {
                for (uint32_t d = 0; d < ctx->proj_dim; d++) {
                    float sum = 0.0f;
                    for (uint32_t k = 0; k < ctx->n_embd; k++) {
                        sum += embeddings[p * ctx->n_embd + k] *
                               ctx->proj[d * ctx->n_embd + k];
                    }
                    projected[p * ctx->proj_dim + d] = sum;
                }
            }
            free(embeddings);
            return projected;
        }
    }

    return embeddings;
}

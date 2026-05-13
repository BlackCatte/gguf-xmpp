/*
 * Dequantization routines
 *
 * Converts quantized tensor blocks to float32 for computation.
 * Each quantization format has a specific block layout that packs
 * weights with scale factors and zero points.
 */

#include "core/types.h"
#include <string.h>
#include <math.h>

/* Q4_0: 32 values per block, 18 bytes (2 bytes scale + 16 bytes quants) */
void dequant_q4_0(const void *src, float *dst, int n)
{
    const uint8_t *p = src;
    int nb = n / 32;

    for (int b = 0; b < nb; b++) {
        float scale;
        uint16_t s16;
        memcpy(&s16, p, 2);

        /* f16 to f32 conversion */
        uint32_t sign = (uint32_t)(s16 >> 15) << 31;
        uint32_t exp  = (s16 >> 10) & 0x1f;
        uint32_t mant = s16 & 0x3ff;
        uint32_t f32;
        if (exp == 0) {
            if (mant == 0) f32 = sign;
            else {
                exp = 1;
                while (!(mant & 0x400)) { mant <<= 1; exp--; }
                mant &= 0x3ff;
                f32 = sign | ((exp + 127 - 15) << 23) | (mant << 13);
            }
        } else if (exp == 31) {
            f32 = sign | 0x7f800000 | (mant << 13);
        } else {
            f32 = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
        memcpy(&scale, &f32, 4);
        p += 2;

        for (int i = 0; i < 16; i++) {
            uint8_t byte = p[i];
            int lo = (byte & 0x0f) - 8;
            int hi = (byte >> 4) - 8;
            dst[b * 32 + i * 2 + 0] = lo * scale;
            dst[b * 32 + i * 2 + 1] = hi * scale;
        }
        p += 16;
    }
}

/* Q4_1: 32 values per block, 20 bytes (2 bytes scale + 2 bytes min + 16 bytes) */
void dequant_q4_1(const void *src, float *dst, int n)
{
    const uint8_t *p = src;
    int nb = n / 32;

    for (int b = 0; b < nb; b++) {
        float scale, minimum;
        uint16_t s16, m16;
        memcpy(&s16, p, 2); p += 2;
        memcpy(&m16, p, 2); p += 2;

        /* f16 to f32 (inline) */
        uint32_t f32_s, f32_m;
        {
            uint32_t sign = (uint32_t)(s16 >> 15) << 31;
            uint32_t exp  = (s16 >> 10) & 0x1f;
            uint32_t mant = s16 & 0x3ff;
            if (exp == 0 && mant == 0) f32_s = sign;
            else if (exp == 31) f32_s = sign | 0x7f800000 | (mant << 13);
            else f32_s = sign | ((exp + 112) << 23) | (mant << 13);
        }
        {
            uint32_t sign = (uint32_t)(m16 >> 15) << 31;
            uint32_t exp  = (m16 >> 10) & 0x1f;
            uint32_t mant = m16 & 0x3ff;
            if (exp == 0 && mant == 0) f32_m = sign;
            else if (exp == 31) f32_m = sign | 0x7f800000 | (mant << 13);
            else f32_m = sign | ((exp + 112) << 23) | (mant << 13);
        }
        memcpy(&scale, &f32_s, 4);
        memcpy(&minimum, &f32_m, 4);

        for (int i = 0; i < 16; i++) {
            uint8_t byte = p[i];
            dst[b * 32 + i * 2 + 0] = (byte & 0x0f) * scale + minimum;
            dst[b * 32 + i * 2 + 1] = (byte >> 4) * scale + minimum;
        }
        p += 16;
    }
}

/* Q8_0: 32 values per block, 34 bytes (2 bytes scale + 32 bytes quants) */
void dequant_q8_0(const void *src, float *dst, int n)
{
    const uint8_t *p = src;
    int nb = n / 32;

    for (int b = 0; b < nb; b++) {
        float scale;
        uint16_t s16;
        memcpy(&s16, p, 2);
        uint32_t f32;
        {
            uint32_t sign = (uint32_t)(s16 >> 15) << 31;
            uint32_t exp  = (s16 >> 10) & 0x1f;
            uint32_t mant = s16 & 0x3ff;
            if (exp == 0 && mant == 0) f32 = sign;
            else if (exp == 31) f32 = sign | 0x7f800000 | (mant << 13);
            else f32 = sign | ((exp + 112) << 23) | (mant << 13);
        }
        memcpy(&scale, &f32, 4);
        p += 2;

        const int8_t *qs = (const int8_t *)p;
        for (int i = 0; i < 32; i++) {
            dst[b * 32 + i] = qs[i] * scale;
        }
        p += 32;
    }
}

/* Q5_0: 32 values per block, 22 bytes */
void dequant_q5_0(const void *src, float *dst, int n)
{
    const uint8_t *p = src;
    int nb = n / 32;

    for (int b = 0; b < nb; b++) {
        float scale;
        uint16_t s16;
        memcpy(&s16, p, 2);
        uint32_t f32;
        {
            uint32_t sign = (uint32_t)(s16 >> 15) << 31;
            uint32_t exp  = (s16 >> 10) & 0x1f;
            uint32_t mant = s16 & 0x3ff;
            if (exp == 0 && mant == 0) f32 = sign;
            else if (exp == 31) f32 = sign | 0x7f800000 | (mant << 13);
            else f32 = sign | ((exp + 112) << 23) | (mant << 13);
        }
        memcpy(&scale, &f32, 4);
        p += 2;

        /* 4 bytes of high bits */
        uint32_t qh;
        memcpy(&qh, p, 4);
        p += 4;

        for (int i = 0; i < 16; i++) {
            uint8_t byte = p[i];
            int lo = (byte & 0x0f) | (((qh >> (i * 2)) & 1) << 4);
            int hi = (byte >> 4) | (((qh >> (i * 2 + 1)) & 1) << 4);
            dst[b * 32 + i * 2 + 0] = (lo - 16) * scale;
            dst[b * 32 + i * 2 + 1] = (hi - 16) * scale;
        }
        p += 16;
    }
}

/* Q5_1: 32 values per block, 24 bytes */
void dequant_q5_1(const void *src, float *dst, int n)
{
    const uint8_t *p = src;
    int nb = n / 32;

    for (int b = 0; b < nb; b++) {
        float scale, minimum;
        uint16_t s16, m16;
        memcpy(&s16, p, 2); p += 2;
        memcpy(&m16, p, 2); p += 2;

        uint32_t f32_s, f32_m;
        {
            uint32_t sign = (uint32_t)(s16 >> 15) << 31;
            uint32_t exp  = (s16 >> 10) & 0x1f;
            uint32_t mant = s16 & 0x3ff;
            if (exp == 0 && mant == 0) f32_s = sign;
            else if (exp == 31) f32_s = sign | 0x7f800000 | (mant << 13);
            else f32_s = sign | ((exp + 112) << 23) | (mant << 13);
        }
        {
            uint32_t sign = (uint32_t)(m16 >> 15) << 31;
            uint32_t exp  = (m16 >> 10) & 0x1f;
            uint32_t mant = m16 & 0x3ff;
            if (exp == 0 && mant == 0) f32_m = sign;
            else if (exp == 31) f32_m = sign | 0x7f800000 | (mant << 13);
            else f32_m = sign | ((exp + 112) << 23) | (mant << 13);
        }
        memcpy(&scale, &f32_s, 4);
        memcpy(&minimum, &f32_m, 4);

        uint32_t qh;
        memcpy(&qh, p, 4);
        p += 4;

        for (int i = 0; i < 16; i++) {
            uint8_t byte = p[i];
            int lo = (byte & 0x0f) | (((qh >> (i * 2)) & 1) << 4);
            int hi = (byte >> 4) | (((qh >> (i * 2 + 1)) & 1) << 4);
            dst[b * 32 + i * 2 + 0] = lo * scale + minimum;
            dst[b * 32 + i * 2 + 1] = hi * scale + minimum;
        }
        p += 16;
    }
}

/* Dispatch dequantization by type */
void dequantize(const void *src, float *dst, int n, gguf_quant_type_t type)
{
    switch (type) {
    case GGUF_TYPE_F32:
        memcpy(dst, src, n * sizeof(float));
        break;
    case GGUF_TYPE_F16: {
        const uint16_t *fp16 = src;
        for (int i = 0; i < n; i++) {
            uint16_t h = fp16[i];
            uint32_t sign = (uint32_t)(h >> 15) << 31;
            uint32_t exp  = (h >> 10) & 0x1f;
            uint32_t mant = h & 0x3ff;
            uint32_t f32;
            if (exp == 0 && mant == 0) f32 = sign;
            else if (exp == 31) f32 = sign | 0x7f800000 | (mant << 13);
            else f32 = sign | ((exp + 112) << 23) | (mant << 13);
            memcpy(&dst[i], &f32, 4);
        }
        break;
    }
    case GGUF_TYPE_Q4_0: dequant_q4_0(src, dst, n); break;
    case GGUF_TYPE_Q4_1: dequant_q4_1(src, dst, n); break;
    case GGUF_TYPE_Q5_0: dequant_q5_0(src, dst, n); break;
    case GGUF_TYPE_Q5_1: dequant_q5_1(src, dst, n); break;
    case GGUF_TYPE_Q8_0: dequant_q8_0(src, dst, n); break;
    default:
        memset(dst, 0, n * sizeof(float));
        break;
    }
}

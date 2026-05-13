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

/* ---- K-quant dequantization ---- */

static float f16_to_f32(uint16_t h)
{
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t f32;
    if (exp == 0 && mant == 0) f32 = sign;
    else if (exp == 31) f32 = sign | 0x7f800000 | (mant << 13);
    else f32 = sign | ((exp + 112) << 23) | (mant << 13);
    float result;
    memcpy(&result, &f32, 4);
    return result;
}

/*
 * Q6_K: 256 values per block, 210 bytes
 * Layout: ql[128] + qh[64] + scales[16] + d(f16)
 *   - ql: low 4 bits of each value (packed 2 per byte, 128 bytes for 256 values)
 *   - qh: high 2 bits of each value (packed 4 per byte, 64 bytes for 256 values)
 *   - scales: int8 scale per 16-value group (16 groups of 16)
 *   - d: overall f16 scale
 */
void dequant_q6_k(const void *src, float *dst, int n)
{
    const uint8_t *p = src;
    int nb = n / 256;

    for (int b = 0; b < nb; b++) {
        const uint8_t *ql = p;           /* 128 bytes: low 4 bits */
        const uint8_t *qh = p + 128;     /* 64 bytes: high 2 bits */
        const int8_t *scales = (const int8_t *)(p + 192); /* 16 bytes */
        uint16_t d16;
        memcpy(&d16, p + 208, 2);
        float d = f16_to_f32(d16);

        for (int i = 0; i < 256; i++) {
            /* Extract low 4 bits */
            int ql_idx = i / 2;
            int lo4;
            if (i % 2 == 0)
                lo4 = ql[ql_idx] & 0x0f;
            else
                lo4 = ql[ql_idx] >> 4;

            /* Extract high 2 bits */
            int qh_idx = i / 4;
            int qh_shift = (i % 4) * 2;
            int hi2 = (qh[qh_idx] >> qh_shift) & 0x03;

            /* Combine to 6-bit value (0..63), center at 32 */
            int q = lo4 | (hi2 << 4);
            q -= 32;

            /* Scale group (16 values per group) */
            int group = i / 16;
            float scale = d * scales[group];

            dst[b * 256 + i] = scale * q;
        }

        p += 210;
    }
}

/*
 * Q4_K: 256 values per block, 144 bytes
 * Layout: d(f16) + dmin(f16) + scales[12] + qs[128]
 */
void dequant_q4_k(const void *src, float *dst, int n)
{
    const uint8_t *p = src;
    int nb = n / 256;

    for (int b = 0; b < nb; b++) {
        uint16_t d16, dmin16;
        memcpy(&d16, p, 2);
        memcpy(&dmin16, p + 2, 2);
        float d = f16_to_f32(d16);
        float dmin = f16_to_f32(dmin16);

        const uint8_t *sc = p + 4;  /* 12 bytes of packed scales/mins */
        const uint8_t *qs = p + 16; /* 128 bytes of 4-bit quants */

        /* Decode scales and mins from packed 6-bit format */
        float scales[8], mins[8];
        for (int i = 0; i < 8; i++) {
            uint8_t sc_byte;
            uint8_t min_byte;

            if (i < 4) {
                sc_byte = sc[i] & 0x3f;
                min_byte = sc[i + 4] & 0x3f;
            } else {
                sc_byte = (sc[i + 4] & 0x0f) | ((sc[i - 4] >> 6) << 4);
                min_byte = (sc[i + 4] >> 4) | ((sc[i] >> 6) << 4);
            }
            scales[i] = d * sc_byte;
            mins[i] = dmin * min_byte;
        }

        /* Dequantize 256 values (8 groups of 32) */
        for (int g = 0; g < 8; g++) {
            float sc_val = scales[g];
            float mn_val = mins[g];
            for (int i = 0; i < 32; i++) {
                int qi = g * 32 + i;
                int byte_idx = qi / 2;
                int q;
                if (qi % 2 == 0)
                    q = qs[byte_idx] & 0x0f;
                else
                    q = qs[byte_idx] >> 4;

                dst[b * 256 + qi] = sc_val * q - mn_val;
            }
        }

        p += 144;
    }
}

/*
 * Q5_K: 256 values per block, 176 bytes
 * Layout: d(f16) + dmin(f16) + scales[12] + qh[32] + qs[128]
 */
void dequant_q5_k(const void *src, float *dst, int n)
{
    const uint8_t *p = src;
    int nb = n / 256;

    for (int b = 0; b < nb; b++) {
        uint16_t d16, dmin16;
        memcpy(&d16, p, 2);
        memcpy(&dmin16, p + 2, 2);
        float d = f16_to_f32(d16);
        float dmin = f16_to_f32(dmin16);

        const uint8_t *sc = p + 4;   /* 12 bytes packed scales */
        const uint8_t *qh = p + 16;  /* 32 bytes high bits */
        const uint8_t *qs = p + 48;  /* 128 bytes low 4 bits */

        float scales[8], mins[8];
        for (int i = 0; i < 8; i++) {
            uint8_t sc_byte, min_byte;
            if (i < 4) {
                sc_byte = sc[i] & 0x3f;
                min_byte = sc[i + 4] & 0x3f;
            } else {
                sc_byte = (sc[i + 4] & 0x0f) | ((sc[i - 4] >> 6) << 4);
                min_byte = (sc[i + 4] >> 4) | ((sc[i] >> 6) << 4);
            }
            scales[i] = d * sc_byte;
            mins[i] = dmin * min_byte;
        }

        for (int g = 0; g < 8; g++) {
            float sc_val = scales[g];
            float mn_val = mins[g];
            for (int i = 0; i < 32; i++) {
                int qi = g * 32 + i;
                int byte_idx = qi / 2;
                int lo4;
                if (qi % 2 == 0)
                    lo4 = qs[byte_idx] & 0x0f;
                else
                    lo4 = qs[byte_idx] >> 4;

                /* High bit from qh */
                int hi1 = (qh[qi / 8] >> (qi % 8)) & 1;
                int q = lo4 | (hi1 << 4);

                dst[b * 256 + qi] = sc_val * q - mn_val;
            }
        }

        p += 176;
    }
}

/*
 * Q3_K: 256 values per block, 110 bytes
 * Layout: hmask[32] + qs[64] + scales[12] + d(f16)
 */
void dequant_q3_k(const void *src, float *dst, int n)
{
    const uint8_t *p = src;
    int nb = n / 256;

    for (int b = 0; b < nb; b++) {
        const uint8_t *hmask = p;         /* 32 bytes: high bits */
        const uint8_t *qs = p + 32;       /* 64 bytes: low 2 bits */
        const uint8_t *sc_raw = p + 96;   /* 12 bytes: scales */
        uint16_t d16;
        memcpy(&d16, p + 108, 2);
        float d = f16_to_f32(d16);

        /* Decode 16 scales from 12 bytes (6 bits each, centered) */
        int8_t scales[16];
        for (int i = 0; i < 8; i++) {
            scales[i * 2]     = (int8_t)((sc_raw[i] & 0x0f) - 8);
            scales[i * 2 + 1] = (int8_t)((sc_raw[i] >> 4) - 8);
        }
        /* Remaining 4 scales from high bits */
        for (int i = 0; i < 4; i++) {
            scales[8 + i] = (int8_t)((sc_raw[8 + i] & 0x0f) - 8);
            if (8 + 4 + i < 16)
                scales[8 + 4 + i] = (int8_t)((sc_raw[8 + i] >> 4) - 8);
        }

        for (int i = 0; i < 256; i++) {
            /* Low 2 bits */
            int qs_idx = i / 4;
            int qs_shift = (i % 4) * 2;
            int lo2 = (qs[qs_idx] >> qs_shift) & 0x03;

            /* High bit */
            int hm_idx = i / 8;
            int hm_shift = i % 8;
            int hi1 = (hmask[hm_idx] >> hm_shift) & 1;

            int q = lo2 | (hi1 << 2);
            q -= 4;

            int group = i / 16;
            dst[b * 256 + i] = d * scales[group] * q;
        }

        p += 110;
    }
}

/*
 * Q2_K: 256 values per block, 84 bytes
 * Layout: scales[16] + qs[64] + d(f16) + dmin(f16)
 */
void dequant_q2_k(const void *src, float *dst, int n)
{
    const uint8_t *p = src;
    int nb = n / 256;

    for (int b = 0; b < nb; b++) {
        const uint8_t *sc_raw = p;     /* 16 bytes: packed scales+mins */
        const uint8_t *qs = p + 16;    /* 64 bytes: 2-bit quants */
        uint16_t d16, dmin16;
        memcpy(&d16, p + 80, 2);
        memcpy(&dmin16, p + 82, 2);
        float d = f16_to_f32(d16);
        float dmin = f16_to_f32(dmin16);

        for (int i = 0; i < 256; i++) {
            int group = i / 16;
            uint8_t sc_byte = sc_raw[group];
            float scale = d * (sc_byte & 0x0f);
            float min = dmin * (sc_byte >> 4);

            int qs_idx = i / 4;
            int qs_shift = (i % 4) * 2;
            int q = (qs[qs_idx] >> qs_shift) & 0x03;

            dst[b * 256 + i] = scale * q - min;
        }

        p += 84;
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
        for (int i = 0; i < n; i++)
            dst[i] = f16_to_f32(fp16[i]);
        break;
    }
    case GGUF_TYPE_Q4_0: dequant_q4_0(src, dst, n); break;
    case GGUF_TYPE_Q4_1: dequant_q4_1(src, dst, n); break;
    case GGUF_TYPE_Q5_0: dequant_q5_0(src, dst, n); break;
    case GGUF_TYPE_Q5_1: dequant_q5_1(src, dst, n); break;
    case GGUF_TYPE_Q8_0: dequant_q8_0(src, dst, n); break;
    case GGUF_TYPE_Q2_K: dequant_q2_k(src, dst, n); break;
    case GGUF_TYPE_Q3_K: dequant_q3_k(src, dst, n); break;
    case GGUF_TYPE_Q4_K: dequant_q4_k(src, dst, n); break;
    case GGUF_TYPE_Q5_K: dequant_q5_k(src, dst, n); break;
    case GGUF_TYPE_Q6_K: dequant_q6_k(src, dst, n); break;
    default:
        memset(dst, 0, n * sizeof(float));
        break;
    }
}

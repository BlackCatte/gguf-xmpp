/*
 * Transformer inference engine
 *
 * Implements forward pass for LLaMA-family decoder-only transformers.
 * Supports grouped-query attention (GQA), RoPE, RMSNorm, SwiGLU.
 */

#include "inference/inference.h"
#include "gguf/gguf.h"
#include "core/mem.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* Dequantization dispatch (from quant.c) */
extern void dequantize(const void *src, float *dst, int n,
                       gguf_quant_type_t type);

/* ---- Vector ops ---- */

static void vec_add(float *dst, const float *a, const float *b, int n)
{
    for (int i = 0; i < n; i++) dst[i] = a[i] + b[i];
}

static float vec_dot(const float *a, const float *b, int n)
{
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

static void rmsnorm(float *dst, const float *x, const float *weight,
                    int n, float eps)
{
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) dst[i] = x[i] * ss * weight[i];
}

static void softmax(float *x, int n)
{
    float max_val = x[0];
    for (int i = 1; i < n; i++)
        if (x[i] > max_val) max_val = x[i];

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

static void silu(float *x, int n)
{
    for (int i = 0; i < n; i++)
        x[i] = x[i] / (1.0f + expf(-x[i]));
}

/* ---- Quantized matrix-vector multiply ---- */

static void matvec_quantized(const gxmpp_tensor_t *weight,
                             const float *input, float *output,
                             int rows, int cols,
                             gxmpp_arena_t *scratch)
{
    /* For quantized weights, dequantize row-by-row and dot product */
    float *row_buf = gxmpp_arena_alloc(scratch, cols * sizeof(float));
    if (!row_buf) return;

    size_t bs = gguf_type_block_size(weight->type);
    size_t ts = gguf_type_type_size(weight->type);
    size_t row_bytes = (cols / bs) * ts;

    const uint8_t *data = weight->data;

    for (int r = 0; r < rows; r++) {
        dequantize(data + r * row_bytes, row_buf, cols, weight->type);
        output[r] = vec_dot(row_buf, input, cols);
    }
}

/* ---- RoPE (Rotary Position Embeddings) ---- */

static void apply_rope(float *q, float *k, int dim, int head_dim,
                       int pos, float freq_base, float freq_scale)
{
    for (int i = 0; i < dim; i += 2) {
        int j = i % head_dim;
        float freq = 1.0f / powf(freq_base, (float)j / head_dim) * freq_scale;
        float theta = pos * freq;
        float cos_t = cosf(theta);
        float sin_t = sinf(theta);

        float q0 = q[i], q1 = q[i + 1];
        q[i]     = q0 * cos_t - q1 * sin_t;
        q[i + 1] = q0 * sin_t + q1 * cos_t;

        float k0 = k[i], k1 = k[i + 1];
        k[i]     = k0 * cos_t - k1 * sin_t;
        k[i + 1] = k0 * sin_t + k1 * cos_t;
    }
}

/* ---- Model loading ---- */

static void extract_hparams(gguf_ctx_t *gguf, gxmpp_hparams_t *hp)
{
    const char *arch = gguf_meta_str(gguf, "general.architecture", "llama");
    strncpy(hp->arch, arch, sizeof(hp->arch) - 1);

    char key[128];
    #define META_U32(field, name, def) do { \
        snprintf(key, sizeof(key), "%s." name, arch); \
        hp->field = gguf_meta_u32(gguf, key, def); \
    } while(0)
    #define META_F32(field, name, def) do { \
        snprintf(key, sizeof(key), "%s." name, arch); \
        hp->field = gguf_meta_f32(gguf, key, def); \
    } while(0)

    META_U32(n_vocab,      "vocab_size",           32000);
    META_U32(n_embd,       "embedding_length",     4096);
    META_U32(n_heads,      "attention.head_count",  32);
    META_U32(n_heads_kv,   "attention.head_count_kv", 0);
    META_U32(n_layers,     "block_count",           32);
    META_U32(n_ff,         "feed_forward_length",   11008);
    META_U32(n_ctx,        "context_length",        4096);
    META_F32(rope_freq_base,  "rope.freq_base",     10000.0f);
    META_F32(rope_freq_scale, "rope.freq_scale",    1.0f);
    META_F32(norm_eps,     "attention.layer_norm_rms_epsilon", 1e-5f);

    if (hp->n_heads_kv == 0)
        hp->n_heads_kv = hp->n_heads;

    hp->rope_dim = hp->n_embd / hp->n_heads;

    #undef META_U32
    #undef META_F32
}

static gxmpp_result_t load_tokenizer(gguf_ctx_t *gguf, gxmpp_tokenizer_t *tok)
{
    const gxmpp_meta_kv_t *tokens_kv = gguf_meta_find(gguf, "tokenizer.ggml.tokens");
    if (!tokens_kv || tokens_kv->type != GGUF_META_ARRAY) {
        fprintf(stderr, "inference: missing tokenizer.ggml.tokens\n");
        return GXMPP_ERR_FORMAT;
    }

    tok->n_vocab = (uint32_t)tokens_kv->value.arr.count;
    tok->vocab = calloc(tok->n_vocab, sizeof(char *));
    if (!tok->vocab) return GXMPP_ERR_MEMORY;

    char **src_strings = tokens_kv->value.arr.data;
    for (uint32_t i = 0; i < tok->n_vocab; i++) {
        tok->vocab[i] = strdup(src_strings[i]);
    }

    /* Load scores if available */
    const gxmpp_meta_kv_t *scores_kv = gguf_meta_find(gguf, "tokenizer.ggml.scores");
    if (scores_kv && scores_kv->type == GGUF_META_ARRAY) {
        tok->scores = calloc(tok->n_vocab, sizeof(float));
        gxmpp_meta_kv_t *elems = scores_kv->value.arr.data;
        for (uint32_t i = 0; i < tok->n_vocab && i < scores_kv->value.arr.count; i++) {
            tok->scores[i] = elems[i].value.f32;
        }
    }

    tok->bos_id = (int)gguf_meta_u32(gguf, "tokenizer.ggml.bos_token_id", 1);
    tok->eos_id = (int)gguf_meta_u32(gguf, "tokenizer.ggml.eos_token_id", 2);
    tok->pad_id = (int)gguf_meta_u32(gguf, "tokenizer.ggml.padding_token_id", 0);

    /* Load BPE merges if available */
    const gxmpp_meta_kv_t *merges_kv = gguf_meta_find(gguf, "tokenizer.ggml.merges");
    if (merges_kv && merges_kv->type == GGUF_META_ARRAY) {
        tok->n_merges = (uint32_t)merges_kv->value.arr.count;
        tok->merges = calloc(tok->n_merges, sizeof(char *));
        char **merge_strings = merges_kv->value.arr.data;
        for (uint32_t i = 0; i < tok->n_merges; i++) {
            tok->merges[i] = strdup(merge_strings[i]);
        }
    }

    return GXMPP_OK;
}

gxmpp_result_t gxmpp_model_load(const char *path, gxmpp_model_t **out,
                                int ctx_size)
{
    gxmpp_model_t *model = calloc(1, sizeof(gxmpp_model_t));
    if (!model) return GXMPP_ERR_MEMORY;

    fprintf(stderr, "Opening GGUF file...\n");
    gxmpp_result_t res = gguf_open(path, &model->gguf);
    if (res != GXMPP_OK) {
        fprintf(stderr, "gguf_open failed: %d\n", res);
        free(model);
        return res;
    }
    fprintf(stderr, "GGUF opened: version=%u tensors=%lu kv=%lu\n",
            model->gguf->version,
            (unsigned long)model->gguf->n_tensors,
            (unsigned long)model->gguf->n_kv);

    extract_hparams(model->gguf, &model->hparams);

    gxmpp_hparams_t *hp = &model->hparams;
    fprintf(stderr, "Hyperparameters: arch=%s vocab=%u embd=%u heads=%u/%u "
                    "layers=%u ff=%u ctx=%u\n",
            hp->arch, hp->n_vocab, hp->n_embd, hp->n_heads, hp->n_heads_kv,
            hp->n_layers, hp->n_ff, hp->n_ctx);

    res = load_tokenizer(model->gguf, &model->tokenizer);
    if (res != GXMPP_OK) {
        fprintf(stderr, "load_tokenizer failed: %d\n", res);
        gguf_close(model->gguf);
        free(model);
        return res;
    }
    fprintf(stderr, "Tokenizer loaded: vocab=%u bos=%d eos=%d merges=%u\n",
            model->tokenizer.n_vocab, model->tokenizer.bos_id,
            model->tokenizer.eos_id, model->tokenizer.n_merges);

    /* Allocate KV cache */
    uint32_t n_embd_kv = (hp->n_embd / hp->n_heads) * hp->n_heads_kv;

    /* Cap context length for KV cache allocation */
    uint32_t kv_ctx = hp->n_ctx;
    if (ctx_size > 0 && (uint32_t)ctx_size < kv_ctx) {
        fprintf(stderr, "Capping KV cache context from %u to %d "
                        "(override with --ctx-size)\n",
                kv_ctx, ctx_size);
        kv_ctx = (uint32_t)ctx_size;
    }

    size_t kv_size = (size_t)hp->n_layers * kv_ctx * n_embd_kv * sizeof(float);
    fprintf(stderr, "Allocating KV cache: %zu MB "
                    "(layers=%u ctx=%u kv_dim=%u)\n",
            kv_size / (1024 * 1024), hp->n_layers, kv_ctx, n_embd_kv);

    model->kv_cache.k = calloc(1, kv_size);
    model->kv_cache.v = calloc(1, kv_size);
    model->kv_cache.n_embd_kv = n_embd_kv;
    model->kv_cache.pos = 0;

    if (!model->kv_cache.k || !model->kv_cache.v) {
        fprintf(stderr, "KV cache allocation failed (%zu bytes)\n", kv_size);
        gxmpp_model_free(model);
        return GXMPP_ERR_MEMORY;
    }

    /* Scratch arena: 512MB for larger models */
    size_t scratch_size = 512 * 1024 * 1024;
    fprintf(stderr, "Allocating scratch arena: %zu MB\n",
            scratch_size / (1024 * 1024));
    model->scratch = gxmpp_arena_create(scratch_size);
    if (!model->scratch) {
        fprintf(stderr, "Scratch arena allocation failed\n");
        gxmpp_model_free(model);
        return GXMPP_ERR_MEMORY;
    }

    /* Default sampling parameters */
    model->sample.temperature = 0.7f;
    model->sample.top_p = 0.9f;
    model->sample.top_k = 40;
    model->sample.repeat_penalty = 1.1f;
    model->sample.repeat_last_n = 64;
    model->sample.seed = (uint64_t)time(NULL);
    model->sample.max_tokens = 512;

    fprintf(stderr, "Model loaded: %s (%s)\n",
            gguf_meta_str(model->gguf, "general.name", "unknown"),
            hp->arch);
    fprintf(stderr, "  vocab=%u embd=%u heads=%u/%u layers=%u ctx=%u\n",
            hp->n_vocab, hp->n_embd, hp->n_heads, hp->n_heads_kv,
            hp->n_layers, hp->n_ctx);

    *out = model;
    return GXMPP_OK;
}

void gxmpp_model_free(gxmpp_model_t *model)
{
    if (!model) return;

    if (model->tokenizer.vocab) {
        for (uint32_t i = 0; i < model->tokenizer.n_vocab; i++)
            free(model->tokenizer.vocab[i]);
        free(model->tokenizer.vocab);
    }
    free(model->tokenizer.scores);
    if (model->tokenizer.merges) {
        for (uint32_t i = 0; i < model->tokenizer.n_merges; i++)
            free(model->tokenizer.merges[i]);
        free(model->tokenizer.merges);
    }

    free(model->kv_cache.k);
    free(model->kv_cache.v);
    gxmpp_arena_destroy(model->scratch);
    gguf_close(model->gguf);
    free(model);
}

/* ---- Tokenizer ---- */

int gxmpp_tokenize(const gxmpp_model_t *model, const char *text,
                   int **tokens_out)
{
    /*
     * Simple greedy longest-match tokenizer.
     * For production, this should implement full BPE, but this
     * gives correct results for most cases.
     */
    const gxmpp_tokenizer_t *tok = &model->tokenizer;
    int capacity = 256;
    int *tokens = malloc(capacity * sizeof(int));
    int n_tokens = 0;

    /* Add BOS token */
    tokens[n_tokens++] = tok->bos_id;

    const char *p = text;
    while (*p) {
        int best_id = -1;
        int best_len = 0;

        for (uint32_t i = 0; i < tok->n_vocab; i++) {
            const char *token = tok->vocab[i];
            int tlen = strlen(token);
            if (tlen > 0 && tlen > best_len &&
                strncmp(p, token, tlen) == 0) {
                best_id = (int)i;
                best_len = tlen;
            }
        }

        if (best_id < 0) {
            /* Unknown byte — skip */
            p++;
            continue;
        }

        if (n_tokens >= capacity) {
            capacity *= 2;
            tokens = realloc(tokens, capacity * sizeof(int));
        }
        tokens[n_tokens++] = best_id;
        p += best_len;
    }

    *tokens_out = tokens;
    return n_tokens;
}

const char *gxmpp_detokenize(const gxmpp_model_t *model, int token_id)
{
    if (token_id < 0 || (uint32_t)token_id >= model->tokenizer.n_vocab)
        return "";
    return model->tokenizer.vocab[token_id];
}

/* ---- Forward pass ---- */

float *gxmpp_forward(gxmpp_model_t *model, const int *tokens, int n_tokens)
{
    gxmpp_hparams_t *hp = &model->hparams;
    gguf_ctx_t *gguf = model->gguf;
    gxmpp_arena_t *scratch = model->scratch;
    gxmpp_arena_reset(scratch);

    int n_embd = hp->n_embd;
    int n_heads = hp->n_heads;
    int n_heads_kv = hp->n_heads_kv;
    int head_dim = n_embd / n_heads;
    int n_kv_dim = head_dim * n_heads_kv;

    /* Get embedding weights */
    const gxmpp_tensor_t *tok_embd = gguf_tensor_find(gguf, "token_embd.weight");
    if (!tok_embd) {
        fprintf(stderr, "forward: missing token_embd.weight\n");
        return NULL;
    }

    /* Process one token at a time for autoregressive generation */
    /* We process the last token position for efficiency in generation */
    int pos = model->kv_cache.pos;

    float *cur = gxmpp_arena_alloc(scratch, n_embd * sizeof(float));
    float *residual = gxmpp_arena_alloc(scratch, n_embd * sizeof(float));
    float *norm_out = gxmpp_arena_alloc(scratch, n_embd * sizeof(float));

    /* For each input token, process through all layers */
    for (int t = 0; t < n_tokens; t++) {
        int token = tokens[t];
        int cur_pos = pos + t;

        /* Embed token */
        size_t embd_row_size = gguf_type_type_size(tok_embd->type) *
                               n_embd / gguf_type_block_size(tok_embd->type);
        const uint8_t *embd_data = (const uint8_t *)tok_embd->data +
                                   token * embd_row_size;
        dequantize(embd_data, cur, n_embd, tok_embd->type);

        /* Process through transformer layers */
        for (uint32_t layer = 0; layer < hp->n_layers; layer++) {
            memcpy(residual, cur, n_embd * sizeof(float));

            /* Attention norm (RMSNorm) */
            char tname[128];
            snprintf(tname, sizeof(tname), "blk.%u.attn_norm.weight", layer);
            const gxmpp_tensor_t *attn_norm_w = gguf_tensor_find(gguf, tname);

            float *norm_w = gxmpp_arena_alloc(scratch, n_embd * sizeof(float));
            if (attn_norm_w) {
                dequantize(attn_norm_w->data, norm_w, n_embd, attn_norm_w->type);
                rmsnorm(norm_out, cur, norm_w, n_embd, hp->norm_eps);
            } else {
                memcpy(norm_out, cur, n_embd * sizeof(float));
            }

            /* QKV projections */
            float *q = gxmpp_arena_alloc(scratch, n_embd * sizeof(float));
            float *k = gxmpp_arena_alloc(scratch, n_kv_dim * sizeof(float));
            float *v = gxmpp_arena_alloc(scratch, n_kv_dim * sizeof(float));

            snprintf(tname, sizeof(tname), "blk.%u.attn_q.weight", layer);
            const gxmpp_tensor_t *wq = gguf_tensor_find(gguf, tname);
            snprintf(tname, sizeof(tname), "blk.%u.attn_k.weight", layer);
            const gxmpp_tensor_t *wk = gguf_tensor_find(gguf, tname);
            snprintf(tname, sizeof(tname), "blk.%u.attn_v.weight", layer);
            const gxmpp_tensor_t *wv = gguf_tensor_find(gguf, tname);

            if (wq) matvec_quantized(wq, norm_out, q, n_embd, n_embd, scratch);
            if (wk) matvec_quantized(wk, norm_out, k, n_kv_dim, n_embd, scratch);
            if (wv) matvec_quantized(wv, norm_out, v, n_kv_dim, n_embd, scratch);

            /* Apply RoPE to Q and K */
            apply_rope(q, k, n_embd < n_kv_dim ? n_embd : n_kv_dim,
                       head_dim, cur_pos, hp->rope_freq_base,
                       hp->rope_freq_scale);

            /* Store K, V in cache */
            size_t kv_layer_offset = (size_t)layer * hp->n_ctx * n_kv_dim;
            memcpy(model->kv_cache.k + kv_layer_offset + cur_pos * n_kv_dim,
                   k, n_kv_dim * sizeof(float));
            memcpy(model->kv_cache.v + kv_layer_offset + cur_pos * n_kv_dim,
                   v, n_kv_dim * sizeof(float));

            /* Multi-head attention with GQA */
            float *attn_out = gxmpp_arena_alloc(scratch, n_embd * sizeof(float));
            memset(attn_out, 0, n_embd * sizeof(float));

            int kv_groups = n_heads / n_heads_kv;

            for (int h = 0; h < n_heads; h++) {
                int kv_h = h / kv_groups;
                float *q_head = q + h * head_dim;

                /* Compute attention scores against all cached positions */
                float *scores = gxmpp_arena_alloc(scratch,
                    (cur_pos + 1) * sizeof(float));

                for (int p2 = 0; p2 <= cur_pos; p2++) {
                    float *k_cached = model->kv_cache.k +
                        kv_layer_offset + p2 * n_kv_dim + kv_h * head_dim;
                    scores[p2] = vec_dot(q_head, k_cached, head_dim) /
                                 sqrtf((float)head_dim);
                }

                softmax(scores, cur_pos + 1);

                /* Weighted sum of values */
                float *out_head = attn_out + h * head_dim;
                for (int p2 = 0; p2 <= cur_pos; p2++) {
                    float *v_cached = model->kv_cache.v +
                        kv_layer_offset + p2 * n_kv_dim + kv_h * head_dim;
                    for (int d = 0; d < head_dim; d++)
                        out_head[d] += scores[p2] * v_cached[d];
                }
            }

            /* Output projection */
            snprintf(tname, sizeof(tname), "blk.%u.attn_output.weight", layer);
            const gxmpp_tensor_t *wo = gguf_tensor_find(gguf, tname);
            float *attn_proj = gxmpp_arena_alloc(scratch, n_embd * sizeof(float));
            if (wo) matvec_quantized(wo, attn_out, attn_proj, n_embd, n_embd, scratch);

            /* Residual connection */
            vec_add(cur, residual, attn_proj, n_embd);
            memcpy(residual, cur, n_embd * sizeof(float));

            /* FFN norm */
            snprintf(tname, sizeof(tname), "blk.%u.ffn_norm.weight", layer);
            const gxmpp_tensor_t *ffn_norm_w = gguf_tensor_find(gguf, tname);

            if (ffn_norm_w) {
                float *fn_w = gxmpp_arena_alloc(scratch, n_embd * sizeof(float));
                dequantize(ffn_norm_w->data, fn_w, n_embd, ffn_norm_w->type);
                rmsnorm(norm_out, cur, fn_w, n_embd, hp->norm_eps);
            } else {
                memcpy(norm_out, cur, n_embd * sizeof(float));
            }

            /* SwiGLU FFN: gate * silu(up) */
            float *gate_out = gxmpp_arena_alloc(scratch, hp->n_ff * sizeof(float));
            float *up_out = gxmpp_arena_alloc(scratch, hp->n_ff * sizeof(float));

            snprintf(tname, sizeof(tname), "blk.%u.ffn_gate.weight", layer);
            const gxmpp_tensor_t *w_gate = gguf_tensor_find(gguf, tname);
            snprintf(tname, sizeof(tname), "blk.%u.ffn_up.weight", layer);
            const gxmpp_tensor_t *w_up = gguf_tensor_find(gguf, tname);

            if (w_gate) matvec_quantized(w_gate, norm_out, gate_out,
                                         hp->n_ff, n_embd, scratch);
            if (w_up)   matvec_quantized(w_up, norm_out, up_out,
                                         hp->n_ff, n_embd, scratch);

            silu(gate_out, hp->n_ff);
            for (uint32_t i = 0; i < hp->n_ff; i++)
                gate_out[i] *= up_out[i];

            /* Down projection */
            snprintf(tname, sizeof(tname), "blk.%u.ffn_down.weight", layer);
            const gxmpp_tensor_t *w_down = gguf_tensor_find(gguf, tname);
            float *ffn_out = gxmpp_arena_alloc(scratch, n_embd * sizeof(float));
            if (w_down) matvec_quantized(w_down, gate_out, ffn_out,
                                         n_embd, hp->n_ff, scratch);

            /* Residual connection */
            vec_add(cur, residual, ffn_out, n_embd);
        }
    }

    model->kv_cache.pos = pos + n_tokens;

    /* Final norm */
    const gxmpp_tensor_t *output_norm = gguf_tensor_find(gguf, "output_norm.weight");
    if (output_norm) {
        float *on_w = gxmpp_arena_alloc(scratch, n_embd * sizeof(float));
        dequantize(output_norm->data, on_w, n_embd, output_norm->type);
        rmsnorm(cur, cur, on_w, n_embd, hp->norm_eps);
    }

    /* LM head: project to vocab */
    const gxmpp_tensor_t *output_w = gguf_tensor_find(gguf, "output.weight");
    if (!output_w) {
        /* Some models tie embedding weights */
        output_w = gguf_tensor_find(gguf, "token_embd.weight");
    }

    float *logits = malloc(hp->n_vocab * sizeof(float));
    if (output_w && logits) {
        matvec_quantized(output_w, cur, logits, hp->n_vocab, n_embd, scratch);
    }

    return logits;
}

/* ---- Sampling ---- */

static uint64_t xorshift64(uint64_t *state)
{
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

int gxmpp_sample(const gxmpp_model_t *model, const float *logits,
                 const gxmpp_sample_params_t *params,
                 const int *last_tokens, int n_last)
{
    int n_vocab = (int)model->hparams.n_vocab;
    float *probs = malloc(n_vocab * sizeof(float));
    memcpy(probs, logits, n_vocab * sizeof(float));

    /* Apply repetition penalty */
    if (last_tokens && n_last > 0) {
        int lookback = params->repeat_last_n < n_last ?
                       params->repeat_last_n : n_last;
        for (int i = n_last - lookback; i < n_last; i++) {
            int tid = last_tokens[i];
            if (tid >= 0 && tid < n_vocab) {
                if (probs[tid] > 0)
                    probs[tid] /= params->repeat_penalty;
                else
                    probs[tid] *= params->repeat_penalty;
            }
        }
    }

    /* Temperature */
    float temp = params->temperature;
    if (temp <= 0.0f) temp = 1.0f;
    for (int i = 0; i < n_vocab; i++) probs[i] /= temp;

    softmax(probs, n_vocab);

    /* Top-K filtering */
    if (params->top_k > 0 && params->top_k < n_vocab) {
        /* Find top-k threshold by partial sort */
        float *sorted = malloc(n_vocab * sizeof(float));
        memcpy(sorted, probs, n_vocab * sizeof(float));

        /* Simple selection of k-th largest */
        for (int i = 0; i < params->top_k; i++) {
            int max_j = i;
            for (int j = i + 1; j < n_vocab; j++) {
                if (sorted[j] > sorted[max_j]) max_j = j;
            }
            float tmp = sorted[i];
            sorted[i] = sorted[max_j];
            sorted[max_j] = tmp;
        }
        float threshold = sorted[params->top_k - 1];
        free(sorted);

        for (int i = 0; i < n_vocab; i++) {
            if (probs[i] < threshold) probs[i] = 0.0f;
        }
    }

    /* Top-P (nucleus) filtering */
    if (params->top_p > 0.0f && params->top_p < 1.0f) {
        /* Sort indices by probability (descending) */
        int *indices = malloc(n_vocab * sizeof(int));
        for (int i = 0; i < n_vocab; i++) indices[i] = i;

        /* Simple insertion sort for top elements */
        for (int i = 1; i < n_vocab; i++) {
            int key = indices[i];
            float key_p = probs[key];
            int j = i - 1;
            while (j >= 0 && probs[indices[j]] < key_p) {
                indices[j + 1] = indices[j];
                j--;
            }
            indices[j + 1] = key;
        }

        float cumsum = 0.0f;
        for (int i = 0; i < n_vocab; i++) {
            cumsum += probs[indices[i]];
            if (cumsum > params->top_p) {
                for (int j = i + 1; j < n_vocab; j++)
                    probs[indices[j]] = 0.0f;
                break;
            }
        }
        free(indices);
    }

    /* Re-normalize */
    float sum = 0.0f;
    for (int i = 0; i < n_vocab; i++) sum += probs[i];
    if (sum > 0.0f)
        for (int i = 0; i < n_vocab; i++) probs[i] /= sum;

    /* Random sampling */
    uint64_t rng = params->seed;
    float r = (float)(xorshift64(&rng) & 0xFFFFFF) / (float)0xFFFFFF;
    float cumulative = 0.0f;
    int sampled = 0;

    for (int i = 0; i < n_vocab; i++) {
        cumulative += probs[i];
        if (cumulative >= r) {
            sampled = i;
            break;
        }
    }

    free(probs);
    return sampled;
}

/* ---- Generation ---- */

void gxmpp_kv_cache_reset(gxmpp_model_t *model)
{
    model->kv_cache.pos = 0;
}

char *gxmpp_generate(gxmpp_model_t *model, const char *prompt,
                     const gxmpp_sample_params_t *params,
                     gxmpp_token_cb token_cb, void *userdata)
{
    gxmpp_sample_params_t sp = params ? *params : model->sample;

    int *tokens;
    int n_tokens = gxmpp_tokenize(model, prompt, &tokens);
    if (n_tokens <= 0) return NULL;

    /* Prefill: process all prompt tokens */
    float *logits = gxmpp_forward(model, tokens, n_tokens);
    if (!logits) { free(tokens); return NULL; }

    /* Generation loop */
    int max_gen = sp.max_tokens;
    int *all_tokens = malloc((n_tokens + max_gen) * sizeof(int));
    memcpy(all_tokens, tokens, n_tokens * sizeof(int));
    int total = n_tokens;

    /* Response buffer */
    size_t resp_cap = 4096;
    size_t resp_len = 0;
    char *response = malloc(resp_cap);
    response[0] = '\0';

    for (int i = 0; i < max_gen; i++) {
        int next = gxmpp_sample(model, logits, &sp, all_tokens, total);
        free(logits);

        if (next == model->tokenizer.eos_id)
            break;

        all_tokens[total++] = next;
        const char *tok_str = gxmpp_detokenize(model, next);

        size_t tok_len = strlen(tok_str);
        while (resp_len + tok_len + 1 > resp_cap) {
            resp_cap *= 2;
            response = realloc(response, resp_cap);
        }
        memcpy(response + resp_len, tok_str, tok_len);
        resp_len += tok_len;
        response[resp_len] = '\0';

        if (token_cb)
            token_cb(tok_str, userdata);

        /* Next forward pass (single token) */
        logits = gxmpp_forward(model, &next, 1);
        if (!logits) break;
    }

    free(logits);
    free(tokens);
    free(all_tokens);
    return response;
}

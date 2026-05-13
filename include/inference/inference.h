/*
 * Transformer inference engine
 *
 * Implements forward pass for decoder-only transformer models
 * loaded from GGUF files (LLaMA-family architecture).
 */

#ifndef GXMPP_INFERENCE_H
#define GXMPP_INFERENCE_H

#include "core/types.h"
#include "core/mem.h"
#include "gguf/gguf.h"

/* Model hyperparameters extracted from GGUF metadata */
typedef struct {
    uint32_t n_vocab;
    uint32_t n_embd;
    uint32_t n_heads;
    uint32_t n_heads_kv;      /* for grouped-query attention */
    uint32_t n_layers;
    uint32_t n_ff;
    uint32_t n_ctx;           /* max context length */
    uint32_t rope_dim;
    float    rope_freq_base;
    float    rope_freq_scale;
    float    norm_eps;
    char     arch[64];        /* e.g. "llama", "mistral", "phi" */
} gxmpp_hparams_t;

/* KV cache for autoregressive generation */
typedef struct {
    float   *k;               /* [n_layers * n_ctx * n_embd_kv] */
    float   *v;               /* [n_layers * n_ctx * n_embd_kv] */
    int      pos;             /* current sequence position */
    uint32_t n_embd_kv;       /* per-layer KV dimension */
} gxmpp_kv_cache_t;

/* Tokenizer (loaded from GGUF metadata) */
typedef struct {
    char   **vocab;           /* token strings */
    float   *scores;          /* token scores/logprob */
    uint32_t n_vocab;
    int      bos_id;
    int      eos_id;
    int      pad_id;
    /* BPE merge table */
    char   **merges;
    uint32_t n_merges;
} gxmpp_tokenizer_t;

/* Sampling parameters */
typedef struct {
    float    temperature;
    float    top_p;
    int      top_k;
    float    repeat_penalty;
    int      repeat_last_n;
    uint64_t seed;
    int      max_tokens;
} gxmpp_sample_params_t;

/* Model context */
typedef struct {
    gguf_ctx_t          *gguf;
    gxmpp_hparams_t     hparams;
    gxmpp_tokenizer_t   tokenizer;
    gxmpp_kv_cache_t    kv_cache;
    gxmpp_arena_t      *scratch;    /* per-inference scratch memory */
    gxmpp_sample_params_t sample;
} gxmpp_model_t;

/* Load model from GGUF file */
gxmpp_result_t gxmpp_model_load(const char *path, gxmpp_model_t **model);

/* Free model */
void gxmpp_model_free(gxmpp_model_t *model);

/* Tokenize a string. Returns token count. Caller must free *tokens. */
int gxmpp_tokenize(const gxmpp_model_t *model, const char *text,
                   int **tokens);

/* Detokenize a single token ID to string */
const char *gxmpp_detokenize(const gxmpp_model_t *model, int token_id);

/* Run inference: generate response for the given prompt.
 * Calls token_cb for each generated token (streaming).
 * Returns the full response (caller must free). */
char *gxmpp_generate(gxmpp_model_t *model, const char *prompt,
                     const gxmpp_sample_params_t *params,
                     gxmpp_token_cb token_cb, void *userdata);

/* Reset KV cache (start new conversation) */
void gxmpp_kv_cache_reset(gxmpp_model_t *model);

/* Forward pass for a batch of tokens. Returns logits [n_vocab]. */
float *gxmpp_forward(gxmpp_model_t *model, const int *tokens, int n_tokens);

/* Sample next token from logits */
int gxmpp_sample(const gxmpp_model_t *model, const float *logits,
                 const gxmpp_sample_params_t *params,
                 const int *last_tokens, int n_last);

#endif /* GXMPP_INFERENCE_H */

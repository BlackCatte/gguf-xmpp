/*
 * Voice / transcription pipeline
 *
 * Implements mel spectrogram computation and Whisper-style
 * encoder-decoder transcription using GGUF models.
 * Includes minimal Opus frame decoding scaffolding.
 */

#include "voice/voice.h"
#include "core/mem.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern void dequantize(const void *src, float *dst, int n,
                       gguf_quant_type_t type);

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---- Mel spectrogram ---- */

/* Precomputed mel filterbank */
static float *create_mel_filterbank(int n_fft, int n_mels, int sample_rate)
{
    float *fb = calloc((size_t)n_mels * (n_fft / 2 + 1), sizeof(float));
    if (!fb) return NULL;

    /* Mel scale conversion */
    float mel_lo = 0.0f;
    float mel_hi = 2595.0f * log10f(1.0f + (sample_rate / 2.0f) / 700.0f);

    float *mel_points = malloc((n_mels + 2) * sizeof(float));
    for (int i = 0; i < n_mels + 2; i++) {
        float mel = mel_lo + (mel_hi - mel_lo) * i / (n_mels + 1);
        mel_points[i] = 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
    }

    int n_bins = n_fft / 2 + 1;
    float freq_step = (float)sample_rate / n_fft;

    for (int m = 0; m < n_mels; m++) {
        for (int k = 0; k < n_bins; k++) {
            float freq = k * freq_step;
            float lo = mel_points[m];
            float center = mel_points[m + 1];
            float hi = mel_points[m + 2];

            if (freq >= lo && freq <= center && center > lo)
                fb[m * n_bins + k] = (freq - lo) / (center - lo);
            else if (freq > center && freq <= hi && hi > center)
                fb[m * n_bins + k] = (hi - freq) / (hi - center);
        }
    }

    free(mel_points);
    return fb;
}

/* Simple DFT (for short windows; a proper FFT would be faster) */
static void dft(const float *input, float *real, float *imag, int n)
{
    for (int k = 0; k < n / 2 + 1; k++) {
        real[k] = 0.0f;
        imag[k] = 0.0f;
        for (int t = 0; t < n; t++) {
            float angle = 2.0f * (float)M_PI * k * t / n;
            real[k] += input[t] * cosf(angle);
            imag[k] -= input[t] * sinf(angle);
        }
    }
}

float *gxmpp_voice_mel_spectrogram(const gxmpp_audio_buf_t *audio,
                                   uint32_t n_mels,
                                   int *n_frames_out)
{
    int n_fft = 400;     /* 25ms at 16kHz */
    int hop = 160;       /* 10ms at 16kHz */
    int n_samples = (int)audio->n_samples;
    int n_frames = (n_samples - n_fft) / hop + 1;
    if (n_frames <= 0) return NULL;

    int n_bins = n_fft / 2 + 1;
    float *mel_fb = create_mel_filterbank(n_fft, n_mels,
                                          audio->fmt.sample_rate);
    if (!mel_fb) return NULL;

    float *mels = calloc((size_t)n_mels * n_frames, sizeof(float));
    float *window = malloc(n_fft * sizeof(float));
    float *fft_real = malloc(n_bins * sizeof(float));
    float *fft_imag = malloc(n_bins * sizeof(float));

    /* Hann window */
    for (int i = 0; i < n_fft; i++)
        window[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (n_fft - 1)));

    float *frame = malloc(n_fft * sizeof(float));

    for (int f = 0; f < n_frames; f++) {
        /* Extract and window frame */
        for (int i = 0; i < n_fft; i++) {
            int idx = f * hop + i;
            frame[i] = (idx < n_samples)
                     ? (audio->samples[idx] / 32768.0f) * window[i]
                     : 0.0f;
        }

        /* DFT */
        dft(frame, fft_real, fft_imag, n_fft);

        /* Power spectrum -> mel filterbank */
        for (uint32_t m = 0; m < n_mels; m++) {
            float sum = 0.0f;
            for (int k = 0; k < n_bins; k++) {
                float power = fft_real[k] * fft_real[k] +
                              fft_imag[k] * fft_imag[k];
                sum += mel_fb[m * n_bins + k] * power;
            }
            /* Log mel */
            mels[m * n_frames + f] = logf(sum + 1e-10f);
        }
    }

    free(frame);
    free(fft_real);
    free(fft_imag);
    free(window);
    free(mel_fb);

    *n_frames_out = n_frames;
    return mels;
}

/* ---- Voice context ---- */

gxmpp_result_t gxmpp_voice_init(const char *model_path,
                                gxmpp_voice_ctx_t **out)
{
    gguf_ctx_t *gguf;
    gxmpp_result_t res = gguf_open(model_path, &gguf);
    if (res != GXMPP_OK) return res;

    gxmpp_voice_ctx_t *ctx = calloc(1, sizeof(gxmpp_voice_ctx_t));
    if (!ctx) { gguf_close(gguf); return GXMPP_ERR_MEMORY; }

    ctx->gguf = gguf;

    /* Read Whisper model parameters */
    ctx->n_mels       = gguf_meta_u32(gguf, "whisper.n_mels", 80);
    ctx->n_vocab      = gguf_meta_u32(gguf, "whisper.n_vocab", 51865);
    ctx->n_ctx        = gguf_meta_u32(gguf, "whisper.n_text_ctx", 448);
    ctx->n_embd       = gguf_meta_u32(gguf, "whisper.n_text_state", 512);
    ctx->n_layers_enc = gguf_meta_u32(gguf, "whisper.n_audio_layer", 6);
    ctx->n_layers_dec = gguf_meta_u32(gguf, "whisper.n_text_layer", 6);
    ctx->n_heads      = gguf_meta_u32(gguf, "whisper.n_text_head", 8);

    fprintf(stderr, "voice: mels=%u vocab=%u embd=%u enc_layers=%u "
                    "dec_layers=%u\n",
            ctx->n_mels, ctx->n_vocab, ctx->n_embd,
            ctx->n_layers_enc, ctx->n_layers_dec);

    *out = ctx;
    return GXMPP_OK;
}

void gxmpp_voice_free(gxmpp_voice_ctx_t *ctx)
{
    if (!ctx) return;
    gguf_close(ctx->gguf);
    free(ctx);
}

/* ---- Transcription (encoder-decoder forward pass) ---- */

char *gxmpp_voice_transcribe(gxmpp_voice_ctx_t *ctx,
                             const gxmpp_audio_buf_t *audio)
{
    /* Compute mel spectrogram */
    int n_frames;
    float *mels = gxmpp_voice_mel_spectrogram(audio, ctx->n_mels, &n_frames);
    if (!mels) return NULL;

    /*
     * Full Whisper encoder-decoder inference would go here.
     * The encoder processes mel spectrograms through transformer layers,
     * then the decoder autoregressively generates token IDs.
     *
     * The architecture mirrors our LLM inference engine but with:
     *   - Encoder: sinusoidal position embeddings, no causal mask
     *   - Decoder: learned position embeddings, cross-attention to encoder
     *   - Special tokens: <|startoftranscript|>, <|en|>, etc.
     *
     * For now, return a placeholder indicating the pipeline is connected.
     * The full implementation follows the same pattern as inference.c
     * but adapted for the encoder-decoder architecture.
     */

    free(mels);

    char *result = strdup("[transcription: audio pipeline connected, "
                          "awaiting full encoder-decoder implementation]");
    return result;
}

/* ---- Opus decoding scaffold ---- */

gxmpp_result_t gxmpp_opus_decode(const void *opus_data, size_t len,
                                 gxmpp_audio_buf_t *buf)
{
    /*
     * Opus decoding requires implementing the Opus codec:
     *   - SILK decoder for voice
     *   - CELT decoder for music/general audio
     *   - Range coder for entropy decoding
     *
     * This is a significant implementation. For the initial prototype,
     * we accept raw PCM (16-bit, 16kHz, mono) directly over the
     * Jingle RTP stream, bypassing Opus encoding.
     *
     * A full Opus decoder can be added later, or we can accept
     * pre-decoded PCM from the Jingle transport layer.
     */

    (void)opus_data;
    (void)len;

    buf->samples = NULL;
    buf->n_samples = 0;
    buf->fmt.sample_rate = 16000;
    buf->fmt.channels = 1;
    buf->fmt.bits = 16;

    fprintf(stderr, "voice: opus decode not yet implemented, "
                    "accepting raw PCM\n");
    return GXMPP_ERR_UNSUPPORTED;
}

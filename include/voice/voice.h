/*
 * Voice communication layer
 *
 * Handles XMPP Jingle voice sessions:
 *   - Audio stream decoding (Opus/PCM)
 *   - Speech-to-text transcription (via GGUF Whisper-style model)
 *   - Integration with the inference engine for voice conversations
 *
 * The transcription service runs locally using a GGUF-format
 * speech recognition model, keeping everything self-contained.
 */

#ifndef GXMPP_VOICE_H
#define GXMPP_VOICE_H

#include "core/types.h"
#include "gguf/gguf.h"

/* Audio format for PCM samples */
typedef struct {
    uint32_t sample_rate;   /* typically 16000 for transcription */
    uint16_t channels;      /* 1 = mono */
    uint16_t bits;          /* 16 */
} gxmpp_audio_fmt_t;

/* Audio buffer */
typedef struct {
    int16_t *samples;
    size_t   n_samples;
    gxmpp_audio_fmt_t fmt;
} gxmpp_audio_buf_t;

/* Voice/transcription context */
typedef struct {
    gguf_ctx_t  *gguf;         /* Whisper-style GGUF model */
    uint32_t     n_mels;       /* mel spectrogram bins */
    uint32_t     n_vocab;
    uint32_t     n_ctx;
    uint32_t     n_embd;
    uint32_t     n_layers_enc;
    uint32_t     n_layers_dec;
    uint32_t     n_heads;
} gxmpp_voice_ctx_t;

/* Jingle session state */
typedef enum {
    JINGLE_IDLE,
    JINGLE_PENDING,
    JINGLE_ACTIVE,
    JINGLE_ENDED,
} gxmpp_jingle_state_t;

/* Initialize voice/transcription from a Whisper GGUF model */
gxmpp_result_t gxmpp_voice_init(const char *model_path,
                                gxmpp_voice_ctx_t **ctx);

/* Free voice context */
void gxmpp_voice_free(gxmpp_voice_ctx_t *ctx);

/* Transcribe audio buffer to text. Caller must free result. */
char *gxmpp_voice_transcribe(gxmpp_voice_ctx_t *ctx,
                             const gxmpp_audio_buf_t *audio);

/* Compute mel spectrogram from PCM audio.
 * Returns float array [n_mels * n_frames]. Caller must free.
 * Writes frame count to *n_frames_out. */
float *gxmpp_voice_mel_spectrogram(const gxmpp_audio_buf_t *audio,
                                   uint32_t n_mels,
                                   int *n_frames_out);

/* Decode Opus packet to PCM. Caller must free buf->samples. */
gxmpp_result_t gxmpp_opus_decode(const void *opus_data, size_t len,
                                 gxmpp_audio_buf_t *buf);

#endif /* GXMPP_VOICE_H */

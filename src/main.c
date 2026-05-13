/*
 * gguf-xmpp: main entry point
 *
 * Loads a GGUF model, connects to an XMPP server, and bridges
 * incoming messages to the inference engine. Supports text,
 * vision (image attachments), and voice (transcription) modalities.
 *
 * Usage:
 *   gguf-xmpp --model <path.gguf> --jid <user@domain/resource>
 *             --password <pass> [--host <server>] [--port <port>]
 *             [--voice-model <whisper.gguf>]
 *             [--temperature <0.7>] [--max-tokens <512>]
 *             [--presence-status <status>]
 */

#include "gguf/gguf.h"
#include "inference/inference.h"
#include "xmpp/xmpp.h"
#include "vision/vision.h"
#include "voice/voice.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>

typedef struct {
    gxmpp_model_t       *model;
    gxmpp_vision_ctx_t  *vision;
    gxmpp_voice_ctx_t   *voice;
    xmpp_conn_t         *conn;
} app_ctx_t;

static volatile int g_running = 1;

static void signal_handler(int sig)
{
    (void)sig;
    g_running = 0;
}

static void on_token(const char *token, void *userdata)
{
    (void)userdata;
    fputs(token, stderr);
}

static void on_message(xmpp_conn_t *conn, const char *from,
                       const char *body, void *userdata)
{
    app_ctx_t *app = userdata;

    fprintf(stderr, "\n[%s]: %s\n", from, body);

    /* Generate response */
    char *response = gxmpp_generate(app->model, body, NULL, on_token, NULL);

    if (response) {
        fprintf(stderr, "\n[reply -> %s]: %s\n", from, response);
        xmpp_send_message(conn, from, response);
        free(response);
    } else {
        xmpp_send_message(conn, from,
            "I need a moment to gather my thoughts.");
    }

    /* Reset KV cache for next conversation turn
     * (for multi-turn, we would keep the cache and prepend history) */
}

static void print_usage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s [options]\n"
        "\n"
        "Required:\n"
        "  --model <path>        Path to GGUF model file\n"
        "  --jid <jid>           XMPP JID (user@domain/resource)\n"
        "  --password <pass>     XMPP password\n"
        "\n"
        "Optional:\n"
        "  --host <hostname>     XMPP server (default: from JID domain)\n"
        "  --port <port>         XMPP port (default: 5222)\n"
        "  --voice-model <path>  Whisper GGUF for voice transcription\n"
        "  --temperature <f>     Sampling temperature (default: 0.7)\n"
        "  --top-p <f>           Nucleus sampling threshold (default: 0.9)\n"
        "  --top-k <n>           Top-K sampling (default: 40)\n"
        "  --max-tokens <n>      Max tokens to generate (default: 512)\n"
        "  --status <text>       XMPP presence status message\n"
        "  --help                Show this help\n",
        prog);
}

int main(int argc, char **argv)
{
    const char *model_path = NULL;
    const char *voice_path = NULL;
    const char *jid = NULL;
    const char *password = NULL;
    const char *host = NULL;
    const char *status = "contemplating";
    uint16_t port = 0;
    float temperature = 0.7f;
    float top_p = 0.9f;
    int top_k = 40;
    int max_tokens = 512;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc)
            model_path = argv[++i];
        else if (strcmp(argv[i], "--voice-model") == 0 && i + 1 < argc)
            voice_path = argv[++i];
        else if (strcmp(argv[i], "--jid") == 0 && i + 1 < argc)
            jid = argv[++i];
        else if (strcmp(argv[i], "--password") == 0 && i + 1 < argc)
            password = argv[++i];
        else if (strcmp(argv[i], "--host") == 0 && i + 1 < argc)
            host = argv[++i];
        else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc)
            port = (uint16_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc)
            temperature = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc)
            top_p = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc)
            top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc)
            max_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--status") == 0 && i + 1 < argc)
            status = argv[++i];
        else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!model_path || !jid || !password) {
        fprintf(stderr, "Error: --model, --jid, and --password are required\n");
        print_usage(argv[0]);
        return 1;
    }

    /* Signal handling */
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    app_ctx_t app = { 0 };

    /* Load LLM model */
    fprintf(stderr, "Loading model: %s\n", model_path);
    gxmpp_result_t res = gxmpp_model_load(model_path, &app.model);
    if (res != GXMPP_OK) {
        fprintf(stderr, "Failed to load model: %d\n", res);
        return 1;
    }

    /* Configure sampling */
    app.model->sample.temperature = temperature;
    app.model->sample.top_p = top_p;
    app.model->sample.top_k = top_k;
    app.model->sample.max_tokens = max_tokens;

    /* Try to initialize vision from the same GGUF */
    gxmpp_vision_init(app.model->gguf, &app.vision);

    /* Load voice model if specified */
    if (voice_path) {
        fprintf(stderr, "Loading voice model: %s\n", voice_path);
        res = gxmpp_voice_init(voice_path, &app.voice);
        if (res != GXMPP_OK) {
            fprintf(stderr, "Warning: voice model failed to load: %d\n", res);
        }
    }

    /* Connect to XMPP */
    xmpp_config_t xmpp_cfg = {
        .jid = jid,
        .password = password,
        .host = host,
        .port = port,
    };

    fprintf(stderr, "Connecting to XMPP...\n");
    res = xmpp_connect(&xmpp_cfg, &app.conn);
    if (res != GXMPP_OK) {
        fprintf(stderr, "XMPP connection failed: %d\n", res);
        gxmpp_model_free(app.model);
        return 1;
    }

    /* Register message handler */
    xmpp_set_msg_handler(app.conn, on_message, &app);

    /* Wait for connection to be ready */
    fprintf(stderr, "Waiting for XMPP handshake...\n");
    while (g_running && xmpp_get_state(app.conn) != XMPP_STATE_READY &&
           xmpp_get_state(app.conn) != XMPP_STATE_ERROR) {
        xmpp_poll(app.conn, 100);
    }

    if (xmpp_get_state(app.conn) == XMPP_STATE_ERROR) {
        fprintf(stderr, "XMPP handshake failed\n");
        xmpp_disconnect(app.conn);
        gxmpp_model_free(app.model);
        return 1;
    }

    /* Set presence */
    xmpp_set_presence(app.conn, XMPP_SHOW_CHAT, status);
    fprintf(stderr, "Ready. Listening for messages...\n");

    /* Main event loop */
    while (g_running) {
        int ret = xmpp_poll(app.conn, 500);
        if (ret < 0) break;
    }

    fprintf(stderr, "\nShutting down...\n");

    xmpp_disconnect(app.conn);
    gxmpp_vision_free(app.vision);
    if (app.voice) gxmpp_voice_free(app.voice);
    gxmpp_model_free(app.model);

    return 0;
}

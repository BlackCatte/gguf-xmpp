/*
 * Minimal XMPP client implementation
 *
 * Connects via TLS to an XMPP server, performs SASL PLAIN
 * authentication, resource binding, and handles message/presence
 * stanzas. Uses our minimal XML stream parser (xml.c).
 */

#include "xmpp/xmpp.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <arpa/inet.h>
#include <openssl/ssl.h>
#include <openssl/err.h>
#include <openssl/bio.h>
#include <openssl/evp.h>

/* From xml.c */
typedef struct { char *name; char *value; } xml_attr_t;
typedef enum {
    XML_EVENT_OPEN, XML_EVENT_CLOSE,
    XML_EVENT_TEXT, XML_EVENT_COMPLETE,
} xml_event_type_t;
typedef void (*xml_event_cb)(xml_event_type_t, const char *,
                             const xml_attr_t *, int, const char *, void *);
typedef struct {
    char buf[65536];
    int  buf_len;
    int  depth;
    xml_event_cb callback;
    void *userdata;
} xml_parser_t;

extern void xml_parser_init(xml_parser_t *, xml_event_cb, void *);
extern void xml_parser_feed(xml_parser_t *, const char *, int);
extern void xml_parser_reset(xml_parser_t *);

#define XMPP_BUF_SIZE 65536
#define XMPP_DEFAULT_PORT 5222

/* Internal connection structure */
struct xmpp_conn {
    /* Network */
    int             sock;
    SSL_CTX        *ssl_ctx;
    SSL            *ssl;

    /* XMPP state */
    xmpp_state_t    state;
    char           *jid;
    char           *user;
    char           *domain;
    char           *resource;
    char           *password;
    char           *host;
    uint16_t        port;
    char           *bound_jid;

    /* XML parser */
    xml_parser_t    parser;

    /* Stanza accumulation */
    char           *stanza_tag;
    char            stanza_body[XMPP_BUF_SIZE];
    int             stanza_depth;
    int             in_stanza;

    /* Stream features tracking */
    int             features_tls;
    int             features_sasl;
    int             features_bind;
    int             tls_done;
    int             auth_done;

    /* Handlers */
    xmpp_msg_handler_t      msg_handler;
    void                   *msg_userdata;
    xmpp_presence_handler_t presence_handler;
    void                   *presence_userdata;
    xmpp_file_handler_t     file_handler;
    void                   *file_userdata;

    /* Event loop control */
    volatile int    running;
};

/* ---- Utility ---- */

static char *base64_encode(const void *data, size_t len)
{
    size_t out_len = 4 * ((len + 2) / 3) + 1;
    char *out = malloc(out_len);
    if (!out) return NULL;

    EVP_EncodeBlock((unsigned char *)out, data, (int)len);
    return out;
}

static void parse_jid(const char *jid, char **user, char **domain,
                      char **resource)
{
    *user = *domain = *resource = NULL;

    const char *at = strchr(jid, '@');
    const char *slash = strchr(jid, '/');

    if (at) {
        *user = strndup(jid, at - jid);
        if (slash)
            *domain = strndup(at + 1, slash - at - 1);
        else
            *domain = strdup(at + 1);
    } else {
        if (slash)
            *domain = strndup(jid, slash - jid);
        else
            *domain = strdup(jid);
    }

    if (slash)
        *resource = strdup(slash + 1);
    else
        *resource = strdup("gxmpp");
}

/* ---- TLS I/O ---- */

static int tls_send(xmpp_conn_t *conn, const char *data, int len)
{
    if (conn->ssl)
        return SSL_write(conn->ssl, data, len);
    return write(conn->sock, data, len);
}

static int tls_recv(xmpp_conn_t *conn, char *buf, int len)
{
    if (conn->ssl)
        return SSL_read(conn->ssl, buf, len);
    return read(conn->sock, buf, len);
}

static void send_raw(xmpp_conn_t *conn, const char *fmt, ...)
{
    char buf[XMPP_BUF_SIZE];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);

    if (n > 0)
        tls_send(conn, buf, n);
}

/* ---- Stream initiation ---- */

static void send_stream_header(xmpp_conn_t *conn)
{
    send_raw(conn,
        "<?xml version='1.0'?>"
        "<stream:stream "
        "to='%s' "
        "xmlns='jabber:client' "
        "xmlns:stream='http://etherx.jabber.org/streams' "
        "version='1.0'>",
        conn->domain);
}

/* ---- STARTTLS ---- */

static gxmpp_result_t do_starttls(xmpp_conn_t *conn)
{
    send_raw(conn,
        "<starttls xmlns='urn:ietf:params:xml:ns:xmpp-tls'/>");

    /* Read response — expect <proceed/> */
    char buf[4096];
    int n = read(conn->sock, buf, sizeof(buf) - 1);
    if (n <= 0) return GXMPP_ERR_TLS;
    buf[n] = '\0';

    if (!strstr(buf, "proceed")) {
        fprintf(stderr, "xmpp: starttls rejected\n");
        return GXMPP_ERR_TLS;
    }

    /* Upgrade to TLS */
    conn->ssl_ctx = SSL_CTX_new(TLS_client_method());
    if (!conn->ssl_ctx) return GXMPP_ERR_TLS;

    conn->ssl = SSL_new(conn->ssl_ctx);
    SSL_set_fd(conn->ssl, conn->sock);
    SSL_set_tlsext_host_name(conn->ssl, conn->domain);

    if (SSL_connect(conn->ssl) != 1) {
        ERR_print_errors_fp(stderr);
        return GXMPP_ERR_TLS;
    }

    conn->tls_done = 1;
    conn->state = XMPP_STATE_TLS;

    /* Restart stream */
    xml_parser_reset(&conn->parser);
    send_stream_header(conn);

    return GXMPP_OK;
}

/* ---- SASL PLAIN authentication ---- */

static gxmpp_result_t do_sasl_plain(xmpp_conn_t *conn)
{
    /* SASL PLAIN: \0user\0password */
    size_t user_len = strlen(conn->user);
    size_t pass_len = strlen(conn->password);
    size_t auth_len = 1 + user_len + 1 + pass_len;
    char *auth_data = calloc(1, auth_len);

    memcpy(auth_data + 1, conn->user, user_len);
    memcpy(auth_data + 2 + user_len, conn->password, pass_len);

    char *b64 = base64_encode(auth_data, auth_len);
    free(auth_data);

    send_raw(conn,
        "<auth xmlns='urn:ietf:params:xml:ns:xmpp-sasl' "
        "mechanism='PLAIN'>%s</auth>", b64);
    free(b64);

    conn->state = XMPP_STATE_AUTHENTICATING;
    return GXMPP_OK;
}

/* ---- Resource binding ---- */

static void do_bind(xmpp_conn_t *conn)
{
    send_raw(conn,
        "<iq type='set' id='bind1'>"
        "<bind xmlns='urn:ietf:params:xml:ns:xmpp-bind'>"
        "<resource>%s</resource>"
        "</bind></iq>",
        conn->resource);
}

static void do_session(xmpp_conn_t *conn)
{
    send_raw(conn,
        "<iq type='set' id='session1'>"
        "<session xmlns='urn:ietf:params:xml:ns:xmpp-session'/>"
        "</iq>");
}

/* ---- XML event handler ---- */

static const char *find_attr(const xml_attr_t *attrs, int n, const char *name)
{
    for (int i = 0; i < n; i++) {
        if (strcmp(attrs[i].name, name) == 0)
            return attrs[i].value;
    }
    return NULL;
}

static void handle_message(xmpp_conn_t *conn, const char *from,
                           const char *body)
{
    if (conn->msg_handler && body && body[0])
        conn->msg_handler(conn, from, body, conn->msg_userdata);
}

static void xml_event(xml_event_type_t type, const char *tag,
                      const xml_attr_t *attrs, int n_attrs,
                      const char *text, void *userdata)
{
    xmpp_conn_t *conn = userdata;

    if (type == XML_EVENT_OPEN) {
        /* Track stream features */
        if (tag && strcmp(tag, "starttls") == 0)
            conn->features_tls = 1;
        if (tag && strcmp(tag, "mechanisms") == 0)
            conn->features_sasl = 1;
        if (tag && strcmp(tag, "bind") == 0)
            conn->features_bind = 1;

        /* Stanza tracking for message/presence/iq */
        if (tag && (strcmp(tag, "message") == 0 ||
                    strcmp(tag, "presence") == 0 ||
                    strcmp(tag, "iq") == 0)) {
            conn->in_stanza = 1;
            conn->stanza_depth = 0;
            free(conn->stanza_tag);
            conn->stanza_tag = strdup(tag);
            conn->stanza_body[0] = '\0';

            /* Store 'from' attribute in stanza_body header */
            const char *from = find_attr(attrs, n_attrs, "from");
            if (from) {
                snprintf(conn->stanza_body, XMPP_BUF_SIZE, "from=%s\n", from);
            }
        } else if (conn->in_stanza) {
            conn->stanza_depth++;
            if (tag && strcmp(tag, "body") == 0) {
                /* Next text event is the message body */
            }
        }
    }

    if (type == XML_EVENT_TEXT && conn->in_stanza) {
        /* Accumulate text content */
        size_t cur_len = strlen(conn->stanza_body);
        size_t txt_len = text ? strlen(text) : 0;
        if (cur_len + txt_len + 16 < XMPP_BUF_SIZE) {
            strncat(conn->stanza_body, text, XMPP_BUF_SIZE - cur_len - 1);
        }
    }

    if (type == XML_EVENT_CLOSE) {
        /* Handle authentication result */
        if (tag && strcmp(tag, "success") == 0 &&
            conn->state == XMPP_STATE_AUTHENTICATING) {
            conn->auth_done = 1;
            xml_parser_reset(&conn->parser);
            send_stream_header(conn);
            return;
        }

        if (tag && strcmp(tag, "failure") == 0 &&
            conn->state == XMPP_STATE_AUTHENTICATING) {
            fprintf(stderr, "xmpp: authentication failed\n");
            conn->state = XMPP_STATE_ERROR;
            return;
        }

        /* Handle stream features completion */
        if (tag && strcmp(tag, "stream:features") == 0) {
            if (conn->features_tls && !conn->tls_done) {
                do_starttls(conn);
            } else if (conn->features_sasl && !conn->auth_done) {
                do_sasl_plain(conn);
            } else if (conn->features_bind) {
                do_bind(conn);
            }
            conn->features_tls = 0;
            conn->features_sasl = 0;
            conn->features_bind = 0;
            return;
        }

        /* Stanza completion */
        if (conn->in_stanza) {
            if (conn->stanza_depth > 0) {
                conn->stanza_depth--;
            } else {
                /* Stanza complete */
                conn->in_stanza = 0;

                if (conn->stanza_tag &&
                    strcmp(conn->stanza_tag, "message") == 0) {
                    /* Extract from and body from accumulated text */
                    char *from = NULL;
                    char *body = NULL;
                    char *line = conn->stanza_body;

                    if (strncmp(line, "from=", 5) == 0) {
                        char *nl = strchr(line, '\n');
                        if (nl) {
                            from = strndup(line + 5, nl - line - 5);
                            body = nl + 1;
                        }
                    } else {
                        body = line;
                    }

                    handle_message(conn, from ? from : "", body ? body : "");
                    free(from);
                }

                if (conn->stanza_tag &&
                    strcmp(conn->stanza_tag, "iq") == 0) {
                    /* Check for bind result */
                    char *jid_start = strstr(conn->stanza_body, "<jid>");
                    if (jid_start) {
                        jid_start += 5;
                        char *jid_end = strstr(jid_start, "</jid>");
                        if (jid_end) {
                            free(conn->bound_jid);
                            conn->bound_jid = strndup(jid_start,
                                                      jid_end - jid_start);
                            fprintf(stderr, "xmpp: bound as %s\n",
                                    conn->bound_jid);
                            do_session(conn);
                            conn->state = XMPP_STATE_READY;
                        }
                    }
                }
            }
        }
    }

    if (type == XML_EVENT_COMPLETE) {
        /* Self-closing elements in stream features */
        if (tag && strcmp(tag, "starttls") == 0)
            conn->features_tls = 1;
        if (tag && strcmp(tag, "bind") == 0)
            conn->features_bind = 1;
    }
}

/* ---- Public API ---- */

gxmpp_result_t xmpp_connect(const xmpp_config_t *config, xmpp_conn_t **out)
{
    xmpp_conn_t *conn = calloc(1, sizeof(xmpp_conn_t));
    if (!conn) return GXMPP_ERR_MEMORY;

    conn->sock = -1;
    conn->state = XMPP_STATE_DISCONNECTED;

    /* Parse JID */
    conn->jid = strdup(config->jid);
    parse_jid(config->jid, &conn->user, &conn->domain, &conn->resource);
    conn->password = strdup(config->password);
    conn->host = strdup(config->host ? config->host : conn->domain);
    conn->port = config->port ? config->port : XMPP_DEFAULT_PORT;

    fprintf(stderr, "xmpp: connecting to %s:%u as %s@%s/%s\n",
            conn->host, conn->port, conn->user, conn->domain, conn->resource);

    /* TCP connection */
    struct addrinfo hints = { .ai_family = AF_UNSPEC, .ai_socktype = SOCK_STREAM };
    struct addrinfo *res;
    char port_str[8];
    snprintf(port_str, sizeof(port_str), "%u", conn->port);

    int gai = getaddrinfo(conn->host, port_str, &hints, &res);
    if (gai != 0) {
        fprintf(stderr, "xmpp: DNS lookup failed: %s\n", gai_strerror(gai));
        xmpp_disconnect(conn);
        return GXMPP_ERR_NET;
    }

    conn->sock = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (conn->sock < 0) {
        freeaddrinfo(res);
        xmpp_disconnect(conn);
        return GXMPP_ERR_NET;
    }

    if (connect(conn->sock, res->ai_addr, res->ai_addrlen) < 0) {
        fprintf(stderr, "xmpp: connect failed: %s\n", strerror(errno));
        freeaddrinfo(res);
        xmpp_disconnect(conn);
        return GXMPP_ERR_NET;
    }
    freeaddrinfo(res);

    conn->state = XMPP_STATE_CONNECTING;

    /* Initialize XML parser */
    xml_parser_init(&conn->parser, xml_event, conn);

    /* Send initial stream header */
    send_stream_header(conn);

    *out = conn;
    return GXMPP_OK;
}

void xmpp_disconnect(xmpp_conn_t *conn)
{
    if (!conn) return;

    if (conn->state >= XMPP_STATE_CONNECTING && conn->sock >= 0) {
        send_raw(conn, "</stream:stream>");
    }

    if (conn->ssl) {
        SSL_shutdown(conn->ssl);
        SSL_free(conn->ssl);
    }
    if (conn->ssl_ctx) SSL_CTX_free(conn->ssl_ctx);
    if (conn->sock >= 0) close(conn->sock);

    free(conn->jid);
    free(conn->user);
    free(conn->domain);
    free(conn->resource);
    free(conn->password);
    free(conn->host);
    free(conn->bound_jid);
    free(conn->stanza_tag);
    free(conn);
}

xmpp_state_t xmpp_get_state(const xmpp_conn_t *conn)
{
    return conn->state;
}

gxmpp_result_t xmpp_set_presence(xmpp_conn_t *conn, xmpp_show_t show,
                                 const char *status)
{
    const char *show_str[] = {
        NULL, "chat", "away", "xa", "dnd"
    };

    if (show == XMPP_SHOW_AVAILABLE) {
        if (status)
            send_raw(conn, "<presence><status>%s</status></presence>", status);
        else
            send_raw(conn, "<presence/>");
    } else {
        send_raw(conn,
            "<presence><show>%s</show>%s%s%s</presence>",
            show_str[show],
            status ? "<status>" : "",
            status ? status : "",
            status ? "</status>" : "");
    }
    return GXMPP_OK;
}

gxmpp_result_t xmpp_send_message(xmpp_conn_t *conn, const char *to,
                                 const char *body)
{
    send_raw(conn,
        "<message type='chat' to='%s'>"
        "<body>%s</body>"
        "</message>",
        to, body);
    return GXMPP_OK;
}

gxmpp_result_t xmpp_send_file(xmpp_conn_t *conn, const char *to,
                              const char *filename, const void *data,
                              size_t len)
{
    char *b64 = base64_encode(data, len);
    if (!b64) return GXMPP_ERR_MEMORY;

    send_raw(conn,
        "<message type='chat' to='%s'>"
        "<body>[file: %s]</body>"
        "<x xmlns='gxmpp:file' name='%s'>%s</x>"
        "</message>",
        to, filename, filename, b64);
    free(b64);
    return GXMPP_OK;
}

void xmpp_set_msg_handler(xmpp_conn_t *conn, xmpp_msg_handler_t handler,
                          void *userdata)
{
    conn->msg_handler = handler;
    conn->msg_userdata = userdata;
}

void xmpp_set_presence_handler(xmpp_conn_t *conn,
                               xmpp_presence_handler_t handler,
                               void *userdata)
{
    conn->presence_handler = handler;
    conn->presence_userdata = userdata;
}

void xmpp_set_file_handler(xmpp_conn_t *conn, xmpp_file_handler_t handler,
                           void *userdata)
{
    conn->file_handler = handler;
    conn->file_userdata = userdata;
}

int xmpp_poll(xmpp_conn_t *conn, int timeout_ms)
{
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(conn->sock, &rfds);

    struct timeval tv = {
        .tv_sec = timeout_ms / 1000,
        .tv_usec = (timeout_ms % 1000) * 1000,
    };

    int ret = select(conn->sock + 1, &rfds, NULL, NULL, &tv);
    if (ret <= 0) return 0;

    char buf[XMPP_BUF_SIZE];
    int n = tls_recv(conn, buf, sizeof(buf) - 1);
    if (n <= 0) {
        conn->state = XMPP_STATE_ERROR;
        return -1;
    }
    buf[n] = '\0';

    xml_parser_feed(&conn->parser, buf, n);
    return 1;
}

gxmpp_result_t xmpp_run(xmpp_conn_t *conn)
{
    conn->running = 1;
    while (conn->running && conn->state != XMPP_STATE_ERROR) {
        int ret = xmpp_poll(conn, 100);
        if (ret < 0) break;
    }
    return conn->state == XMPP_STATE_ERROR ? GXMPP_ERR_NET : GXMPP_OK;
}

void xmpp_stop(xmpp_conn_t *conn)
{
    conn->running = 0;
}

/*
 * Minimal XMPP client
 *
 * Implements just enough of RFC 6120/6121 for:
 *   - TLS connection to an XMPP server
 *   - SASL PLAIN authentication
 *   - Presence management
 *   - Message send/receive
 *   - Basic stanza parsing (no full XML DOM - stream parser only)
 */

#ifndef GXMPP_XMPP_H
#define GXMPP_XMPP_H

#include "core/types.h"

/* Connection state */
typedef enum {
    XMPP_STATE_DISCONNECTED,
    XMPP_STATE_CONNECTING,
    XMPP_STATE_TLS,
    XMPP_STATE_AUTHENTICATING,
    XMPP_STATE_BOUND,
    XMPP_STATE_READY,
    XMPP_STATE_ERROR,
} xmpp_state_t;

/* Presence show values */
typedef enum {
    XMPP_SHOW_AVAILABLE,
    XMPP_SHOW_CHAT,
    XMPP_SHOW_AWAY,
    XMPP_SHOW_XA,
    XMPP_SHOW_DND,
} xmpp_show_t;

/* XMPP connection configuration */
typedef struct {
    const char *jid;          /* user@domain/resource */
    const char *password;
    const char *host;         /* server hostname (NULL = derive from JID) */
    uint16_t    port;         /* 0 = default (5222) */
} xmpp_config_t;

/* Opaque connection handle */
typedef struct xmpp_conn xmpp_conn_t;

/* Message handler callback */
typedef void (*xmpp_msg_handler_t)(xmpp_conn_t *conn,
                                   const char *from,
                                   const char *body,
                                   void *userdata);

/* Presence handler callback */
typedef void (*xmpp_presence_handler_t)(xmpp_conn_t *conn,
                                       const char *from,
                                       xmpp_show_t show,
                                       const char *status,
                                       void *userdata);

/* File transfer receive callback (for vision/media) */
typedef void (*xmpp_file_handler_t)(xmpp_conn_t *conn,
                                    const char *from,
                                    const char *filename,
                                    const void *data,
                                    size_t len,
                                    void *userdata);

/* Connect to XMPP server */
gxmpp_result_t xmpp_connect(const xmpp_config_t *config, xmpp_conn_t **conn);

/* Disconnect and free */
void xmpp_disconnect(xmpp_conn_t *conn);

/* Get current connection state */
xmpp_state_t xmpp_get_state(const xmpp_conn_t *conn);

/* Set presence */
gxmpp_result_t xmpp_set_presence(xmpp_conn_t *conn, xmpp_show_t show,
                                 const char *status);

/* Send a message to a JID */
gxmpp_result_t xmpp_send_message(xmpp_conn_t *conn, const char *to,
                                 const char *body);

/* Send a file (for vision: images) using in-band bytestreams */
gxmpp_result_t xmpp_send_file(xmpp_conn_t *conn, const char *to,
                              const char *filename, const void *data,
                              size_t len);

/* Register handlers */
void xmpp_set_msg_handler(xmpp_conn_t *conn, xmpp_msg_handler_t handler,
                          void *userdata);
void xmpp_set_presence_handler(xmpp_conn_t *conn,
                               xmpp_presence_handler_t handler,
                               void *userdata);
void xmpp_set_file_handler(xmpp_conn_t *conn, xmpp_file_handler_t handler,
                           void *userdata);

/* Run the event loop (blocking). Returns on disconnect or error. */
gxmpp_result_t xmpp_run(xmpp_conn_t *conn);

/* Request the event loop to stop */
void xmpp_stop(xmpp_conn_t *conn);

/* Process events for a given timeout (ms). Returns number of events. */
int xmpp_poll(xmpp_conn_t *conn, int timeout_ms);

#endif /* GXMPP_XMPP_H */

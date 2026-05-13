/*
 * Minimal XML stream parser for XMPP
 *
 * This is NOT a general-purpose XML parser. It handles only the
 * subset needed for XMPP stanzas: elements, attributes, text content,
 * and namespaces. No DTDs, no CDATA, no processing instructions.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define XML_MAX_DEPTH    16
#define XML_MAX_ATTRS    16
#define XML_BUF_SIZE     65536

typedef struct {
    char *name;
    char *value;
} xml_attr_t;

typedef struct xml_node {
    char            *tag;
    char            *ns;
    xml_attr_t       attrs[XML_MAX_ATTRS];
    int              n_attrs;
    char            *text;
    struct xml_node *children;
    struct xml_node *next;
    struct xml_node *parent;
} xml_node_t;

typedef enum {
    XML_EVENT_OPEN,
    XML_EVENT_CLOSE,
    XML_EVENT_TEXT,
    XML_EVENT_COMPLETE,   /* self-closing element */
} xml_event_type_t;

typedef void (*xml_event_cb)(xml_event_type_t type, const char *tag,
                             const xml_attr_t *attrs, int n_attrs,
                             const char *text, void *userdata);

typedef struct {
    char          buf[XML_BUF_SIZE];
    int           buf_len;
    int           depth;
    xml_event_cb  callback;
    void         *userdata;
} xml_parser_t;

void xml_parser_init(xml_parser_t *p, xml_event_cb cb, void *userdata)
{
    memset(p, 0, sizeof(*p));
    p->callback = cb;
    p->userdata = userdata;
}

static void skip_ws(const char **s)
{
    while (**s && isspace((unsigned char)**s)) (*s)++;
}

static char *read_name(const char **s)
{
    const char *start = *s;
    while (**s && !isspace((unsigned char)**s) &&
           **s != '=' && **s != '>' && **s != '/' && **s != '\0')
        (*s)++;
    int len = *s - start;
    char *name = malloc(len + 1);
    memcpy(name, start, len);
    name[len] = '\0';
    return name;
}

static char *read_quoted(const char **s)
{
    char quote = **s;
    if (quote != '"' && quote != '\'') return strdup("");
    (*s)++;
    const char *start = *s;
    while (**s && **s != quote) (*s)++;
    int len = *s - start;
    char *val = malloc(len + 1);
    memcpy(val, start, len);
    val[len] = '\0';
    if (**s == quote) (*s)++;
    return val;
}

static int parse_attrs(const char **s, xml_attr_t *attrs, int max)
{
    int n = 0;
    while (n < max) {
        skip_ws(s);
        if (**s == '>' || **s == '/' || **s == '?' || **s == '\0')
            break;
        attrs[n].name = read_name(s);
        skip_ws(s);
        if (**s == '=') {
            (*s)++;
            skip_ws(s);
            attrs[n].value = read_quoted(s);
        } else {
            attrs[n].value = strdup("");
        }
        n++;
    }
    return n;
}

void xml_parser_feed(xml_parser_t *p, const char *data, int len)
{
    /* Append to buffer */
    int space = XML_BUF_SIZE - p->buf_len - 1;
    if (len > space) len = space;
    memcpy(p->buf + p->buf_len, data, len);
    p->buf_len += len;
    p->buf[p->buf_len] = '\0';

    /* Process complete elements */
    char *pos = p->buf;
    while (*pos) {
        char *lt = strchr(pos, '<');
        if (!lt) break;

        /* Text content before tag */
        if (lt > pos) {
            int tlen = lt - pos;
            char *text = malloc(tlen + 1);
            memcpy(text, pos, tlen);
            text[tlen] = '\0';

            /* Only emit if non-whitespace */
            int has_content = 0;
            for (int i = 0; i < tlen; i++) {
                if (!isspace((unsigned char)text[i])) {
                    has_content = 1;
                    break;
                }
            }
            if (has_content && p->callback)
                p->callback(XML_EVENT_TEXT, NULL, NULL, 0, text, p->userdata);
            free(text);
        }

        /* Find closing '>' */
        char *gt = strchr(lt + 1, '>');
        if (!gt) break;  /* incomplete tag, wait for more data */

        /* Process tag */
        int tag_len = gt - lt - 1;
        char *tag_buf = malloc(tag_len + 1);
        memcpy(tag_buf, lt + 1, tag_len);
        tag_buf[tag_len] = '\0';

        if (tag_buf[0] == '/') {
            /* Closing tag */
            const char *tp = tag_buf + 1;
            skip_ws(&tp);
            char *tag_name = read_name(&tp);
            if (p->callback)
                p->callback(XML_EVENT_CLOSE, tag_name, NULL, 0, NULL,
                           p->userdata);
            p->depth--;
            free(tag_name);
        } else if (tag_buf[0] == '?') {
            /* XML declaration — skip */
        } else {
            /* Opening or self-closing tag */
            int self_closing = (tag_buf[tag_len - 1] == '/');
            if (self_closing) tag_buf[tag_len - 1] = '\0';

            const char *tp = tag_buf;
            char *tag_name = read_name(&tp);
            xml_attr_t attrs[XML_MAX_ATTRS];
            int n_attrs = parse_attrs(&tp, attrs, XML_MAX_ATTRS);

            if (self_closing) {
                if (p->callback)
                    p->callback(XML_EVENT_COMPLETE, tag_name, attrs, n_attrs,
                               NULL, p->userdata);
            } else {
                if (p->callback)
                    p->callback(XML_EVENT_OPEN, tag_name, attrs, n_attrs,
                               NULL, p->userdata);
                p->depth++;
            }

            free(tag_name);
            for (int i = 0; i < n_attrs; i++) {
                free(attrs[i].name);
                free(attrs[i].value);
            }
        }

        free(tag_buf);
        pos = gt + 1;
    }

    /* Compact buffer */
    int remaining = p->buf + p->buf_len - pos;
    if (remaining > 0 && pos != p->buf) {
        memmove(p->buf, pos, remaining);
    }
    p->buf_len = remaining;
    p->buf[p->buf_len] = '\0';
}

void xml_parser_reset(xml_parser_t *p)
{
    p->buf_len = 0;
    p->depth = 0;
}

CC      ?= gcc
CFLAGS  ?= -O2 -Wall -Wextra -std=c11 -D_POSIX_C_SOURCE=200809L
LDFLAGS ?=
LIBS    = -lm -lssl -lcrypto

PREFIX  ?= /usr/local
BINDIR  = $(PREFIX)/bin

SRC_DIR = src
INC_DIR = include
BUILD   = build

SRCS = \
	$(SRC_DIR)/core/mem.c \
	$(SRC_DIR)/gguf/gguf.c \
	$(SRC_DIR)/inference/quant.c \
	$(SRC_DIR)/inference/inference.c \
	$(SRC_DIR)/xmpp/xml.c \
	$(SRC_DIR)/xmpp/xmpp.c \
	$(SRC_DIR)/vision/vision.c \
	$(SRC_DIR)/voice/voice.c \
	$(SRC_DIR)/main.c

OBJS = $(SRCS:$(SRC_DIR)/%.c=$(BUILD)/%.o)

TARGET = $(BUILD)/gguf-xmpp

.PHONY: all clean install

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)

$(BUILD)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -I$(INC_DIR) -c -o $@ $<

clean:
	rm -rf $(BUILD)

install: $(TARGET)
	install -d $(BINDIR)
	install -m 755 $(TARGET) $(BINDIR)/gguf-xmpp

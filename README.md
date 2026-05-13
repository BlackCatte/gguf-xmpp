# gguf-xmpp

A lightweight, framework-free bridge between GGUF language models and XMPP messaging. Built for unhurried, thoughtful AI conversations where time is no factor.

## Philosophy

Modern AI chat tools optimize for speed — instant responses, streaming tokens, rapid-fire exchanges. **gguf-xmpp** takes the opposite approach: it treats AI conversation as something worth sitting with. Using XMPP's asynchronous, federated messaging protocol, it lets you find and converse with AI partners that match your guidelines, at whatever pace feels right.

No Ollama. No llama.cpp. No heavyweight frameworks. Just raw GGUF model parsing, hand-written transformer inference, and a minimal XMPP client — all in C.

## Features

- **Direct GGUF inference** — Parses GGUF files and runs transformer forward passes without any external inference framework
- **Quantization support** — Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, F16, F32 dequantization
- **LLaMA-family architecture** — Grouped-query attention, RoPE, RMSNorm, SwiGLU FFN
- **XMPP messaging** — Minimal client with TLS, SASL PLAIN auth, presence, and message handling
- **Vision** — CLIP-style image encoding for multimodal GGUF models (LLaVA-family)
- **Voice** — Mel spectrogram computation and Whisper-style transcription pipeline (GGUF)
- **Zero dependencies beyond libc** — Only requires OpenSSL for TLS (a system library on most platforms)

## Building

```
make
```

Requires:
- A C11 compiler (gcc or clang)
- OpenSSL development headers (`libssl-dev` / `openssl-devel`)
- Standard POSIX (Linux, macOS, BSD)

## Usage

```
./build/gguf-xmpp \
    --model /path/to/model.gguf \
    --jid mybot@xmpp.example.com/thoughtful \
    --password secret \
    --host xmpp.example.com \
    --status "contemplating" \
    --temperature 0.7 \
    --max-tokens 512
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Path to GGUF model file | (required) |
| `--jid` | XMPP JID (user@domain/resource) | (required) |
| `--password` | XMPP account password | (required) |
| `--host` | XMPP server hostname | from JID |
| `--port` | XMPP server port | 5222 |
| `--voice-model` | Whisper GGUF for transcription | (none) |
| `--temperature` | Sampling temperature | 0.7 |
| `--top-p` | Nucleus sampling | 0.9 |
| `--top-k` | Top-K sampling | 40 |
| `--max-tokens` | Max tokens per response | 512 |
| `--status` | XMPP presence status | contemplating |

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  XMPP       │────▶│  Inference   │────▶│  GGUF Parser    │
│  Client     │     │  Engine      │     │  (mmap tensors) │
│  (TLS+SASL) │◀────│  (forward    │     └─────────────────┘
└─────────────┘     │   pass)      │
      │             └──────────────┘
      │                    │
      ▼                    ▼
┌──────────┐      ┌──────────────┐
│  Voice   │      │  Vision      │
│  (Mel +  │      │  (CLIP       │
│  Whisper)│      │  encoder)    │
└──────────┘      └──────────────┘
```

### Components

- **`src/gguf/`** — GGUF binary format parser. Memory-maps tensor data for zero-copy access. Handles metadata extraction (hyperparameters, tokenizer vocabulary, architecture info).

- **`src/inference/`** — Transformer forward pass and sampling. Implements RMSNorm, multi-head attention with GQA, RoPE positional embeddings, SwiGLU FFN, KV cache, and top-k/top-p/temperature sampling.

- **`src/xmpp/`** — Minimal XMPP client. Hand-rolled XML stream parser (not a general XML parser — just enough for XMPP stanzas). TLS via OpenSSL, SASL PLAIN authentication, presence management, message send/receive.

- **`src/vision/`** — Image decoding (BMP native, PNG/JPEG planned) and CLIP vision encoder. Patch embedding, bilinear resize, ImageNet normalization, projection to LLM embedding space.

- **`src/voice/`** — Mel spectrogram computation (Hann window, DFT, mel filterbank). Whisper-style encoder-decoder transcription scaffold. Raw PCM audio input (Opus decoding planned).

- **`src/core/`** — Arena allocator for per-inference scratch memory, aligned allocation for SIMD-friendly tensor operations.

## How It Works

1. The program loads a GGUF model file, parsing its metadata and memory-mapping the tensor data
2. It connects to an XMPP server using TLS and authenticates via SASL PLAIN
3. It sets presence (showing as "available" with a custom status) and waits for messages
4. When a message arrives, it tokenizes the text, runs the full transformer forward pass, samples tokens autoregressively, and sends the response back via XMPP
5. For vision-capable models, image attachments are decoded, encoded through the CLIP pipeline, and injected as embeddings alongside the text prompt
6. For voice, audio is converted to mel spectrograms and passed through a Whisper-style transcription model

## Supported Models

Any GGUF model following the LLaMA architecture family:
- LLaMA / LLaMA 2 / LLaMA 3
- Mistral / Mixtral
- Phi
- Qwen
- LLaVA (text + vision)

Whisper GGUF models for voice transcription.

## Current Status

This is a working prototype. The core inference pipeline and XMPP integration are functional. Areas for future development:

- [ ] Full BPE tokenizer (currently greedy longest-match)
- [ ] K-quant dequantization (Q2_K through Q6_K)
- [ ] PNG/JPEG image decoding
- [ ] Opus audio codec
- [ ] Full Whisper encoder-decoder inference
- [ ] XMPP Jingle voice session negotiation
- [ ] Multi-turn conversation history management
- [ ] PubSub-based model profile discovery
- [ ] SIMD-accelerated tensor operations (AVX2/NEON)

## License

This project is released into the public domain. Do with it as you will.

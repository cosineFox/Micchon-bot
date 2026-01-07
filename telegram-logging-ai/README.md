# Telegram Logging AI with Waifu

A personal AI companion Telegram bot with unified memory, journal compilation, voice responses, and automatic fine-tuning. Designed for extreme optimization on consumer GPUs (RTX 3060 12GB).

## Overview

This bot operates in two modes:
- **Normal Mode**: An AI companion (waifu) with Senko-san personality responds to everything you say, remembering all past conversations
- **Journal Mode**: Silent logging of text and images, later compiled into AI-written articles

All interactions are stored in a unified memory system that the AI can query for context-aware responses.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA FLOW                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  JOURNAL MODE (Silent)              NORMAL MODE (Waifu Active)  │
│                                                                 │
│  /journal start                    Any message (no command)     │
│       │                                   │                     │
│       v                                   v                     │
│  ┌───────────┐                     ┌───────────────┐            │
│  │journal.db │                     │   WAIFU       │            │
│  │  (draft)  │                     │  responds     │            │
│  └───────────┘                     └───────┬───────┘            │
│       │                                    │                    │
│  /journal done                             v                    │
│       │                              ┌───────────┐              │
│       v                              │memory.db  │              │
│  ┌───────────┐                       │ (master)  │              │
│  │ AI compile│ ──────────────────────┘           │              │
│  └───────────┘                                                  │
│                                                                 │
│  journal.db cleared after transfer to memory.db                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Two-Database System

| Database | Purpose | Lifetime |
|----------|---------|----------|
| `journal.db` | Temporary draft entries during journal sessions | Cleared after compilation |
| `memory.db` | Master database - all finalized memories, journals, chat history | Permanent |

### Tech Stack

| Component | Technology | Notes |
|-----------|------------|-------|
| LLM Runtime | llama-cpp-python | 40-60% faster than Ollama |
| Main Model | Gemma-3n-E4B Q4_K_M GGUF | Vision + conversation, ~2.5GB VRAM |
| Embeddings | EmbeddingGemma-300M | 256d MRL for speed |
| TTS | Chatterbox-Turbo | 350MB, <200ms latency |
| Vector Search | sqlite-vec | SQLite extension |
| Fine-tuning | Unsloth + PEFT | LoRA adapters |
| Bot Framework | python-telegram-bot | Async-native |

## Hardware Requirements

**Minimum (tested on):**
- CPU: AMD Ryzen 7 2700X (or equivalent)
- GPU: NVIDIA RTX 3060 12GB VRAM
- RAM: 16GB
- OS: Linux (macOS may work, Windows untested)

**VRAM Usage:**
- Main model: ~2.5GB
- TTS: ~350MB
- Embeddings: ~300MB (loaded on-demand)
- Total: ~4GB (67% free on 12GB card)

## Installation

### 1. Clone the Repository

```bash
git clone <your-repo>
cd telegram-logging-ai
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**For CUDA support (NVIDIA GPU):**
```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

**For fine-tuning support (optional):**
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### 4. Download Models

Create the models directory:
```bash
mkdir -p models
```

**Main Model (Required):**
Download a Gemma-3n-E4B GGUF model (Q4_K_M quantization recommended):
- Source: [HuggingFace - huihui-ai/Huihui-gemma-3n-E4B-it-abliterated](https://huggingface.co/huihui-ai/Huihui-gemma-3n-E4B-it-abliterated)
- Look for GGUF versions or convert using `llama.cpp`
- Save as: `models/gemma-3n-e4b-q4_k_m.gguf`

**Note:** The exact model path is configured in `config.py` as `MAIN_MODEL_PATH`. Adjust if using a different filename.

### 5. Configure Environment

```bash
cp .env.example .env
```

Edit `.env`:
```bash
# Required
TELEGRAM_BOT_TOKEN=your_bot_token_from_botfather
TELEGRAM_ALLOWED_USERS=123456789  # Your Telegram user ID (comma-separated for multiple users)

# Optional (for Bluesky posting - leave blank to disable)
# NOTE: Only posting functionality is implemented, not reading from your timeline
BLUESKY_HANDLE=your.handle.bsky.social
BLUESKY_APP_PASSWORD=your_app_password
```

**Note about privacy:**
- **Telegram access**: The bot is designed to be personal, so it only responds to users in the `TELEGRAM_ALLOWED_USERS` list. This is intentional for privacy.
- **Bluesky integration**: The bot only posts to your Bluesky account (not read from it). Leave credentials blank if you don't want this functionality.

**Getting your Telegram user ID:**
1. Message [@userinfobot](https://t.me/userinfobot) on Telegram
2. It will reply with your user ID

### 6. Run the Bot

```bash
python -m bot.main
```

## Usage

### Commands

| Command | Description |
|---------|-------------|
| `/start` | Initialize bot, show help |
| `/help` | Show command reference |
| `/journal start` | Enter journal mode (silent logging) |
| `/journal done` | Compile journal entries into article |
| `/journal cancel` | Exit journal mode, discard entries |
| `/journal status` | Show current journal session info |
| `/journal preview` | Preview entries before compiling |
| `/bsky <text>` | Post to Bluesky (AI-polished) - requires credentials in .env |
| `/voice` | Toggle voice responses on/off |
| `/rate <1-5>` | Rate the last AI response |
| `/status` | Show system status |

### Normal Mode

Simply send any message (text or image) without a command. The waifu will:
1. Store your message as a memory
2. Query relevant past memories for context
3. Generate a personalized response
4. Optionally respond with voice (if enabled)

### Journal Mode

1. Start: `/journal start`
2. Send text and images - they're logged silently (✓ confirmation only)
3. End: `/journal done`
4. AI compiles entries into a cohesive article with title and tags
5. Article is saved to `data/journals/` and stored in memory

### Voice Responses

- Toggle with `/voice`
- Uses Chatterbox-Turbo TTS
- Supports paralinguistics: `[laugh]`, `[sigh]`, `[chuckle]`
- Sent as Telegram voice messages

### Rating Responses (for Fine-tuning)

- After receiving a response, use `/rate <1-5>` to rate it
- Ratings 4-5 are used for fine-tuning
- Fine-tuning runs automatically at 2 AM (configurable)

## Configuration

All settings are in `config.py`. Key options:

### Model Settings

```python
MAIN_MODEL_PATH = MODELS_DIR / "gemma-3n-e4b-q4_k_m.gguf"
EMBED_MODEL_NAME = "google/embeddinggemma-300m"
EMBEDDING_DIMENSION = 256  # MRL dimension (256=fast, 768=quality)
```

### llama.cpp Parameters

```python
LLAMA_PARAMS = {
    "n_gpu_layers": -1,     # Full GPU offload
    "n_ctx": 2048,          # Context window
    "n_batch": 512,         # Batch size
    "n_threads": 4,         # CPU threads
    "f16_kv": True,         # Half-precision KV cache
}
```

### Waifu Personality

The personality is defined in `config.py` as `WAIFU_PERSONALITY`. Current personality is Senko-san (warm, motherly, nurturing). Edit to customize.

### Fine-tuning Settings

```python
FINE_TUNE_ENABLED = True
FINE_TUNE_HOUR = 2          # Run at 2 AM
FINE_TUNE_MIN_EXAMPLES = 50 # Minimum rated examples needed
FINE_TUNE_MIN_RATING = 4    # Only use 4-5 star examples
```

### TTS Settings

```python
TTS_ENABLED = True
TTS_VOICE_SAMPLE = None     # Path to voice sample for cloning (optional)
TTS_PARALINGUISTICS = True  # Enable [laugh], [chuckle] tags
TTS_SEND_AS_VOICE = True    # Send as Telegram voice message
```

## File Structure

```
telegram-logging-ai/
├── bot/
│   ├── __init__.py
│   ├── main.py              # Entry point
│   ├── handlers.py          # Telegram command handlers
│   ├── model_manager.py     # GGUF model loading/unloading
│   ├── llama_client.py      # llama-cpp-python wrapper
│   ├── tts_client.py        # Chatterbox-Turbo TTS
│   ├── waifu.py             # AI personality + response generation
│   ├── journal_mode.py      # Session state management
│   ├── journal_compiler.py  # Draft → article compilation
│   ├── bluesky_client.py    # Bluesky integration
│   ├── fine_tuner.py        # LoRA fine-tuning
│   └── scheduler.py         # Scheduled tasks (2 AM fine-tuning)
├── memory/
│   ├── __init__.py
│   ├── models.py            # Data models (Memory, Journal, etc.)
│   ├── master_repo.py       # memory.db repository
│   ├── journal_repo.py      # journal.db repository
│   ├── embedder.py          # EmbeddingGemma wrapper
│   └── context.py           # Context builder for waifu
├── data/                    # Runtime data (gitignored)
│   ├── memory.db            # Master database
│   ├── journal.db           # Draft entries
│   ├── images/              # Stored images
│   ├── journals/            # Compiled markdown articles
│   └── exports/             # Other exports
├── models/                  # GGUF model files (gitignored)
│   ├── gemma-3n-e4b-q4_k_m.gguf
│   ├── adapters/            # Fine-tuned LoRA adapters
│   └── training/            # Training data
├── config.py                # All configuration
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (gitignored)
├── .env.example             # Environment template
└── .gitignore
```

## Known Issues & Limitations

### Model Availability
- **Gemma-3n-E4B GGUF**: May need to be converted from HuggingFace format using `llama.cpp`'s conversion tools. Pre-quantized GGUF versions may not be readily available.
- **EmbeddingGemma-300M**: Loads from HuggingFace automatically, requires ~300MB download on first run.

### TTS (Chatterbox-Turbo)
- **Package availability**: `chatterbox-tts` may not be on PyPI. Install from source:
  ```bash
  pip install git+https://github.com/resemble-ai/chatterbox.git
  ```
- **CUDA required**: TTS needs GPU acceleration for reasonable speed.
- **First run slow**: Model downloads on first use (~350MB).

### sqlite-vec
- **Installation**: May require building from source on some platforms:
  ```bash
  pip install sqlite-vec
  ```
- **Graceful fallback**: If sqlite-vec fails to load, vector search falls back to recent memories (time-based).

### Fine-tuning
- **Heavy dependencies**: Unsloth, PEFT, transformers, etc. add ~5GB of dependencies.
- **VRAM requirements**: Fine-tuning may require more VRAM than inference. Consider disabling if VRAM-constrained.
- **LoRA → GGUF**: The current implementation creates LoRA adapters but doesn't merge them back into GGUF. This is a placeholder for future enhancement.

### Vision Model
- **Image description**: The vision capability depends on the model supporting multimodal input. Gemma-3n-E4B should support this, but verify your specific GGUF includes vision.

### Platform Compatibility
- **Linux**: Primary development platform, should work.
- **macOS**: Should work with Metal acceleration (untested). Change `DLLAMA_CUBLAS` to `DLLAMA_METAL`.
- **Windows**: Untested. May have path issues.

## Troubleshooting

### "Model not found" error
```
FileNotFoundError: models/gemma-3n-e4b-q4_k_m.gguf
```
Download the GGUF model and place it in the `models/` directory.

### "CUDA out of memory"
- Reduce `n_ctx` in `LLAMA_PARAMS` (try 1024)
- Disable TTS: Set `TTS_ENABLED = False`
- Use smaller quantization (Q3_K_S instead of Q4_K_M)

### "sqlite-vec not found"
Vector search will fall back to time-based queries. To fix:
```bash
pip install sqlite-vec
```

### "Chatterbox import error"
```bash
pip install git+https://github.com/resemble-ai/chatterbox.git
```

### Bot doesn't respond
1. Check `TELEGRAM_ALLOWED_USERS` includes your user ID
2. Check bot token is correct
3. Check logs for errors

### Slow first response
Normal - model loads into VRAM on first use. Subsequent responses will be faster.

## Performance Tuning

### For faster responses:
- Reduce `n_ctx` to 1024
- Reduce `max_tokens` in `GEN_PARAMS` to 256
- Disable streaming: `ENABLE_STREAMING = False`

### For better quality:
- Increase `EMBEDDING_DIMENSION` to 768
- Increase `WAIFU_MAX_RELEVANT_MEMORIES` to 10
- Use larger quantization (Q5_K_M or Q6_K)

### For lower VRAM:
- Use smaller quantization (Q3_K_S)
- Set `TTS_ENABLED = False`
- Reduce `n_ctx` to 1024

## Contributing

This is a personal project. Feel free to fork and adapt for your own use.

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- **Senko-san** personality inspired by "The Helpful Fox Senko-san"
- **llama.cpp** for efficient GGUF inference
- **Chatterbox** by Resemble AI for TTS
- **sqlite-vec** for vector search in SQLite

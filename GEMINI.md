# Micchon / Telegram Logging AI

This workspace contains "Micchon" (also referred to as `telegram-logging-ai`), a sophisticated Telegram bot designed as a personal AI companion ("Waifu") with unified memory, journaling capabilities, and voice response features. It is optimized for consumer hardware (specifically NVIDIA RTX 3060 12GB).

## Project Overview

The system operates in two primary modes:
1.  **Normal Mode (Waifu):** An always-on AI companion with a specific personality (e.g., Senko-san). It remembers past conversations, posts to Bluesky, and can speak using TTS.
2.  **Journal Mode:** A silent logging tool. Users capture text and images which are later compiled by the AI into cohesive Markdown articles and stored in the permanent memory.

### Architecture
- **Unified Memory:** Uses a two-database system.
    - `journal.db`: Temporary storage for draft entries.
    - `memory.db`: Master storage for finalized journals, chat history, and vector embeddings.
- **Local AI:** Runs entirely locally using `llama.cpp` for inference and `sqlite-vec` for semantic search.
- **Optimization:** Heavily optimized for 12GB VRAM, utilizing manual model loading/unloading and 4-bit quantization.

## Tech Stack

- **Language:** Python 3.10+
- **LLM Runtime:** `llama-cpp-python` (CUDA enabled)
- **Main Model:** Gemma-3n-E4B (Q4_K_M GGUF)
- **Embeddings:** EmbeddingGemma-300M
- **TTS:** Chatterbox-Turbo
- **Database:** SQLite with `sqlite-vec` extension
- **Bot Framework:** `python-telegram-bot`

## Directory Structure

```text
/Users/kytzu/Globals/code/Micchon/
├── telegram-logging-ai/       # Main application source
│   ├── bot/                   # Bot logic, handlers, and AI personality
│   ├── memory/                # Database interactions and embedding logic
│   ├── data/                  # Runtime data (databases, images, journals) - gitignored
│   ├── models/                # GGUF model weights - gitignored
│   ├── scripts/               # Utility scripts (Qwen3, etc.)
│   ├── config.py              # Central configuration
│   ├── requirements.txt       # Python dependencies
│   └── .env                   # Secrets (API keys, allowed users)
├── PROJECT_HANDBOOK.md        # Consolidated Project Documentation (Arch, Setup, Maint)
├── AGENT_SYNC.md              # Live Agent Status
├── watchdog.sh                # Process Supervisor
└── GEMINI.md                  # This file
```

## Setup & Installation

### 1. Environment Setup
The project requires a Python virtual environment.

```bash
cd telegram-logging-ai
python -m venv venv
source venv/bin/activate
```

### 2. Dependencies
Install core dependencies. Note the specific requirement for `llama-cpp-python` with CUDA support for GPU acceleration.

```bash
# Standard dependencies
pip install -r requirements.txt

# Reinstall llama-cpp-python with CUDA support (CRITICAL for performance)
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# Install SQLite vector extension
pip install sqlite-vec

# Install TTS support (from source if PyPI is outdated)
pip install git+https://github.com/resemble-ai/chatterbox.git
```

### 3. Models
Models must be placed in `telegram-logging-ai/models/`.
- **Main Model:** `gemma-3n-e4b-q4_k_m.gguf` (Download from HuggingFace)
- **Embeddings:** Downloads automatically on first run.

### 4. Configuration
Copy `.env.example` to `.env` and populate:
- `TELEGRAM_BOT_TOKEN`: From BotFather.
- `TELEGRAM_ALLOWED_USERS`: Your Telegram User ID.
- `BLUESKY_*`: (Optional) Credentials for Bluesky posting.

## Running the Application

To start the bot:

```bash
cd telegram-logging-ai
python -m bot.main
```

## Development Conventions

- **Async First:** The bot utilizes `asyncio` and `python-telegram-bot`'s async capabilities. Ensure database and IO operations do not block the main loop.
- **Memory Management:** VRAM is a scarce resource.
    - The **Main Model** stays loaded.
    - **Embedding Models** should be loaded on-demand and unloaded immediately after use.
    - **TTS Models** stay loaded if space permits.
- **Code Style:** Pythonic, typed hints are encouraged.
- **File Paths:** Use `pathlib.Path` for file system operations. Configured paths are central in `config.py`.

## Key Commands (User Perspective)

- `/journal start` / `/journal done`: Manage logging sessions.
- `/voice`: Toggle TTS responses.
- `/bsky <text>`: Post to Bluesky.
- `/status`: Check system health and VRAM usage.

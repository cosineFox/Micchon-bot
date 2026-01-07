# Initialization Guide

Setup instructions for the Telegram Logging AI with Waifu.

---

## Prerequisites

### Hardware Requirements
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA with 8GB VRAM | RTX 3060 12GB |
| RAM | 16GB | 32GB |
| Storage | 10GB free | 20GB free |
| OS | Linux | Ubuntu 22.04+ |

### Software Requirements
- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- Git

---

## Step 1: Clone Repository

```bash
cd /path/to/Micchon
git clone <repo-url> telegram-logging-ai  # or use existing
cd telegram-logging-ai
```

---

## Step 2: Python Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install base dependencies
pip install -r requirements.txt
```

---

## Step 3: Install llama-cpp-python with CUDA

**Critical**: Must compile with CUDA support for GPU acceleration.

```bash
# Uninstall any existing version
pip uninstall llama-cpp-python -y

# Install with CUDA support
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

**Verify installation:**
```python
from llama_cpp import Llama
print("llama-cpp-python installed successfully")
```

---

## Step 4: Install TTS (Chatterbox)

```bash
# Try PyPI first
pip install chatterbox-tts

# If that fails, install from source
pip install git+https://github.com/resemble-ai/chatterbox.git
```

---

## Step 5: Install sqlite-vec

```bash
pip install sqlite-vec
```

**Verify:**
```python
import sqlite3
import sqlite_vec
db = sqlite3.connect(":memory:")
db.enable_load_extension(True)
sqlite_vec.load(db)
print("sqlite-vec loaded successfully")
```

---

## Step 6: Download Models

### Main Model (Required)
Download Gemma-3n-E4B GGUF (Q4_K_M quantization):

```bash
mkdir -p models

# Option 1: From HuggingFace (if available as GGUF)
wget -P models/ https://huggingface.co/huihui-ai/Huihui-gemma-3n-E4B-it-abliterated/resolve/main/gemma-3n-e4b-q4_k_m.gguf

# Option 2: Convert from HuggingFace format using llama.cpp
# (Requires llama.cpp repo cloned)
# python llama.cpp/convert.py --outtype q4_k_m ...
```

**Expected file:** `models/gemma-3n-e4b-q4_k_m.gguf` (~2.5GB)

### Embeddings Model (Auto-downloaded)
EmbeddingGemma-300M downloads automatically from HuggingFace on first use (~300MB).

---

## Step 7: Configure Environment

```bash
cp .env.example .env
```

Edit `.env`:
```bash
# Required
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_ALLOWED_USERS=your_telegram_user_id

# Optional (Bluesky integration)
BLUESKY_HANDLE=your.handle.bsky.social
BLUESKY_APP_PASSWORD=your_app_password
```

### Getting Telegram Bot Token
1. Message [@BotFather](https://t.me/BotFather) on Telegram
2. Send `/newbot`
3. Follow prompts to create bot
4. Copy the token

### Getting Your Telegram User ID
1. Message [@userinfobot](https://t.me/userinfobot) on Telegram
2. It replies with your user ID

### Getting Bluesky App Password (Optional)
1. Go to [bsky.app](https://bsky.app) → Settings → App Passwords
2. Create new app password
3. Copy and save it

---

## Step 8: Initialize Directories

```bash
mkdir -p data/images data/journals data/exports data/audio
mkdir -p models/adapters models/training
```

Or just run the bot - it creates directories automatically.

---

## Step 9: Verify Setup

```bash
# Run verification script
python -c "
import config
from pathlib import Path

print('Checking configuration...')

# Check model file
if config.MAIN_MODEL_PATH.exists():
    print(f'✓ Model found: {config.MAIN_MODEL_PATH}')
else:
    print(f'✗ Model missing: {config.MAIN_MODEL_PATH}')

# Check env vars
if config.TELEGRAM_BOT_TOKEN:
    print('✓ Telegram token configured')
else:
    print('✗ Telegram token missing')

if config.TELEGRAM_ALLOWED_USERS:
    print(f'✓ Allowed users: {config.TELEGRAM_ALLOWED_USERS}')
else:
    print('✗ No allowed users configured')

print('Done.')
"
```

---

## Step 10: Run the Bot

```bash
python -m bot.main
```

**Expected output:**
```
INFO - Starting Telegram Logging AI with Waifu...
INFO - Initializing databases...
INFO - Databases initialized
INFO - Loading LLM model...
INFO - LLM model loaded and warmed up
INFO - TTS client initialized
INFO - Waifu initialized
INFO - Bot handlers registered
INFO - Starting Telegram polling...
```

---

## First Run Checklist

- [ ] Bot responds to `/start` command
- [ ] Bot responds to text messages (waifu mode)
- [ ] `/status` shows system info
- [ ] `/journal start` enters journal mode
- [ ] `/voice` toggles TTS (if enabled)

---

## Troubleshooting

### "Model not found"
```
FileNotFoundError: models/gemma-3n-e4b-q4_k_m.gguf
```
Download the GGUF model to `models/` directory.

### "CUDA out of memory"
- Reduce `n_ctx` in config.py (try 1024)
- Disable TTS: `TTS_ENABLED = False`
- Use smaller quantization (Q3_K_S)

### "llama-cpp-python not using GPU"
Reinstall with CUDA:
```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### "sqlite-vec not found"
```bash
pip install sqlite-vec
```
Vector search falls back to time-based if unavailable.

### "Chatterbox import error"
```bash
pip install git+https://github.com/resemble-ai/chatterbox.git
```

### "Bot doesn't respond"
1. Check `TELEGRAM_ALLOWED_USERS` includes your user ID
2. Verify bot token is correct
3. Check logs for errors

---

## Quick Reference

| Path | Purpose |
|------|---------|
| `telegram-logging-ai/` | Project root |
| `models/` | GGUF model files |
| `data/memory.db` | Master database |
| `data/journal.db` | Draft entries |
| `data/images/` | Stored images |
| `data/journals/` | Compiled articles |
| `.env` | Secrets (never commit) |

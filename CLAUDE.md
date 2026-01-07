# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Telegram bot with AI waifu companion (Senko-san personality). Two-mode system: Normal Mode (waifu responds with memory context) and Journal Mode (silent logging → AI-compiled articles). Uses llama.cpp for inference, sqlite-vec for semantic search, Chatterbox for TTS.

## Commands

```bash
# Run the bot
cd telegram-logging-ai
python -m bot.main

# Install with CUDA support (required for GPU)
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# Install TTS from source (if PyPI fails)
pip install git+https://github.com/resemble-ai/chatterbox.git

# Check syntax
python3 -m py_compile telegram-logging-ai/bot/handlers.py
```

## Architecture

### Two-Database System
- **journal.db**: Temporary drafts during journal sessions (cleared after compilation)
- **memory.db**: Master database with all finalized content (waifu reads from here)

### Key Data Flow
```
Normal Mode: User message → Store in memory.db → Build context → Waifu responds → Store response
Journal Mode: User message → Store in journal.db → Silent ✓ → /journal done → AI compiles → Transfer to memory.db
```

### Module Responsibilities

**bot/**
- `model_manager.py`: GGUF model loading/unloading, warmup
- `llama_client.py`: llama-cpp-python wrapper with streaming
- `waifu.py`: Personality layer, context formatting, response generation
- `handlers.py`: Telegram command routing, mode-dependent behavior
- `journal_compiler.py`: Converts draft entries → AI-written articles
- `tts_client.py`: Chatterbox-Turbo voice synthesis
- `fine_tuner.py` + `scheduler.py`: LoRA fine-tuning from user ratings (2 AM daily)

**memory/**
- `master_repo.py`: memory.db CRUD + sqlite-vec vector search + PRAGMA optimizations
- `journal_repo.py`: journal.db CRUD for draft entries
- `embedder.py`: EmbeddingGemma-300M wrapper with LRU caching
- `context.py`: Builds WaifuContext (recent memories + semantic search results)
- `models.py`: Dataclasses (Memory, Journal, JournalEntry, WaifuContext, SessionState)

### Singleton Pattern
Components use `get_*` factory functions for singleton access:
- `get_model_manager()`, `get_waifu()`, `get_tts_client()`, `get_embedder()`, etc.

## Configuration

All settings in `telegram-logging-ai/config.py`. Key parameters:
- `LLAMA_PARAMS`: llama.cpp settings (n_ctx, n_gpu_layers, etc.)
- `GEN_PARAMS`: Generation settings (temperature, max_tokens, etc.)
- `WAIFU_PERSONALITY`: System prompt for AI personality
- `EMBEDDING_DIMENSION`: 256 (fast) or 768 (quality)

## Agent Coordination

Shared markdown files in Micchon root for multi-agent work:
- `AGENT_SYNC.md`: Implementation status, handoff notes, "Currently Working On"
- `PRD.md`: Full architecture spec
- `INIT.md`: Setup instructions
- `MAINTENANCE.md`: Test cases and maintenance tasks

Update `AGENT_SYNC.md` when picking up or completing work.

# Agent Sync File

Last updated: 2026-01-07

---

## SHARED DIRECTORY FOR MULTI-AGENT COORDINATION

These markdown files in `/Micchon/` root are the shared space for coordinating between coding agents.

### Key Files for Agents
| File | Purpose |
|------|---------|
| `AGENT_SYNC.md` | **THIS FILE** - Implementation status, current state, handoff notes |
| `PROJECT_HANDBOOK.md` | **MASTER DOC** - Consolidated PRD, Setup, Maintenance, and Logs |
| `telegram-logging-ai/README.md` | User-facing documentation for the bot |

### Project Structure
```
Micchon/                     # Root - shared agent files here
├── AGENT_SYNC.md            # Agent coordination (THIS FILE)
├── PROJECT_HANDBOOK.md      # Consolidated Documentation
├── watchdog.sh              # Process Supervisor
└── telegram-logging-ai/     # Telegram bot project
    ├── bot/
    ├── memory/
    ├── config.py
    ├── requirements.txt
    ├── README.md
    └── ...
```

### How to Use This File
1. **Before starting work**: Read this file to understand current state
2. **While working**: Update status table as you complete tasks
3. **After finishing**: Add notes to "Handoff Notes" section below
4. **Claiming work**: Add your agent identifier to "Currently Working On"

### Currently Working On
<!-- Add your agent ID and what you're working on -->
- Agent: Gemini CLI
- Task: Final verification and handoff
- Started: 2026-01-07 15:40 (approx)

### Handoff Notes
<!-- Add notes for the next agent here -->
- 2026-01-07: Implemented robust security and maintenance features.
  - Changed access control to `TELEGRAM_ALLOWED_CHAT_IDS` (supports groups).
  - Added `MaintenanceManager` for daily `VACUUM INTO` backups and integrity checks (2:30 AM).
  - Added `watchdog.sh` for process self-healing.
  - Added `Dockerfile` for Python 3.11 compatibility (resolves Chatterbox issues).
  - Global error handling added to `main.py`.
  - **Action Required:** User needs to set `TELEGRAM_ALLOWED_CHAT_IDS` in `.env`.

---

## Project Status: COMPLETE

The Telegram Logging AI with Waifu implementation is **fully complete**. All files have been created and the architecture is in place.

---

## Implementation Status

### Phase 1: Core Infrastructure ✅ COMPLETE
| File | Status | Description |
|------|--------|-------------|
| `memory/models.py` | ✅ | JournalEntry, Memory, Journal, WaifuContext, SessionState dataclasses |
| `memory/journal_repo.py` | ✅ | journal.db CRUD (draft entries) |
| `memory/master_repo.py` | ✅ | memory.db CRUD + sqlite-vec + PRAGMA optimizations |
| `memory/embedder.py` | ✅ | EmbeddingGemma wrapper with caching |
| `memory/context.py` | ✅ | Context builder for waifu (optimized window) |
| `memory/__init__.py` | ✅ | Module exports |
| `config.py` | ✅ | All settings, llama.cpp params, waifu personality |

### Phase 2: AI Components ✅ COMPLETE
| File | Status | Description |
|------|--------|-------------|
| `bot/model_manager.py` | ✅ | GGUF model loading, dynamic unloading, warmup |
| `bot/llama_client.py` | ✅ | llama-cpp-python wrapper with streaming |
| `bot/tts_client.py` | ✅ | Chatterbox-Turbo wrapper, voice generation |
| `bot/waifu.py` | ✅ | Personality + response generation + TTS |
| `bot/journal_compiler.py` | ✅ | Draft → article compilation |
| `bot/fine_tuner.py` | ✅ | LoRA fine-tuning from user ratings |
| `bot/scheduler.py` | ✅ | APScheduler for 2 AM fine-tuning |

### Phase 3: Bot Integration ✅ COMPLETE
| File | Status | Description |
|------|--------|-------------|
| `bot/journal_mode.py` | ✅ | Session state manager |
| `bot/handlers.py` | ✅ | All Telegram command handlers |
| `bot/bluesky_client.py` | ✅ | Bluesky read/write + memory storage |
| `bot/main.py` | ✅ | Entry point, wires everything together |
| `bot/__init__.py` | ✅ | Module exports |

### Supporting Files ✅ COMPLETE
| File | Status | Description |
|------|--------|-------------|
| `requirements.txt` | ✅ | All Python dependencies |
| `README.md` | ✅ | Complete documentation |
| `.gitignore` | ✅ | Git ignore patterns |
| `.env.example` | ✅ | Environment template |

---

## Architecture Summary

```
Two-Mode System:
├── Normal Mode: Waifu responds with full memory context
└── Journal Mode: Silent logging → AI compiles to articles

Two-Database Architecture:
├── journal.db: Temporary draft entries (cleared after compile)
└── memory.db: Master database (waifu reads here)

Tech Stack:
├── LLM: llama-cpp-python + Gemma-3n-E4B Q4_K_M GGUF (~2.5GB VRAM)
├── Embeddings: EmbeddingGemma-300M (256d MRL, on-demand loading)
├── TTS: Chatterbox-Turbo (350MB, <200ms latency)
├── Vector Search: sqlite-vec for semantic memory retrieval
└── Fine-tuning: Unsloth + PEFT LoRA (2 AM daily job)
```

---

## What Needs Testing

1. **Model Loading**: Ensure GGUF model exists at `models/gemma-3n-e4b-q4_k_m.gguf`
2. **CUDA Installation**: `llama-cpp-python` must be compiled with CUDA support
3. **TTS**: Chatterbox may need source install: `pip install git+https://github.com/resemble-ai/chatterbox.git`
4. **sqlite-vec**: May need manual install on some platforms
5. **Telegram Bot**: Create bot via @BotFather, add token to `.env` (bot only responds to users in TELEGRAM_ALLOWED_USERS)
6. **Bluesky**: Optional - add credentials to `.env` if needed (only posting functionality, no reading from timeline)

---

## Known Issues to Address (Future Work)

1. **LoRA → GGUF Merging**: Fine-tuner creates LoRA adapters but doesn't merge back to GGUF yet
2. **Vision Model**: Verify GGUF includes multimodal support for image description
3. ~~**Memory Cleanup**: `AUTO_CLEANUP_DAYS` logic not yet implemented~~ ✅ FIXED - scheduler now runs cleanup at 3 AM
4. **Error Recovery**: Journal session recovery after crash not fully tested

---

## Privacy Considerations

- **Telegram Access**: The bot is designed to be personal and only responds to users in the `TELEGRAM_ALLOWED_USERS` list in the .env file. This is intentional for privacy and security.
- **Bluesky Integration**: The bot only supports posting to your Bluesky account (not reading from it). Credentials are optional and can be left blank if you don't want this functionality.
- **Data Storage**: All data is stored locally in SQLite databases (journal.db and memory.db) on your device.

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install llama-cpp-python with CUDA
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# 3. Download GGUF model to models/ directory

# 4. Configure .env
cp .env.example .env
# Edit .env with your Telegram bot token and user ID

# 5. Run
python -m bot.main
```

---

## File Tree

```
Micchon/                         # Root directory
├── AGENT_SYNC.md                # Agent coordination (SHARED)
├── PRD.md                       # Product requirements (SHARED)
├── INIT.md                      # Setup instructions (SHARED)
├── MAINTENANCE.md               # Test cases & maintenance (SHARED)
└── telegram-logging-ai/         # Bot project
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
    │   ├── scheduler.py         # Scheduled tasks (fine-tune 2AM, cleanup 3AM)
    │   └── keyword_manager.py   # Dynamic keyword/interest tracking
    ├── memory/
    │   ├── __init__.py
    │   ├── models.py            # Data models
    │   ├── master_repo.py       # memory.db repository
    │   ├── journal_repo.py      # journal.db repository
    │   ├── embedder.py          # EmbeddingGemma wrapper
    │   └── context.py           # Context builder for waifu
    ├── data/                    # Runtime data (gitignored)
    ├── models/                  # GGUF model files (gitignored)
    ├── config.py
    ├── requirements.txt
    ├── README.md
    ├── .env.example
    └── .gitignore
```

---

## Commands Reference

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

---

## Notes for Other Agents

- **PRD**: Full architecture spec at `Micchon/PRD.md`
- **Bot code**: All source in `Micchon/telegram-logging-ai/`
- **All Python files exist**: Implementation is complete, focus on testing/debugging
- **No migrations needed**: Fresh start design, no legacy data to handle
- **Modular design**: Each component is self-contained with singleton getters
- **Async everywhere**: All I/O operations are async-native

## Agent Communication Protocol

When handing off to another agent:
1. Update "Currently Working On" section to `_none_`
2. Add entry to "Handoff Notes" with date and summary
3. Update any status tables if tasks changed
4. Save this file

When picking up work:
1. Read this entire file first
2. Read `PRD.md` for architecture context
3. Update "Currently Working On" with your agent ID and task
4. Proceed with work

# Unified Memory + Waifu Architecture Plan

## Overview

Restructure the telegram bot from project-scoped logging to a unified memory system where a "waifu" AI has omniscient access to all user activity.

## Hardware (Self-hosted)

| Component | Spec |
|-----------|------|
| CPU | AMD Ryzen 7 2700X (8c/16t) |
| GPU | NVIDIA RTX 3060 12GB VRAM |
| RAM | 16GB |
| OS | Linux |

## Tech Stack (Extreme Optimization)

| Component | Choice | Reason |
|-----------|--------|--------|
| LLM Runtime | **llama.cpp** (llama-cpp-python) | 2-3x faster inference, manual memory control |
| Main Model | Gemma-3n-E4B Q4_K_M GGUF | Vision + conversation, 4-bit quant (~2.5GB) |
| Embeddings | **EmbeddingGemma-300M** | 768d (or 256d MRL), Gemma 3 family, task-aware |
| **TTS** | **Chatterbox-Turbo** | 350M params, <200ms latency, paralinguistics |
| Bot Framework | python-telegram-bot | Async-native |
| Database | SQLite + sqlite-vec | Lightweight, in-process |
| Bluesky SDK | atproto | Official SDK |

**Why llama.cpp:**
- 40-60% faster inference vs Ollama
- Manual model loading/unloading (save VRAM)
- Better quantization control
- Lower memory overhead
- Thread pool optimization

**Model Strategy - Dynamic Loading:**
```python
# Prioritize speed - keep frequently used models loaded

STATE:
- Main model: Always loaded - Gemma-3n-E4B Q4_K_M (2.5GB)
- TTS: Always loaded - Chatterbox-Turbo (350MB)
- Embeddings: On-demand - EmbeddingGemma-300M (300MB)
- No FunctionGemma (use regex-based routing instead)
```

**VRAM Usage (optimized):**
- Gemma-3n: ~2.5GB
- Chatterbox-Turbo: ~350MB
- Safety buffer: ~1GB
- Total: ~4GB used of 12GB
- **67% VRAM free for other tasks**

**Download Models:**
```bash
# Main model (vision + conversation)
wget https://huggingface.co/.../gemma-3n-e4b-q4_k_m.gguf

# Embeddings (on-demand)
wget https://huggingface.co/.../nomic-embed-text-q4_0.gguf
```

**Performance Targets:**
- Waifu response: <2s (vs ~4s Ollama)
- Image description: <3s
- Journal compilation: <10s
- Embedding generation: <500ms

## Core Concepts

1. **Two Modes**: Journal Mode (silent capture) vs Normal Mode (waifu active)
2. **Two Databases**: journal.db (drafts) → memory.db (master, waifu reads)
3. **Waifu** - Always-on AI companion with full context (Senko personality)
4. **No more "projects"** - Everything flows into unified memory

---

## Extreme Optimization Strategies

### 1. Model Management
```python
class ModelManager:
    def __init__(self):
        self.main_model = None  # Gemma-3n loaded on startup
        self.embed_model = None # Load on-demand only

    async def get_embedding(self, text: str):
        # Load embedding model
        if not self.embed_model:
            self.embed_model = load_gguf("nomic-embed-q4_0.gguf")

        result = self.embed_model.embed(text)

        # Unload immediately to free VRAM
        del self.embed_model
        self.embed_model = None
        torch.cuda.empty_cache()

        return result
```

### 2. Context Window Optimization
- **Max context**: 2048 tokens (vs 8K default)
- **Waifu context**: Last 10 messages + 5 relevant memories
- **Journal compile**: Summarize if >1500 tokens
- **Saves**: 70% inference time on long contexts

### 3. Inference Optimization
```python
# llama.cpp settings
params = {
    "n_gpu_layers": -1,        # Full GPU offload
    "n_ctx": 2048,             # Tight context
    "n_batch": 512,            # Optimal for RTX 3060
    "n_threads": 4,            # Leave cores for system
    "f16_kv": True,            # Half-precision KV cache
    "use_mlock": True,         # Lock model in RAM
    "rope_freq_base": 10000,   # RoPE optimization
}
```

### 4. Database Optimization
```python
# SQLite optimizations
PRAGMA journal_mode = WAL;        # Write-ahead logging
PRAGMA synchronous = NORMAL;      # Faster writes
PRAGMA cache_size = -64000;       # 64MB cache
PRAGMA temp_store = MEMORY;       # Temp tables in RAM
PRAGMA mmap_size = 268435456;     # 256MB memory-mapped I/O
```

### 5. Batch Operations
- **Embeddings**: Generate in batches of 10
- **Image descriptions**: Queue and process together
- **Journal entries**: Compile every 50 entries or 24h

### 6. Caching Strategy
```python
# LRU cache for repeated queries
@lru_cache(maxsize=100)
async def get_embedding_cached(text: str):
    return await embedder.embed(text)

# Cache recent waifu responses (5 min TTL)
response_cache = TTLCache(maxsize=50, ttl=300)
```

### 7. Memory Optimizations
- **Lazy loading**: Only load memories when needed
- **Streaming**: Stream LLM tokens instead of waiting
- **Compression**: Compress old journal entries (gzip)
- **Cleanup**: Delete embeddings for entries >90 days old

### 8. Async Optimizations
```python
# Process image description in background
asyncio.create_task(describe_and_store(image))
await update.message.reply_text("✓")  # Instant ack

# Pre-warm model on startup
await model.warmup(prompt="Hello")
```

---

## Two-Database Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA FLOW                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  JOURNAL MODE (Silent)              NORMAL MODE (Waifu Active)  │
│  ════════════════════              ═══════════════════════════  │
│                                                                 │
│  /journal start                    Any message (no command)     │
│       │                                   │                     │
│       v                                   v                     │
│  ┌───────────┐                     ┌───────────────┐            │
│  │journal.db │                     │   WAIFU       │            │
│  │  (draft)  │                     │  responds     │            │
│  └───────────┘                     └───────┬───────┘            │
│       │                                    │                    │
│       │ text, images                       v                    │
│       │ stored silently              ┌───────────┐              │
│       │                              │memory.db  │              │
│       │                              │ (master)  │              │
│  /journal done                       └───────────┘              │
│       │                                    ^                    │
│       v                                    │                    │
│  ┌───────────┐                             │                    │
│  │ AI compile│ → title + article           │                    │
│  └───────────┘                             │                    │
│       │                                    │                    │
│       └────────── finalized ───────────────┘                    │
│                   journal moved                                 │
│                   to master db                                  │
│                                                                 │
│  journal.db cleared after transfer                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**journal.db** (temporary):
- Active only during journal mode
- Stores raw entries, images, notes
- Cleared after compile + transfer

**memory.db** (master):
- All finalized content
- Waifu reads ONLY from here
- Contains: compiled journals, chat history, bsky posts, images

---

## Data Models

### JournalEntry (draft - lives in journal.db during session)
```python
@dataclass
class JournalEntry:
    id: str                    # UUID
    timestamp: datetime
    type: str                  # "text", "image"
    content: str               # Raw user input
    media_path: str | None     # Local path for images
    image_description: str | None
```

### Memory (finalized - lives in memory.db, waifu reads this)
```python
@dataclass
class Memory:
    id: str                    # UUID
    timestamp: datetime
    type: str                  # "journal", "image", "chat", "bsky_post"
    content: str               # Text content or AI description
    raw_content: str | None    # Original unprocessed input
    media_path: str | None     # Local path for images
    metadata: dict             # {tags, bsky_uri, journal_id, etc.}
```

### Journal (compiled article - lives in memory.db)
```python
@dataclass
class Journal:
    id: str
    title: str
    body: str                  # AI-written article (markdown)
    source_entry_ids: list[str]  # Original JournalEntry IDs
    created_at: datetime
    tags: list[str]
    markdown_path: str | None  # Exported .md file path
```

### WaifuContext (runtime only, not persisted)
```python
@dataclass
class WaifuContext:
    recent_memories: list[Memory]      # Last N hours
    relevant_memories: list[Memory]    # Semantic search results
    recent_journals: list[Journal]     # Latest compiled articles
    user_profile: dict                 # Learned preferences
```

### SessionState (runtime, tracks current mode)
```python
@dataclass
class SessionState:
    user_id: int
    mode: str                  # "normal" or "journal"
    journal_started_at: datetime | None
```

---

## Database Schema

### journal.db (temporary, active during journal mode)
```sql
-- Draft entries during journal session
CREATE TABLE journal_entries (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    type TEXT NOT NULL,           -- "text", "image"
    content TEXT NOT NULL,
    media_path TEXT,
    image_description TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_journal_timestamp ON journal_entries(timestamp DESC);
```

### memory.db (master, waifu reads from here)
```sql
-- All finalized memories
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    type TEXT NOT NULL,           -- "journal", "image", "chat", "bsky_post"
    content TEXT NOT NULL,
    raw_content TEXT,
    media_path TEXT,
    metadata TEXT,                -- JSON
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Vector embeddings for semantic search
CREATE VIRTUAL TABLE memory_embeddings USING vec0(
    id TEXT PRIMARY KEY,
    embedding FLOAT[384]          -- nomic-embed-text dimension
);

-- Compiled journal articles
CREATE TABLE journals (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    source_entry_ids TEXT,        -- JSON array of original entry IDs
    tags TEXT,                    -- JSON array
    markdown_path TEXT,
    created_at TEXT NOT NULL
);

-- Indexes
CREATE INDEX idx_memories_timestamp ON memories(timestamp DESC);
CREATE INDEX idx_memories_type ON memories(type);
CREATE INDEX idx_journals_created ON journals(created_at DESC);
```

---

## Component Details

### 1. Memory Repository (`memory/master_repo.py`)

**Key methods:**
- `add_memory(type, content, media_path, metadata)` → stores + generates embedding
- `get_recent(hours=24, limit=50)` → time-based retrieval
- `search_similar(query, limit=10)` → vector similarity search
- `get_by_type(type, limit=20)` → filter by memory type
- `add_journal(title, body, source_ids, tags)` → store compiled article
- `get_recent_journals(limit=5)` → for waifu context

### 2. Embedder (`memory/embedder.py`)

Uses EmbeddingGemma-300M via sentence-transformers:
- `embed(text) → list[float]`
- Batch support for efficiency
- 256-dimension vectors (MRL for speed)

### 3. Context Builder (`memory/context.py`)

Builds waifu's context window:
```python
async def build_context(query: str, user_id: int) -> WaifuContext:
    # 1. Get last 24h of memories
    recent = await repo.get_recent(hours=24)

    # 2. Semantic search for relevant older memories
    relevant = await repo.search_similar(query, limit=10)

    # 3. Get recent journals (waifu knows what she wrote)
    journals = await repo.get_recent_journals(limit=3)

    # 4. Load user profile (future: learned preferences)
    profile = await repo.get_user_profile(user_id)

    return WaifuContext(recent, relevant, journals, profile)
```

### 4. Waifu (`bot/waifu.py`)

Personality layer + response generation:
```python
class Waifu:
    def __init__(self, llama: LlamaClient, context_builder: ContextBuilder):
        self.llama = llama
        self.context = context_builder
        self.personality = WAIFU_SYSTEM_PROMPT  # Configurable

    async def respond(self, message: str, user_id: int) -> str:
        # Build context
        ctx = await self.context.build_context(message, user_id)

        # Format prompt with memories
        prompt = self._format_prompt(message, ctx)

        # Generate response
        response = await self.llama.generate(prompt, system=self.personality)

        # Store this conversation as memory
        await self.repo.add_memory("chat", response, metadata={"role": "assistant"})

        return response
```

### 5. Journal Compiler (`bot/journal_compiler.py`)

Compiles draft entries from journal.db into finalized article in memory.db.

### 6. Handlers (`bot/handlers.py`)

**Commands:**
| Command | Action |
|---------|--------|
| `/start` | Initialize bot, show help |
| `/journal start` | Enter journal mode (silent logging) |
| `/journal done` | Exit journal mode, compile article, move to master DB |
| `/journal cancel` | Exit journal mode without compiling |
| `/bsky <text>` | Post to Bluesky + store as memory (normal mode only) |
| `/status` | Show current mode, memory stats |
| `/voice` | Toggle voice responses |
| `/rate <1-5>` | Rate last response (for fine-tuning) |
| `/help` | Show command reference |

---

## Telegram Handler Flow

```
User sends message
       │
       v
┌──────────────────────────────────────────────────────────────┐
│                    CHECK CURRENT MODE                        │
└──────────────────────────────────────────────────────────────┘
       │
       ├─── JOURNAL MODE ────────────────────────────────────┐
       │                                                      │
       │    ┌─────────────────────────────────────────────┐  │
       │    │ /journal done?                              │  │
       │    │   YES → Compile entries → AI generates      │  │
       │    │         title + article → Save to memory.db │  │
       │    │         → Clear journal.db → Exit mode      │  │
       │    │   NO  → Store silently to journal.db        │  │
       │    │         (text or image, no response)        │  │
       │    └─────────────────────────────────────────────┘  │
       │                                                      │
       └─── NORMAL MODE ─────────────────────────────────────┐
                                                              │
            ┌─────────────────────────────────────────────┐  │
            │ Has command?                                │  │
            │   /journal start → Enter journal mode       │  │
            │   /bsky → Post to Bluesky + store memory    │  │
            │   /status → Show stats                      │  │
            │                                             │  │
            │ No command (default):                       │  │
            │   1. Store user msg as memory (type=chat)   │  │
            │   2. Build waifu context                    │  │
            │   3. Generate response with personality     │  │
            │   4. Store response as memory               │  │
            │   5. Reply to user                          │  │
            └─────────────────────────────────────────────┘  │
                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Quality of Life Features

### Error Handling
| Scenario | Handling |
|----------|----------|
| Model not loaded | Graceful fallback, store raw content, retry queue |
| Bluesky API failure | Log locally, mark as "pending sync", retry later |
| Image too large | **Auto-compress** to max 2048px, 85% quality JPEG |
| Journal mode timeout | Warn after 1 hour idle, auto-compile after 24h |

### User Experience
- **Immediate feedback**: Acknowledge within 500ms, process async
- **Progress indicators**: "Processing image...", "Compiling journal..."
- **Silent confirmations in journal mode**: Just ✓ when entry stored

### Reliability
- **Auto-save**: Entries persisted before any AI processing
- **Crash recovery**: On restart, check for pending journal sessions
- **Backup**: Daily SQLite backup to timestamped file

### Security
- **Telegram user whitelist**: Only respond to authorised user IDs
- **Credentials**: Stored in `.env`, never logged
- **Local storage**: All data stays on your machine

---

## Future Enhancements (v2)

- Bluesky feed reading → memories (waifu knows your timeline)
- Scheduled journal compilation (daily/weekly)
- Voice messages → transcription → memory
- Web dashboard for browsing memories
- Multiple waifu personalities (switchable)
- `/edit` command to modify entries

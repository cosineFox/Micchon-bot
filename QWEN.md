# Micchon Project - Telegram Logging AI with Waifu

## Project Overview

The Micchon project is a sophisticated Telegram bot called "Telegram Logging AI with Waifu" that combines personal journaling with an AI companion experience. The bot operates in two distinct modes:

1. **Normal Mode**: An AI companion (waifu) with Senko-san personality responds to all messages, remembering past conversations and interactions
2. **Journal Mode**: Silent logging of text and images, later compiled into AI-written articles

The project features a unified memory system where all interactions are stored and the AI can query for context-aware responses. It's designed for extreme optimization on consumer GPUs (specifically RTX 3060 12GB) using llama-cpp-python for efficient inference.

## Architecture

### Two-Mode System
- **Normal Mode**: Waifu responds with full memory context
- **Journal Mode**: Silent logging → AI compiles to articles

### Two-Database Architecture
- **journal.db**: Temporary draft entries during journal sessions (cleared after compilation)
- **memory.db**: Master database storing all finalized memories, journals, and chat history (permanent)

### Tech Stack
- **LLM**: llama-cpp-python + Gemma-3n-E4B Q4_K_M GGUF (~2.5GB VRAM)
- **Embeddings**: EmbeddingGemma-300M for semantic search (256d MRL, on-demand loading)
- **TTS**: Chatterbox-Turbo (350MB, <200ms latency)
- **Vector Search**: sqlite-vec for semantic memory retrieval
- **Fine-tuning**: Unsloth + PEFT LoRA (2 AM daily job)

## Key Features

- **Unified Memory**: All interactions stored in a persistent memory system that the AI can query
- **Adaptive Personality**: Senko-san personality (warm, motherly, nurturing) with contextual awareness
- **Automatic Improvement**: Continuous fine-tuning based on user feedback
- **Dual-Purpose Design**: Functions as both a personal companion and a logging tool
- **Export Capabilities**: Compiles journal entries into markdown articles with AI-generated summaries
- **Bluesky Integration**: Automatic posting with AI-polished content
- **Voice Responses**: TTS with paralinguistics support ([laugh], [sigh], [chuckle], etc.)

## File Structure

```
Micchon/
├── telegram-logging-ai/          # Main application directory
│   ├── bot/                     # Bot logic components
│   │   ├── main.py              # Entry point
│   │   ├── handlers.py          # Telegram command handlers
│   │   ├── model_manager.py     # GGUF model loading/unloading
│   │   ├── llama_client.py      # llama-cpp-python wrapper
│   │   ├── tts_client.py        # Chatterbox-Turbo TTS
│   │   ├── waifu.py             # AI personality + response generation
│   │   ├── journal_mode.py      # Session state management
│   │   ├── journal_compiler.py  # Draft → article compilation
│   │   ├── bluesky_client.py    # Bluesky integration
│   │   ├── fine_tuner.py        # LoRA fine-tuning
│   │   └── scheduler.py         # Scheduled tasks (2 AM fine-tuning)
│   ├── memory/                  # Memory and data models
│   │   ├── models.py            # Data models (Memory, Journal, etc.)
│   │   ├── master_repo.py       # memory.db repository
│   │   ├── journal_repo.py      # journal.db repository
│   │   ├── embedder.py          # EmbeddingGemma wrapper
│   │   └── context.py           # Context builder for waifu
│   ├── data/                    # Runtime data (gitignored)
│   ├── models/                  # GGUF model files (gitignored)
│   ├── config.py                # All configuration
│   ├── requirements.txt         # Python dependencies
│   └── .env.example             # Environment template
├── SPEC.md                      # Project specification
├── PRD.md                       # Product requirements document
├── AGENT_SYNC.md                # Agent synchronization file
├── requirements.txt             # Python dependencies (root)
└── LICENSE                      # License information
```

## Building and Running

### Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA support (tested on RTX 3060 12GB)
- Git

### Setup
1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or: venv\Scripts\activate  # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Install llama-cpp-python with CUDA support:
   ```bash
   CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
   ```
5. Download the GGUF model to the `models/` directory:
   - Get Gemma-3n-E4B Q4_K_M GGUF model and save as `models/gemma-3n-e4b-q4_k_m.gguf`
6. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your Telegram bot token and user ID
   ```
7. Run the bot:
   ```bash
   python -m bot.main
   ```

### Model Requirements
- Main model: Gemma-3n-E4B Q4_K_M GGUF (~2.5GB VRAM)
- Embedding model: EmbeddingGemma-300M (loaded on-demand)
- TTS model: Chatterbox-Turbo (~350MB VRAM)

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
| `/bsky <text>` | Post to Bluesky (AI-polished) |
| `/voice` | Toggle voice responses on/off |
| `/rate <1-5>` | Rate the last AI response |
| `/status` | Show system status |

## Development Conventions

- All components are built with async/await patterns for optimal performance
- Model loading/unloading is managed dynamically to optimize VRAM usage
- SQLite with PRAGMA optimizations is used for persistence
- Vector search implemented with sqlite-vec for semantic memory retrieval
- Error handling includes graceful fallbacks and proper logging
- Resource cleanup is implemented throughout the application lifecycle

## Known Issues for Future Work

1. **LoRA → GGUF Merging**: Fine-tuner creates LoRA adapters but doesn't merge back to GGUF yet
2. **Vision Model**: Verify GGUF includes multimodal support for image description
3. **Memory Cleanup**: AUTO_CLEANUP_DAYS logic not yet implemented
4. **Error Recovery**: Journal session recovery after crash not fully tested

## Project Status

The implementation is marked as **COMPLETE** in the AGENT_SYNC.md file, with all components built according to the specifications.
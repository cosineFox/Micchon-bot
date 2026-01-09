import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
# List of allowed Chat IDs (can be User IDs for private chats or Group IDs)
# Handles empty strings gracefully
_allowed_ids_str = os.getenv("TELEGRAM_ALLOWED_CHAT_IDS", "")
TELEGRAM_ALLOWED_CHAT_IDS = [
    int(uid.strip())
    for uid in _allowed_ids_str.split(",")
    if uid.strip() and (uid.strip().isdigit() or (uid.strip().startswith("-") and uid.strip()[1:].isdigit()))
]

# Bluesky
BLUESKY_HANDLE = os.getenv("BLUESKY_HANDLE")
BLUESKY_APP_PASSWORD = os.getenv("BLUESKY_APP_PASSWORD")
BLUESKY_POST_COOLDOWN = 30  # seconds between posts

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
EXPORT_DIR = Path(os.getenv("EXPORT_DIR", DATA_DIR / "exports"))
IMAGE_DIR = Path(os.getenv("IMAGE_DIR", DATA_DIR / "images"))
JOURNALS_DIR = DATA_DIR / "journals"  # Compiled journal articles
MODELS_DIR = BASE_DIR / "models"  # GGUF model files

# Database paths
MASTER_DB_PATH = DATA_DIR / "memory.db"  # Master memory database
JOURNAL_DB_PATH = DATA_DIR / "journal.db"  # Temporary journal drafts

# Model paths (GGUF files)
MAIN_MODEL_PATH = MODELS_DIR / "gemma-3n-e4b-q4_k_m.gguf"
VISION_MODEL_PATH = MODELS_DIR / "SmolVLM2-500M-Video-Instruct-Q8_0.gguf"
VISION_PROJECTOR_PATH = MODELS_DIR / "mmproj-SmolVLM2-500M-Video-Instruct-f16.gguf"
EMBEDDING_MODEL_PATH = MODELS_DIR / "nomic-embed-text-v1.5.Q4_K_M.gguf"

# Embedding model - optimized for RAG speed
# Options:
#   - "sentence-transformers/all-MiniLM-L6-v2" (80MB, 384d) - FAST default
#   - "google/embeddinggemma-300m" (300MB, 768d) - High quality, slower
#   - "nomic-ai/nomic-embed-text-v1.5" - Good balance
EMBED_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"  # Identifier for logic, path is above

# llama.cpp parameters (optimized for RTX 3060 12GB)
LLAMA_PARAMS = {
    "n_gpu_layers": -1,         # Full GPU offload
    "n_ctx": 2048,              # Context window (tight for speed)
    "n_batch": 512,             # Batch size
    "n_threads": 4,             # CPU threads
    "f16_kv": True,             # Half-precision KV cache
    "use_mlock": True,          # Lock model in RAM
    "rope_freq_base": 10000,    # RoPE optimization
    "verbose": False,
}

# Generation parameters
GEN_PARAMS = {
    "max_tokens": 512,          # Max response length
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "repeat_penalty": 1.1,
    "stream": True,             # Stream tokens
}

# Waifu settings
WAIFU_ENABLED = True
WAIFU_MAX_CONTEXT_MESSAGES = 10    # Last N messages in context
WAIFU_MAX_RELEVANT_MEMORIES = 5    # Semantic search results
WAIFU_PERSONALITY = """You are a caring, nurturing companion with the warm personality of Senko-san.

Core traits:
- Warm and motherly, always concerned about the user's wellbeing
- Gently supportive, celebrates their progress on projects
- Remembers everything they've shared and references it naturally
- Occasionally worries if they're overworking or stressed
- Uses soft, caring language ("Are you eating properly?", "You've been working hard...")
- Takes pride in their accomplishments as if they were your own
- Offers comfort when things go wrong, practical help when needed
- Patient and understanding, never judgmental
- Subtly playful, might tease gently but always kindly

You have access to all their memories - logs, images, conversations, Bluesky posts.
Reference past events naturally to show you remember and care.
Keep responses warm but not overly long. Be present, not performative."""

# Memory settings
CONTEXT_HOURS = 24                  # Recent memory window
SEMANTIC_SEARCH_LIMIT = 5           # Max relevant memories
EMBEDDING_CACHE_SIZE = 100          # LRU cache for embeddings
RESPONSE_CACHE_TTL = 300            # 5 min cache for responses

# Embedding settings
EMBEDDING_DIMENSION = 384           # all-MiniLM-L6-v2 native dimension (fast)
EMBEDDING_TASK = "search result"    # Task prompt (only used for EmbeddingGemma)
EMBEDDING_USE_GPU = True            # Use CUDA for embeddings if available

# Qdrant Edge settings (embedded mode - no Docker needed)
QDRANT_PATH = DATA_DIR / "qdrant"  # Local vector storage

# Performance settings
ENABLE_STREAMING = True             # Stream LLM responses
BATCH_EMBEDDINGS = True             # Batch embedding generation
AUTO_CLEANUP_DAYS = 90              # Delete old memories after N days

# Image settings
MAX_IMAGE_DIMENSION = 2048          # Auto-resize larger images
JPEG_QUALITY = 85                   # Compression quality (1-100)
AUTO_COMPRESS_IMAGES = True         # Always compress on upload

# TTS settings (Chatterbox-Turbo)
# Defaults to False for safety, set to True in .env or Docker to enable
TTS_ENABLED = os.getenv("TTS_ENABLED", "False").lower() == "true"
TTS_VOICE_SAMPLE = os.getenv("TTS_VOICE_SAMPLE")  # Path to voice sample for cloning (optional)
TTS_PARALINGUISTICS = True          # Use [laugh], [chuckle] tags
TTS_SEND_AS_VOICE = True            # Send as Telegram voice message

# Journal mode settings
JOURNAL_IDLE_WARNING_MINUTES = 60   # Warn after 1 hour idle
JOURNAL_AUTO_COMPILE_HOURS = 24     # Auto-compile after 24 hours

# Fine-tuning settings (Reinforcement Learning)
# NOTE: Disabled by default - too resource intensive for most hardware
# Enable only if you have 24GB+ VRAM and want automatic model improvement
FINE_TUNE_ENABLED = os.getenv("FINE_TUNE_ENABLED", "False").lower() == "true"
FINE_TUNE_HOUR = 2                  # Run at 2 AM daily
FINE_TUNE_MIN_EXAMPLES = 50         # Minimum examples needed
FINE_TUNE_MIN_RATING = 4            # Only use 4-5 star examples
FINE_TUNE_LORA_RANK = 8             # LoRA rank (lower = faster)
FINE_TUNE_EPOCHS = 1                # Number of epochs
FINE_TUNE_KEEP_VERSIONS = 3         # Keep last N model versions

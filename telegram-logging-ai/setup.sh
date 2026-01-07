#!/bin/bash
# Setup script for Telegram Logging AI with Waifu
# Optimized for RTX 3060 12GB + 16GB RAM

set -e

echo "========================================"
echo "Telegram Logging AI - Setup Script"
echo "========================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
python3 --version || { echo -e "${RED}Python 3 not found!${NC}"; exit 1; }

# Create directories
echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p data/images data/journals data/exports data/audio models

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}Creating .env from template...${NC}"
    cp .env.example .env
    echo -e "${RED}IMPORTANT: Edit .env with your Telegram bot token!${NC}"
fi

# Install Python dependencies
echo -e "${YELLOW}Installing Python dependencies...${NC}"
pip install -r requirements.txt

# Check CUDA availability
echo -e "${YELLOW}Checking CUDA availability...${NC}"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Install llama-cpp-python with CUDA
echo ""
echo -e "${YELLOW}Installing llama-cpp-python with CUDA support...${NC}"
echo "This may take a few minutes..."
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir || {
    echo -e "${RED}CUDA installation failed. Trying CPU-only version...${NC}"
    pip install llama-cpp-python --force-reinstall --no-cache-dir
}

# Install Chatterbox TTS from GitHub
echo ""
echo -e "${YELLOW}Installing Chatterbox TTS from GitHub...${NC}"
pip install git+https://github.com/resemble-ai/chatterbox.git || {
    echo -e "${YELLOW}Warning: Chatterbox installation failed. TTS will be disabled.${NC}"
}

# Download embedding model (will auto-download on first use, but let's pre-cache it)
echo ""
echo -e "${YELLOW}Pre-downloading embedding model...${NC}"
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" || {
    echo -e "${YELLOW}Warning: Could not pre-download embedding model. It will download on first use.${NC}"
}

# Check for GGUF model
echo ""
echo -e "${YELLOW}Checking for LLM model...${NC}"
if [ -f "models/gemma-3n-e4b-q4_k_m.gguf" ]; then
    echo -e "${GREEN}GGUF model found!${NC}"
else
    echo -e "${RED}GGUF model not found at models/gemma-3n-e4b-q4_k_m.gguf${NC}"
    echo ""
    echo "Download a GGUF model and place it in the models/ directory."
    echo "Recommended models for RTX 3060 12GB:"
    echo "  - gemma-2-2b-it-Q4_K_M.gguf (~1.5GB VRAM)"
    echo "  - gemma-3n-e4b-q4_k_m.gguf (~2.5GB VRAM)"
    echo "  - llama-3.2-3b-Q4_K_M.gguf (~2GB VRAM)"
    echo ""
    echo "Download from: https://huggingface.co/models?search=gguf"
fi

# Validate config
echo ""
echo -e "${YELLOW}Validating configuration...${NC}"
python3 -c "
import config
print(f'Data dir: {config.DATA_DIR}')
print(f'Model path: {config.MAIN_MODEL_PATH}')
print(f'Embed model: {config.EMBED_MODEL_NAME}')
print(f'Fine-tuning: {\"ENABLED\" if config.FINE_TUNE_ENABLED else \"DISABLED (RAG mode)\"}')
print(f'TTS: {\"ENABLED\" if config.TTS_ENABLED else \"DISABLED\"}')
" || {
    echo -e "${RED}Config validation failed!${NC}"
    exit 1
}

echo ""
echo -e "${GREEN}========================================"
echo "Setup complete!"
echo "========================================${NC}"
echo ""
echo "Next steps:"
echo "1. Edit .env with your TELEGRAM_BOT_TOKEN"
echo "2. Download a GGUF model to models/"
echo "3. Run: python -m bot.main"
echo ""
echo "VRAM usage estimate:"
echo "  - LLM (Q4): ~2-3GB"
echo "  - TTS: ~350MB"
echo "  - Embeddings: ~100MB (GPU) or CPU"
echo "  - Total: ~3-4GB of 12GB"
echo ""

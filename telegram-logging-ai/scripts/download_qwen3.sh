#!/bin/bash

# Script to download Qwen3-0.6B model for Raspberry Pi 5
echo "Downloading Qwen3-0.6B model..."

# Create models directory if it doesn't exist
mkdir -p telegram-logging-ai/models

# Navigate to models directory
cd telegram-logging-ai/models

# Check if model already exists
if [ -f "Qwen3-0.6B-heretic-abliterated-uncensored.Q5_K_M.gguf" ]; then
    echo "Qwen3-0.6B model already exists. Skipping download."
    exit 0
fi

echo "Downloading Qwen3-0.6B-heretic-abliterated-uncensored.Q5_K_M.gguf..."
# Using wget to download the model file
wget -O Qwen3-0.6B-heretic-abliterated-uncensored.Q5_K_M.gguf \
    https://huggingface.co/DavidAU/Qwen3-0.6B-heretic-abliterated-uncensored/resolve/main/Qwen3-0.6B-heretic-abliterated-uncensored.Q5_K_M.gguf

if [ $? -eq 0 ]; then
    echo "Download completed successfully!"
    ls -lh Qwen3-0.6B-heretic-abliterated-uncensored.Q5_K_M.gguf
else
    echo "Download failed. Please check the URL or your internet connection."
    exit 1
fi

echo "Model download complete. You can now run the Qwen3 version of the bot."
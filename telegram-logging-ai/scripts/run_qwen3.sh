#!/bin/bash

# Script to run the Qwen3 version of Telegram Logging AI with Waifu

echo "Starting Telegram Logging AI with Qwen3 Waifu..."

# Activate virtual environment if it exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "Virtual environment activated"
elif [ -f "../venv/bin/activate" ]; then
    source ../venv/bin/activate
    echo "Virtual environment activated"
else
    echo "Warning: No virtual environment found. Using system Python."
fi

# Navigate to the telegram-logging-ai directory
cd telegram-logging-ai

# Run the Qwen3 version of the bot
python -m bot.main_qwen3

echo "Bot has stopped."
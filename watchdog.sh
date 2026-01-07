#!/bin/bash

# Configuration
PROJECT_DIR="/Users/kytzu/Globals/code/Micchon/telegram-logging-ai"
VENV_ACTIVATE="$PROJECT_DIR/venv/bin/activate"
PYTHON_CMD="python -m bot.main"
LOG_FILE="$PROJECT_DIR/watchdog.log"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo "Starting Micchon Watchdog..." | tee -a "$LOG_FILE"

cd "$PROJECT_DIR" || { echo "Failed to cd to $PROJECT_DIR"; exit 1; }

while true; do
    echo -e "${GREEN}[$(date)] Starting bot...${NC}" | tee -a "$LOG_FILE"
    
    # Run the bot
    source "$VENV_ACTIVATE"
    $PYTHON_CMD >> "$LOG_FILE" 2>&1
    
    EXIT_CODE=$?
    echo -e "${RED}[$(date)] Bot crashed with exit code $EXIT_CODE. Restarting in 5 seconds...${NC}" | tee -a "$LOG_FILE"
    
    sleep 5
done

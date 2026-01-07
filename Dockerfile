# Use Python 3.11 for better compatibility with ML libraries (Chatterbox, older numpy)
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY telegram-logging-ai/requirements.txt .

# Install dependencies
# Reinstall llama-cpp-python with CUDA support (if using NVIDIA Docker)
# For CPU only remove CMAKE_ARGS
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=on"
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir llama-cpp-python --force-reinstall --no-cache-dir

# Install Chatterbox from source
RUN pip install --no-cache-dir git+https://github.com/resemble-ai/chatterbox.git

# Copy application code
COPY telegram-logging-ai/ .

# Create data directories
RUN mkdir -p data/images data/journals data/exports data/audio models

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=/app/data

# Command
CMD ["python", "-m", "bot.main"]

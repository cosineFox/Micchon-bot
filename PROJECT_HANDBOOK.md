# Micchon / Telegram Logging AI - Project Handbook

This is the consolidated master documentation for the Micchon project. It combines the Product Requirements (PRD), Initialization Guide (INIT), Maintenance Procedures, and Implementation Logs.

**Agent Coordination:**
> **Note:** This file replaces `PRD.md`, `INIT.md`, `MAINTENANCE.md`, and `IMPLEMENTATION_LOG.md`.
> **Exceptions:** `GEMINI.md`, `CLAUDE.md`, and `QWEN.md` are preserved for agent-specific context.
> **Sync:** Check `AGENT_SYNC.md` for current task status and handoffs.

---

## 1. Project Overview

**Name:** Micchon (Telegram Logging AI with Waifu)
**Goal:** A unified personal AI companion that operates in two modes:
1.  **Waifu Mode:** An always-on conversational partner (Senko-san personality) that remembers past interactions.
2.  **Journal Mode:** A silent logger that captures text/images and compiles them into daily markdown articles.

**Key Architecture:**
*   **Unified Memory:** Two-database system (`journal.db` for drafts, `memory.db` for permanent storage).
*   **Local AI:** Runs entirely locally using `llama.cpp` (GGUF models) + `sqlite-vec` (Vector Search).
*   **Hardware:** Optimized for Consumer GPUs (RTX 3060 12GB) and Raspberry Pi 5 (Qwen3 variant).

---

## 2. Tech Stack

| Component | Choice | Reason |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Ecosystem support |
| **LLM Runtime** | `llama-cpp-python` | Fast, low VRAM, GGUF support |
| **Main Model** | Gemma-3n-E4B (Q4_K_M) | Balanced Vision + Chat (2.5GB VRAM) |
| **Embeddings** | EmbeddingGemma-300M | 256d MRL vectors (On-demand load) |
| **Vector DB** | SQLite + `sqlite-vec` | Zero-dependency, file-based |
| **TTS** | Chatterbox-Turbo | Low latency, paralinguistics (e.g. `[laugh]`) |
| **Platform** | Docker (Python 3.11) | Ensures compatibility for TTS |

---

## 3. Setup & Installation

### 3.1 Prerequisites
*   **Docker** (Recommended) with NVIDIA Container Toolkit (if using GPU).
*   **Python 3.11** (If running natively).
*   **Telegram Bot Token** (from @BotFather).
*   **Allowed Chat IDs** (User or Group IDs).

### 3.2 Quick Start (Docker)
1.  **Clone:** `git clone <repo> telegram-logging-ai`
2.  **Config:** `cp .env.example .env` and edit `TELEGRAM_ALLOWED_CHAT_IDS`.
3.  **Models:** Download `gemma-3n-e4b-q4_k_m.gguf` to `models/`.
4.  **Run:**
    ```bash
    docker build -t micchon .
    docker run --gpus all -v $(pwd)/telegram-logging-ai/data:/app/data micchon
    ```

### 3.3 Native Installation
1.  **Venv:** `python3.11 -m venv venv && source venv/bin/activate`
2.  **Deps:** `pip install -r requirements.txt` (Ensure CUDA support for llama-cpp).
3.  **Run:** `./watchdog.sh` (Auto-restarting supervisor).

---

## 4. Maintenance & Backups

### 4.1 Automated Maintenance
The bot includes a built-in `MaintenanceManager` that runs daily at **02:30 AM**.
*   **Backup:** Creates a hot copy of databases (`VACUUM INTO`) to `data/backups/`.
*   **Integrity:** Runs `PRAGMA integrity_check` to detect corruption.
*   **Cleanup:** Deletes backups older than 7 days and memories older than `AUTO_CLEANUP_DAYS`.

### 4.2 Manual Tasks
*   **Check Status:** Send `/status` to the bot.
*   **Verify Backups:** Check `data/backups/` for recent `.bak` files.
*   **Logs:** Check `watchdog.log` (if using watchdog) or Docker logs.

---

## 5. Development & Testing

### 5.1 Directory Structure
```text
Micchon/
├── telegram-logging-ai/       # Source Code
│   ├── bot/                   # Handlers, AI logic, Scheduler
│   ├── memory/                # Database Repositories
│   ├── models/                # GGUF Files
│   ├── data/                  # Databases & User Data
│   └── scripts/               # Utility scripts
├── PROJECT_HANDBOOK.md        # This file
├── AGENT_SYNC.md              # Live Agent Status
├── watchdog.sh                # Process Supervisor
└── Dockerfile                 # Container Config
```

### 5.2 Key Test Cases
*   **TC-001 Startup:** Run `./watchdog.sh` and check for "Starting Telegram polling".
*   **TC-004 Journal Mode:** Send `/journal start`, verify silent logging, then `/journal done`.
*   **TC-010 Bluesky:** Send `/bsky Test` (requires credentials).

---

## 6. Implementation History (Log)

### 2026-01-07: Core & Architecture
*   **Initial Build:** Implemented unified memory, Journal/Waifu modes, and local LLM stack.
*   **Optimization:** Added manual model loading/unloading to fit 12GB VRAM.
*   **Qwen3 Variant:** Added support files for Raspberry Pi 5 optimization (`config-qwen3.py`).

### 2026-01-07: Security & Robustness
*   **Access Control:** Switched to `TELEGRAM_ALLOWED_CHAT_IDS` to support Groups.
*   **Watchdog:** Added `watchdog.sh` for process self-healing.
*   **Docker:** Added `Dockerfile` to fix Python 3.14/Chatterbox incompatibility.
*   **Maintenance:** Added automated daily backups and integrity checks.

---

## 7. Configuration Reference (`config.py`)
*   `TELEGRAM_ALLOWED_CHAT_IDS`: Whitelist of users/groups.
*   `TTS_ENABLED`: Set to `True` only if running in Docker/Python 3.11.
*   `Fine-tuning`: Disabled by default (resource intensive).

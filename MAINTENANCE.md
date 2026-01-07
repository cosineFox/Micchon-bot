# Maintenance & Test Cases

Testing procedures and maintenance tasks for the Telegram Logging AI with Waifu.

---

## Test Cases

### TC-001: Bot Startup
**Purpose:** Verify bot starts without errors

```bash
python -m bot.main
```

**Expected:**
- [ ] No import errors
- [ ] Databases initialize
- [ ] Model loads successfully
- [ ] "Starting Telegram polling..." appears

**Failure actions:**
- Check Python dependencies: `pip install -r requirements.txt`
- Check model path in config.py
- Check .env file exists and has valid token

---

### TC-002: Normal Mode - Text Response
**Purpose:** Verify waifu responds to messages

**Steps:**
1. Send any text message to bot
2. Wait for response

**Expected:**
- [ ] Bot acknowledges within 2 seconds
- [ ] Response has Senko-san personality
- [ ] Message stored in memory.db

**Verify storage:**
```bash
sqlite3 data/memory.db "SELECT * FROM memories ORDER BY created_at DESC LIMIT 2;"
```

---

### TC-003: Normal Mode - Image Description
**Purpose:** Verify image processing

**Steps:**
1. Send an image to the bot
2. Wait for response

**Expected:**
- [ ] Bot acknowledges "Processing image..."
- [ ] Returns description of image
- [ ] Waifu comments on the image
- [ ] Image saved to `data/images/`

---

### TC-004: Journal Mode - Start
**Purpose:** Verify journal mode activation

**Steps:**
1. Send `/journal start`
2. Verify response

**Expected:**
- [ ] Bot confirms journal mode started
- [ ] Subsequent messages get silent `✓` confirmation
- [ ] No waifu responses during journal mode

---

### TC-005: Journal Mode - Entries
**Purpose:** Verify silent logging

**Steps:**
1. Start journal mode (`/journal start`)
2. Send 3 text messages
3. Send 1 image
4. Check `/journal status`

**Expected:**
- [ ] Each entry gets `✓` only
- [ ] Status shows 4 entries
- [ ] Entries stored in journal.db

**Verify:**
```bash
sqlite3 data/journal.db "SELECT * FROM journal_entries;"
```

---

### TC-006: Journal Mode - Compile
**Purpose:** Verify journal compilation

**Steps:**
1. Have entries in journal mode
2. Send `/journal done`
3. Wait for compilation

**Expected:**
- [ ] AI generates title
- [ ] AI writes cohesive article
- [ ] Tags extracted
- [ ] Markdown file created in `data/journals/`
- [ ] Journal stored in memory.db
- [ ] journal.db cleared
- [ ] Returns to normal mode

**Verify:**
```bash
ls -la data/journals/
sqlite3 data/memory.db "SELECT * FROM journals ORDER BY created_at DESC LIMIT 1;"
```

---

### TC-007: Journal Mode - Cancel
**Purpose:** Verify clean cancellation

**Steps:**
1. Start journal mode
2. Add some entries
3. Send `/journal cancel`

**Expected:**
- [ ] Confirms cancellation
- [ ] journal.db cleared
- [ ] Returns to normal mode
- [ ] No article created

---

### TC-008: Voice Responses
**Purpose:** Verify TTS functionality

**Steps:**
1. Enable voice: `/voice`
2. Send a message
3. Receive response

**Expected:**
- [ ] Voice toggle confirmed
- [ ] Response includes voice message
- [ ] Audio plays correctly

**Skip if:** TTS_ENABLED = False

---

### TC-009: Rating System
**Purpose:** Verify response rating

**Steps:**
1. Get a waifu response
2. Rate it: `/rate 5`

**Expected:**
- [ ] Rating confirmed
- [ ] Rating stored in memory.db

**Verify:**
```bash
sqlite3 data/memory.db "SELECT * FROM user_feedback ORDER BY created_at DESC LIMIT 1;"
```

---

### TC-010: Bluesky Posting
**Purpose:** Verify Bluesky integration

**Steps:**
1. Send `/bsky Test post from waifu bot`
2. Wait for response

**Expected:**
- [ ] AI polishes text
- [ ] Post appears on Bluesky
- [ ] Post URI returned
- [ ] Stored as memory

**Skip if:** Bluesky credentials not configured

---

### TC-011: Semantic Search
**Purpose:** Verify vector search works

**Steps:**
1. Have multiple memories stored
2. Send a message that should trigger relevant memories
3. Check if waifu references past context

**Expected:**
- [ ] Waifu mentions relevant past events
- [ ] Context feels natural

**Debug:**
```python
from memory.master_repo import MasterRepository
import asyncio

async def test():
    repo = MasterRepository("data/memory.db")
    await repo.init_db()
    results = await repo.search_similar("your test query", limit=5)
    for r in results:
        print(r.content[:100])

asyncio.run(test())
```

---

### TC-012: Status Command
**Purpose:** Verify system status

**Steps:**
1. Send `/status`

**Expected:**
- [ ] Shows current mode
- [ ] Shows memory count
- [ ] Shows model status
- [ ] Shows TTS status

---

## Integration Tests

### INT-001: Full Conversation Flow
1. Start fresh (delete data/*.db)
2. `/start`
3. Send 5 varied messages
4. `/journal start`
5. Log 3 entries
6. `/journal done`
7. Send 2 more messages
8. Verify waifu references journal

### INT-002: Long Session
1. Keep bot running for 1 hour
2. Send messages periodically
3. Verify no memory leaks
4. Check VRAM usage stable

### INT-003: Restart Recovery
1. Start journal mode
2. Add entries
3. Kill bot process (Ctrl+C)
4. Restart bot
5. Verify journal session state

---

## Maintenance Tasks

### Daily
- [ ] Check bot is running
- [ ] Review error logs
- [ ] Verify disk space

### Weekly
- [ ] Backup databases
  ```bash
  cp data/memory.db data/backups/memory_$(date +%Y%m%d).db
  ```
- [ ] Check VRAM usage
- [ ] Review fine-tuning queue (if enabled)

### Monthly
- [ ] Clean old audio files
  ```bash
  find data/audio -mtime +30 -delete
  ```
- [ ] Vacuum databases
  ```bash
  sqlite3 data/memory.db "VACUUM;"
  sqlite3 data/journal.db "VACUUM;"
  ```
- [ ] Check for dependency updates
- [ ] Review memory growth

---

## Database Maintenance

### Vacuum Databases
```bash
sqlite3 data/memory.db "VACUUM;"
sqlite3 data/journal.db "VACUUM;"
```

### Check Database Size
```bash
du -h data/*.db
```

### Export Memories (Backup)
```bash
sqlite3 data/memory.db ".dump" > backup_memory.sql
```

### Count Records
```bash
sqlite3 data/memory.db "
SELECT 'memories' as tbl, COUNT(*) FROM memories
UNION ALL
SELECT 'journals', COUNT(*) FROM journals
UNION ALL
SELECT 'embeddings', COUNT(*) FROM memory_embeddings;
"
```

### Clear Old Embeddings (90+ days)
```sql
-- Run manually if needed
DELETE FROM memory_embeddings
WHERE id IN (
    SELECT id FROM memories
    WHERE created_at < datetime('now', '-90 days')
);
```

---

## Log Analysis

### Check Recent Errors
```bash
grep -i error bot.log | tail -20
```

### Monitor Live
```bash
tail -f bot.log
```

### Count Errors by Type
```bash
grep -i error bot.log | cut -d: -f4 | sort | uniq -c | sort -rn
```

---

## Performance Monitoring

### Check VRAM Usage
```bash
nvidia-smi
```

### Check Memory Usage
```bash
ps aux | grep python | grep bot
```

### Model Load Time
Add to startup:
```python
import time
start = time.time()
await model_manager.load_model()
print(f"Model loaded in {time.time() - start:.2f}s")
```

---

## Known Issues

| Issue | Workaround | Status |
|-------|------------|--------|
| LoRA adapters don't merge to GGUF | Manual merge required | Open |
| Vision model needs verification | Test with image | Open |
| AUTO_CLEANUP_DAYS not implemented | Manual cleanup | Open |
| Journal recovery after crash | Check journal.db on restart | Open |

---

## Health Checks

### Quick Health Check Script
```python
#!/usr/bin/env python3
import asyncio
import sys
sys.path.insert(0, 'telegram-logging-ai')

async def health_check():
    errors = []

    # Check config
    try:
        import config
        if not config.TELEGRAM_BOT_TOKEN:
            errors.append("Missing TELEGRAM_BOT_TOKEN")
        if not config.MAIN_MODEL_PATH.exists():
            errors.append(f"Model not found: {config.MAIN_MODEL_PATH}")
    except Exception as e:
        errors.append(f"Config error: {e}")

    # Check databases
    try:
        from memory.master_repo import MasterRepository
        repo = MasterRepository(config.MASTER_DB_PATH)
        await repo.init_db()
        count = await repo.get_memory_count()
        print(f"✓ Memory DB: {count} memories")
    except Exception as e:
        errors.append(f"Database error: {e}")

    # Check model loading
    try:
        from bot.model_manager import ModelManager
        mm = ModelManager(config.MAIN_MODEL_PATH, config.LLAMA_PARAMS)
        # Don't actually load, just verify import
        print("✓ Model manager importable")
    except Exception as e:
        errors.append(f"Model manager error: {e}")

    if errors:
        print("\n❌ Health check failed:")
        for e in errors:
            print(f"  - {e}")
        return False
    else:
        print("\n✓ All checks passed")
        return True

if __name__ == "__main__":
    result = asyncio.run(health_check())
    sys.exit(0 if result else 1)
```

---

## Rollback Procedures

### Rollback Database
```bash
# Stop bot first
cp data/backups/memory_YYYYMMDD.db data/memory.db
# Restart bot
```

### Rollback Model
```bash
# Keep previous model versions
mv models/gemma-3n-e4b-q4_k_m.gguf models/gemma-3n-e4b-q4_k_m.gguf.new
mv models/gemma-3n-e4b-q4_k_m.gguf.old models/gemma-3n-e4b-q4_k_m.gguf
```

### Factory Reset
```bash
rm -rf data/
python -m bot.main  # Recreates fresh
```

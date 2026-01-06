import aiosqlite
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

from .models import Memory, Journal


class MasterRepository:
    """Repository for memory.db - master database with all finalized memories and journals"""

    def __init__(self, db_path: Path, embedding_dimension: int = 256):
        self.db_path = db_path
        self.embedding_dimension = embedding_dimension

    async def _apply_optimizations(self, db: aiosqlite.Connection):
        """Apply SQLite PRAGMA optimizations for performance"""
        await db.execute("PRAGMA journal_mode = WAL")  # Write-ahead logging
        await db.execute("PRAGMA synchronous = NORMAL")  # Faster writes
        await db.execute("PRAGMA cache_size = -64000")  # 64MB cache
        await db.execute("PRAGMA temp_store = MEMORY")  # Temp tables in RAM
        await db.execute("PRAGMA mmap_size = 268435456")  # 256MB memory-mapped I/O

    async def init_db(self):
        """Initialize master database schema with optimizations"""
        async with aiosqlite.connect(self.db_path) as db:
            await self._apply_optimizations(db)

            # Main memories table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    raw_content TEXT,
                    media_path TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Vector embeddings table (will be created via sqlite-vec)
            # Note: sqlite-vec uses vec0() virtual table
            try:
                await db.execute(f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS memory_embeddings
                    USING vec0(
                        id TEXT PRIMARY KEY,
                        embedding FLOAT[{self.embedding_dimension}]
                    )
                """)
            except Exception as e:
                # sqlite-vec might not be installed yet
                print(f"Warning: Could not create vector table: {e}")

            # Compiled journals table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS journals (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    body TEXT NOT NULL,
                    source_entry_ids TEXT,
                    tags TEXT,
                    markdown_path TEXT,
                    created_at TEXT NOT NULL
                )
            """)

            # Feedback table for reinforcement learning
            await db.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    memory_id TEXT NOT NULL,
                    rating INTEGER NOT NULL,
                    context TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (memory_id) REFERENCES memories(id)
                )
            """)

            # Create indexes
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp DESC)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_journals_created ON journals(created_at DESC)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback(rating DESC)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_feedback_created ON feedback(created_at DESC)"
            )

            await db.commit()

    async def add_memory(
        self,
        type: str,
        content: str,
        raw_content: Optional[str] = None,
        media_path: Optional[str] = None,
        metadata: Optional[dict] = None,
        embedding: Optional[list[float]] = None
    ) -> Memory:
        """Add a new memory to the master database"""
        memory = Memory.create(
            type=type,
            content=content,
            raw_content=raw_content,
            media_path=media_path,
            metadata=metadata
        )

        async with aiosqlite.connect(self.db_path) as db:
            await self._apply_optimizations(db)

            # Insert memory
            await db.execute(
                """INSERT INTO memories
                   (id, timestamp, type, content, raw_content, media_path, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    memory.id,
                    memory.timestamp.isoformat(),
                    memory.type,
                    memory.content,
                    memory.raw_content,
                    memory.media_path,
                    json.dumps(memory.metadata)
                )
            )

            # Insert embedding if provided
            if embedding:
                try:
                    embedding_str = "[" + ",".join(map(str, embedding)) + "]"
                    await db.execute(
                        "INSERT INTO memory_embeddings (id, embedding) VALUES (?, ?)",
                        (memory.id, embedding_str)
                    )
                except Exception as e:
                    print(f"Warning: Could not insert embedding: {e}")

            await db.commit()
        return memory

    async def get_recent(self, hours: int = 24, limit: int = 50) -> list[Memory]:
        """Get recent memories within the specified time window"""
        cutoff = datetime.now() - timedelta(hours=hours)
        memories = []

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT * FROM memories
                   WHERE timestamp >= ?
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (cutoff.isoformat(), limit)
            ) as cursor:
                async for row in cursor:
                    memories.append(self._row_to_memory(row))
        return memories

    async def get_by_type(self, type: str, limit: int = 20) -> list[Memory]:
        """Get memories filtered by type"""
        memories = []

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT * FROM memories
                   WHERE type = ?
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (type, limit)
            ) as cursor:
                async for row in cursor:
                    memories.append(self._row_to_memory(row))
        return memories

    async def search_similar(self, query_embedding: list[float], limit: int = 10) -> list[Memory]:
        """Search for semantically similar memories using vector similarity"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row

                # Use sqlite-vec for similarity search
                embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

                async with db.execute(
                    f"""SELECT m.* FROM memories m
                       JOIN (
                           SELECT id,
                                  vec_distance_cosine(embedding, ?) as distance
                           FROM memory_embeddings
                           ORDER BY distance
                           LIMIT ?
                       ) e ON m.id = e.id
                       ORDER BY e.distance""",
                    (embedding_str, limit)
                ) as cursor:
                    memories = []
                    async for row in cursor:
                        memories.append(self._row_to_memory(row))
                    return memories
        except Exception as e:
            print(f"Warning: Vector search failed: {e}")
            # Fallback to recent memories
            return await self.get_recent(hours=168, limit=limit)  # Last week

    async def add_journal(
        self,
        title: str,
        body: str,
        source_entry_ids: list[str],
        tags: Optional[list[str]] = None,
        markdown_path: Optional[str] = None
    ) -> Journal:
        """Add a compiled journal to the database"""
        journal = Journal.create(
            title=title,
            body=body,
            source_entry_ids=source_entry_ids,
            tags=tags,
            markdown_path=markdown_path
        )

        async with aiosqlite.connect(self.db_path) as db:
            await self._apply_optimizations(db)

            await db.execute(
                """INSERT INTO journals
                   (id, title, body, source_entry_ids, tags, markdown_path, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    journal.id,
                    journal.title,
                    journal.body,
                    json.dumps(journal.source_entry_ids),
                    json.dumps(journal.tags),
                    journal.markdown_path,
                    journal.created_at.isoformat()
                )
            )
            await db.commit()
        return journal

    async def get_recent_journals(self, limit: int = 5) -> list[Journal]:
        """Get recent compiled journals"""
        journals = []

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM journals ORDER BY created_at DESC LIMIT ?",
                (limit,)
            ) as cursor:
                async for row in cursor:
                    journals.append(self._row_to_journal(row))
        return journals

    async def update_journal(self, journal_id: str, **kwargs) -> bool:
        """Update specific fields of a journal"""
        if not kwargs:
            return False

        set_clause = ", ".join(f"{k} = ?" for k in kwargs.keys())
        values = list(kwargs.values()) + [journal_id]

        async with aiosqlite.connect(self.db_path) as db:
            result = await db.execute(
                f"UPDATE journals SET {set_clause} WHERE id = ?",
                values
            )
            await db.commit()
            return result.rowcount > 0

    async def delete_old_embeddings(self, days: int = 90):
        """Delete embeddings for memories older than specified days"""
        cutoff = datetime.now() - timedelta(days=days)

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """DELETE FROM memory_embeddings
                   WHERE id IN (
                       SELECT id FROM memories
                       WHERE timestamp < ?
                   )""",
                (cutoff.isoformat(),)
            )
            await db.commit()

    def _row_to_memory(self, row: aiosqlite.Row) -> Memory:
        """Convert database row to Memory object"""
        return Memory(
            id=row["id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            type=row["type"],
            content=row["content"],
            raw_content=row["raw_content"],
            media_path=row["media_path"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {}
        )

    def _row_to_journal(self, row: aiosqlite.Row) -> Journal:
        """Convert database row to Journal object"""
        return Journal(
            id=row["id"],
            title=row["title"],
            body=row["body"],
            source_entry_ids=json.loads(row["source_entry_ids"]) if row["source_entry_ids"] else [],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            markdown_path=row["markdown_path"],
            created_at=datetime.fromisoformat(row["created_at"])
        )

    async def get_memory_count(self) -> int:
        """Get total number of memories"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT COUNT(*) FROM memories") as cursor:
                row = await cursor.fetchone()
                return row[0] if row else 0

    async def add_feedback(self, memory_id: str, rating: int, context: Optional[str] = None) -> str:
        """Add user feedback/rating for a memory (for RL fine-tuning)"""
        from uuid import uuid4
        feedback_id = str(uuid4())

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO feedback (id, memory_id, rating, context) VALUES (?, ?, ?, ?)",
                (feedback_id, memory_id, rating, context)
            )
            await db.commit()
        return feedback_id

    async def get_positive_examples(self, min_rating: int = 4, limit: int = 1000) -> list[dict]:
        """Get highly-rated examples for fine-tuning"""
        examples = []

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT m.*, f.rating, f.context as feedback_context
                   FROM memories m
                   JOIN feedback f ON m.id = f.memory_id
                   WHERE f.rating >= ? AND m.type = 'chat'
                   ORDER BY f.created_at DESC
                   LIMIT ?""",
                (min_rating, limit)
            ) as cursor:
                async for row in cursor:
                    examples.append({
                        "memory_id": row["id"],
                        "content": row["content"],
                        "context": row["feedback_context"],
                        "rating": row["rating"],
                        "timestamp": row["timestamp"]
                    })
        return examples

import aiosqlite
from pathlib import Path
from datetime import datetime
from typing import Optional

from .models import JournalEntry


class JournalRepository:
    """Repository for journal.db - temporary draft entries during journal mode"""

    def __init__(self, db_path: Path):
        self.db_path = db_path

    async def init_db(self):
        """Initialize journal database schema"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS journal_entries (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    media_path TEXT,
                    image_description TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_journal_timestamp ON journal_entries(timestamp DESC)"
            )
            await db.commit()

    async def add_entry(self, entry: JournalEntry) -> JournalEntry:
        """Add a new journal entry"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT INTO journal_entries
                   (id, timestamp, type, content, media_path, image_description)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    entry.id,
                    entry.timestamp.isoformat(),
                    entry.type,
                    entry.content,
                    entry.media_path,
                    entry.image_description
                )
            )
            await db.commit()
        return entry

    async def get_all_entries(self) -> list[JournalEntry]:
        """Get all entries from current journal session, ordered by timestamp"""
        entries = []
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM journal_entries ORDER BY timestamp ASC"
            ) as cursor:
                async for row in cursor:
                    entries.append(JournalEntry(
                        id=row["id"],
                        timestamp=datetime.fromisoformat(row["timestamp"]),
                        type=row["type"],
                        content=row["content"],
                        media_path=row["media_path"],
                        image_description=row["image_description"]
                    ))
        return entries

    async def get_entry_count(self) -> int:
        """Get total number of entries in current session"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT COUNT(*) FROM journal_entries") as cursor:
                row = await cursor.fetchone()
                return row[0] if row else 0

    async def get_latest_entry(self) -> Optional[JournalEntry]:
        """Get the most recent entry"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM journal_entries ORDER BY timestamp DESC LIMIT 1"
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return JournalEntry(
                        id=row["id"],
                        timestamp=datetime.fromisoformat(row["timestamp"]),
                        type=row["type"],
                        content=row["content"],
                        media_path=row["media_path"],
                        image_description=row["image_description"]
                    )
        return None

    async def update_entry(self, entry_id: str, **kwargs) -> bool:
        """Update specific fields of an entry"""
        if not kwargs:
            return False

        set_clause = ", ".join(f"{k} = ?" for k in kwargs.keys())
        values = list(kwargs.values()) + [entry_id]

        async with aiosqlite.connect(self.db_path) as db:
            result = await db.execute(
                f"UPDATE journal_entries SET {set_clause} WHERE id = ?",
                values
            )
            await db.commit()
            return result.rowcount > 0

    async def delete_entry(self, entry_id: str) -> bool:
        """Delete a specific entry"""
        async with aiosqlite.connect(self.db_path) as db:
            result = await db.execute(
                "DELETE FROM journal_entries WHERE id = ?",
                (entry_id,)
            )
            await db.commit()
            return result.rowcount > 0

    async def clear_all(self):
        """Clear all entries from journal.db (used after compilation)"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM journal_entries")
            await db.commit()

    async def has_entries(self) -> bool:
        """Check if there are any entries in the current session"""
        count = await self.get_entry_count()
        return count > 0

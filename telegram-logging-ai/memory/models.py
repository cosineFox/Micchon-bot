from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from uuid import uuid4


@dataclass
class JournalEntry:
    """Draft entry stored in journal.db during active journaling session"""
    id: str
    timestamp: datetime
    type: str  # "text", "image"
    content: str
    media_path: Optional[str] = None
    image_description: Optional[str] = None

    @classmethod
    def create(cls, type: str, content: str, media_path: Optional[str] = None) -> "JournalEntry":
        """Factory method to create a new journal entry with auto-generated ID"""
        return cls(
            id=str(uuid4()),
            timestamp=datetime.now(),
            type=type,
            content=content,
            media_path=media_path
        )


@dataclass
class Memory:
    """Finalized memory stored in memory.db - waifu reads from here"""
    id: str
    timestamp: datetime
    type: str  # "journal", "image", "chat", "bsky_post"
    content: str
    raw_content: Optional[str] = None
    media_path: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        type: str,
        content: str,
        raw_content: Optional[str] = None,
        media_path: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> "Memory":
        """Factory method to create a new memory with auto-generated ID and timestamp"""
        return cls(
            id=str(uuid4()),
            timestamp=datetime.now(),
            type=type,
            content=content,
            raw_content=raw_content,
            media_path=media_path,
            metadata=metadata or {}
        )


@dataclass
class Journal:
    """Compiled journal article stored in memory.db"""
    id: str
    title: str
    body: str  # AI-written article (markdown)
    source_entry_ids: list[str]  # Original JournalEntry IDs
    created_at: datetime
    tags: list[str] = field(default_factory=list)
    markdown_path: Optional[str] = None

    @classmethod
    def create(
        cls,
        title: str,
        body: str,
        source_entry_ids: list[str],
        tags: Optional[list[str]] = None,
        markdown_path: Optional[str] = None
    ) -> "Journal":
        """Factory method to create a new journal with auto-generated ID and timestamp"""
        return cls(
            id=str(uuid4()),
            title=title,
            body=body,
            source_entry_ids=source_entry_ids,
            created_at=datetime.now(),
            tags=tags or [],
            markdown_path=markdown_path
        )


@dataclass
class WaifuContext:
    """Runtime context for waifu responses - not persisted"""
    recent_memories: list[Memory]
    relevant_memories: list[Memory]
    recent_journals: list[Journal]
    user_profile: dict = field(default_factory=dict)

    def get_all_memories(self) -> list[Memory]:
        """Get all memories (recent + relevant) with duplicates removed"""
        seen_ids = set()
        all_memories = []

        for memory in self.recent_memories + self.relevant_memories:
            if memory.id not in seen_ids:
                seen_ids.add(memory.id)
                all_memories.append(memory)

        return sorted(all_memories, key=lambda m: m.timestamp)


@dataclass
class SessionState:
    """Runtime session state tracking current mode - not persisted"""
    user_id: int
    mode: str  # "normal" or "journal"
    journal_started_at: Optional[datetime] = None

    def is_journal_mode(self) -> bool:
        """Check if currently in journal mode"""
        return self.mode == "journal"

    def is_normal_mode(self) -> bool:
        """Check if currently in normal mode"""
        return self.mode == "normal"

    def start_journal(self):
        """Enter journal mode"""
        self.mode = "journal"
        self.journal_started_at = datetime.now()

    def exit_journal(self):
        """Exit journal mode"""
        self.mode = "normal"
        self.journal_started_at = None

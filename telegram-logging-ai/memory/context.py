from datetime import datetime
from typing import Optional
import logging

from .models import Memory, Journal, WaifuContext
from .master_repo import MasterRepository
from .embedder import get_embedder

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Builds optimized context windows for waifu responses"""

    def __init__(
        self,
        master_repo: MasterRepository,
        context_hours: int = 24,
        max_recent_messages: int = 10,
        max_relevant_memories: int = 5,
        max_journals: int = 3
    ):
        """
        Initialize context builder

        Args:
            master_repo: Master memory repository
            context_hours: Hours of recent memory to include
            max_recent_messages: Maximum recent chat messages
            max_relevant_memories: Maximum semantic search results
            max_journals: Maximum recent journals to include
        """
        self.repo = master_repo
        self.context_hours = context_hours
        self.max_recent_messages = max_recent_messages
        self.max_relevant_memories = max_relevant_memories
        self.max_journals = max_journals
        self.embedder = get_embedder()

    async def build_context(
        self,
        query: str,
        user_id: Optional[int] = None
    ) -> WaifuContext:
        """
        Build full context for waifu response generation

        Args:
            query: Current user message (for semantic search)
            user_id: User ID for profile lookup

        Returns:
            WaifuContext with all relevant memories
        """
        logger.debug(f"Building context for query: {query[:50]}...")

        # 1. Get recent memories (last N hours)
        recent_memories = await self._get_recent_memories()

        # 2. Semantic search for relevant older memories
        relevant_memories = await self._search_relevant(query)

        # 3. Get recent journals
        recent_journals = await self.repo.get_recent_journals(
            limit=self.max_journals
        )

        # 4. Build user profile (future: learned preferences)
        user_profile = await self._build_user_profile(user_id)

        context = WaifuContext(
            recent_memories=recent_memories,
            relevant_memories=relevant_memories,
            recent_journals=recent_journals,
            user_profile=user_profile
        )

        logger.debug(
            f"Context built: {len(recent_memories)} recent, "
            f"{len(relevant_memories)} relevant, "
            f"{len(recent_journals)} journals"
        )

        return context

    async def _get_recent_memories(self) -> list[Memory]:
        """Get recent memories, prioritizing chat messages"""
        # Get all recent memories
        all_recent = await self.repo.get_recent(
            hours=self.context_hours,
            limit=50
        )

        # Separate chat messages from other memories
        chat_messages = [m for m in all_recent if m.type == "chat"]
        other_memories = [m for m in all_recent if m.type != "chat"]

        # Prioritize recent chat messages (for conversation flow)
        recent_chat = chat_messages[:self.max_recent_messages]

        # Add some other memories for context variety
        other_limit = max(0, self.max_recent_messages - len(recent_chat))
        recent_other = other_memories[:other_limit]

        # Combine and sort by timestamp
        combined = recent_chat + recent_other
        combined.sort(key=lambda m: m.timestamp)

        return combined

    async def _search_relevant(self, query: str) -> list[Memory]:
        """Search for semantically relevant memories"""
        if not query.strip():
            return []

        try:
            # Generate query embedding
            query_embedding = await self.embedder.embed(query)

            # Search for similar memories
            relevant = await self.repo.search_similar(
                query_embedding,
                limit=self.max_relevant_memories
            )

            # Filter out memories that are already in recent
            # (deduplication happens in WaifuContext.get_all_memories)
            return relevant

        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")
            return []

    async def _build_user_profile(
        self,
        user_id: Optional[int]
    ) -> dict:
        """
        Build user profile from memory patterns

        Future: Learn preferences, topics of interest, etc.
        """
        # Placeholder for learned preferences
        profile = {
            "user_id": user_id,
            "learned_topics": [],
            "preferences": {},
        }

        return profile

    def format_context_for_prompt(
        self,
        context: WaifuContext,
        max_tokens: int = 1500
    ) -> str:
        """
        Format context into a prompt string for the LLM

        Args:
            context: WaifuContext to format
            max_tokens: Approximate token budget for context

        Returns:
            Formatted context string
        """
        parts = []

        # Add relevant memories (older but semantically related)
        if context.relevant_memories:
            parts.append("=== Relevant Past Memories ===")
            for mem in context.relevant_memories:
                time_str = mem.timestamp.strftime("%Y-%m-%d %H:%M")
                parts.append(f"[{time_str}] ({mem.type}) {mem.content[:200]}")

        # Add recent journals the waifu wrote
        if context.recent_journals:
            parts.append("\n=== Recent Journals You Compiled ===")
            for journal in context.recent_journals:
                date_str = journal.created_at.strftime("%Y-%m-%d")
                parts.append(f"[{date_str}] {journal.title}")

        # Add recent conversation/activity
        if context.recent_memories:
            parts.append("\n=== Recent Activity ===")
            for mem in context.recent_memories:
                time_str = mem.timestamp.strftime("%H:%M")
                if mem.type == "chat":
                    role = mem.metadata.get("role", "user")
                    parts.append(f"[{time_str}] {role}: {mem.content}")
                else:
                    parts.append(f"[{time_str}] ({mem.type}) {mem.content[:150]}")

        return "\n".join(parts)

    async def get_conversation_history(
        self,
        limit: int = 10
    ) -> list[dict]:
        """
        Get recent conversation as chat messages format

        Returns:
            List of {"role": "user/assistant", "content": "..."}
        """
        chat_memories = await self.repo.get_by_type("chat", limit=limit * 2)

        messages = []
        for mem in reversed(chat_memories):  # Oldest first
            role = mem.metadata.get("role", "user")
            messages.append({
                "role": role,
                "content": mem.content
            })

        # Take last N messages
        return messages[-limit:]


# Global singleton
_context_builder: Optional[ContextBuilder] = None


def get_context_builder(
    master_repo: MasterRepository,
    context_hours: int = 24,
    max_recent_messages: int = 10,
    max_relevant_memories: int = 5
) -> ContextBuilder:
    """Get or create global context builder"""
    global _context_builder

    if _context_builder is None:
        _context_builder = ContextBuilder(
            master_repo=master_repo,
            context_hours=context_hours,
            max_recent_messages=max_recent_messages,
            max_relevant_memories=max_relevant_memories
        )

    return _context_builder

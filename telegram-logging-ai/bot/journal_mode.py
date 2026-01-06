import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Callable, Awaitable
import logging

from memory.journal_repo import JournalRepository
from memory.models import JournalEntry, SessionState

logger = logging.getLogger(__name__)


class JournalModeManager:
    """Manages journal mode sessions for users"""

    def __init__(
        self,
        journal_repo: JournalRepository,
        idle_warning_minutes: int = 60,
        auto_compile_hours: int = 24,
        on_idle_warning: Optional[Callable[[int], Awaitable[None]]] = None,
        on_auto_compile: Optional[Callable[[int], Awaitable[None]]] = None
    ):
        """
        Initialize journal mode manager

        Args:
            journal_repo: Journal repository for storing entries
            idle_warning_minutes: Minutes of idle before warning
            auto_compile_hours: Hours before auto-compile
            on_idle_warning: Callback when user is idle
            on_auto_compile: Callback when auto-compiling
        """
        self.repo = journal_repo
        self.idle_warning_minutes = idle_warning_minutes
        self.auto_compile_hours = auto_compile_hours
        self.on_idle_warning = on_idle_warning
        self.on_auto_compile = on_auto_compile

        # Session state per user
        self._sessions: dict[int, SessionState] = {}

        # Track last activity per user
        self._last_activity: dict[int, datetime] = {}

        # Track if idle warning was sent
        self._idle_warned: set[int] = set()

        # Background task for monitoring
        self._monitor_task: Optional[asyncio.Task] = None

    async def start_monitoring(self):
        """Start background monitoring for idle sessions"""
        if self._monitor_task is None:
            self._monitor_task = asyncio.create_task(self._monitor_sessions())
            logger.info("Journal session monitoring started")

    async def stop_monitoring(self):
        """Stop background monitoring"""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
            logger.info("Journal session monitoring stopped")

    async def _monitor_sessions(self):
        """Monitor sessions for idle users and auto-compile"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute

                now = datetime.now()
                users_to_check = list(self._sessions.keys())

                for user_id in users_to_check:
                    session = self._sessions.get(user_id)
                    if not session or not session.is_journal_mode():
                        continue

                    last_activity = self._last_activity.get(user_id)
                    if not last_activity:
                        continue

                    idle_minutes = (now - last_activity).total_seconds() / 60

                    # Check for idle warning
                    if (idle_minutes >= self.idle_warning_minutes and
                            user_id not in self._idle_warned):
                        self._idle_warned.add(user_id)
                        if self.on_idle_warning:
                            await self.on_idle_warning(user_id)
                        logger.info(f"Idle warning sent to user {user_id}")

                    # Check for auto-compile
                    if session.journal_started_at:
                        session_hours = (now - session.journal_started_at).total_seconds() / 3600
                        if session_hours >= self.auto_compile_hours:
                            if self.on_auto_compile:
                                await self.on_auto_compile(user_id)
                            logger.info(f"Auto-compile triggered for user {user_id}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in session monitor: {e}")

    def start_journal(self, user_id: int) -> SessionState:
        """
        Start journal mode for a user

        Args:
            user_id: Telegram user ID

        Returns:
            Updated session state
        """
        session = self._sessions.get(user_id)
        if session is None:
            session = SessionState(user_id=user_id, mode="normal")
            self._sessions[user_id] = session

        session.start_journal()
        self._last_activity[user_id] = datetime.now()
        self._idle_warned.discard(user_id)

        logger.info(f"User {user_id} started journal mode")
        return session

    def end_journal(self, user_id: int) -> SessionState:
        """
        End journal mode for a user

        Args:
            user_id: Telegram user ID

        Returns:
            Updated session state
        """
        session = self._sessions.get(user_id)
        if session:
            session.exit_journal()
            self._idle_warned.discard(user_id)

        logger.info(f"User {user_id} ended journal mode")
        return session or SessionState(user_id=user_id, mode="normal")

    def is_journal_mode(self, user_id: int) -> bool:
        """Check if user is in journal mode"""
        session = self._sessions.get(user_id)
        return session is not None and session.is_journal_mode()

    def get_session(self, user_id: int) -> SessionState:
        """Get or create session state for user"""
        if user_id not in self._sessions:
            self._sessions[user_id] = SessionState(user_id=user_id, mode="normal")
        return self._sessions[user_id]

    async def add_text_entry(
        self,
        user_id: int,
        content: str
    ) -> JournalEntry:
        """
        Add a text entry to the journal

        Args:
            user_id: User ID
            content: Text content

        Returns:
            Created journal entry
        """
        entry = JournalEntry.create(type="text", content=content)
        await self.repo.add_entry(entry)

        self._last_activity[user_id] = datetime.now()
        self._idle_warned.discard(user_id)

        logger.debug(f"Added text entry for user {user_id}")
        return entry

    async def add_image_entry(
        self,
        user_id: int,
        image_path: Path,
        description: str,
        context: Optional[str] = None
    ) -> JournalEntry:
        """
        Add an image entry to the journal

        Args:
            user_id: User ID
            image_path: Path to image file
            description: AI-generated description
            context: User-provided context

        Returns:
            Created journal entry
        """
        entry = JournalEntry.create(
            type="image",
            content=context or "",
            media_path=str(image_path)
        )
        entry.image_description = description

        await self.repo.add_entry(entry)

        self._last_activity[user_id] = datetime.now()
        self._idle_warned.discard(user_id)

        logger.debug(f"Added image entry for user {user_id}")
        return entry

    async def get_entry_count(self, user_id: int) -> int:
        """Get number of entries in current session"""
        return await self.repo.get_entry_count()

    async def get_session_info(self, user_id: int) -> dict:
        """
        Get information about current journal session

        Returns:
            Dictionary with session info
        """
        session = self.get_session(user_id)

        if not session.is_journal_mode():
            return {
                "active": False,
                "mode": "normal"
            }

        entry_count = await self.repo.get_entry_count()
        latest = await self.repo.get_latest_entry()

        started_at = session.journal_started_at
        duration = None
        if started_at:
            duration = datetime.now() - started_at

        return {
            "active": True,
            "mode": "journal",
            "started_at": started_at,
            "duration_minutes": int(duration.total_seconds() / 60) if duration else 0,
            "entry_count": entry_count,
            "latest_entry_time": latest.timestamp if latest else None
        }

    async def cancel_journal(self, user_id: int) -> int:
        """
        Cancel journal mode and discard all entries

        Args:
            user_id: User ID

        Returns:
            Number of discarded entries
        """
        count = await self.repo.get_entry_count()
        await self.repo.clear_all()
        self.end_journal(user_id)

        logger.info(f"User {user_id} cancelled journal, discarded {count} entries")
        return count


# Global singleton
_journal_manager: Optional[JournalModeManager] = None


def get_journal_manager(
    journal_repo: JournalRepository,
    idle_warning_minutes: int = 60,
    auto_compile_hours: int = 24
) -> JournalModeManager:
    """Get or create global journal mode manager"""
    global _journal_manager

    if _journal_manager is None:
        _journal_manager = JournalModeManager(
            journal_repo=journal_repo,
            idle_warning_minutes=idle_warning_minutes,
            auto_compile_hours=auto_compile_hours
        )

    return _journal_manager

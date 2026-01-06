import re
from pathlib import Path
from datetime import datetime
from typing import Optional
import logging

from .llama_client import LlamaClient
from memory.journal_repo import JournalRepository
from memory.master_repo import MasterRepository
from memory.models import Journal, JournalEntry

logger = logging.getLogger(__name__)


class JournalCompiler:
    """Compiles journal entries into AI-written articles"""

    def __init__(
        self,
        llama_client: LlamaClient,
        journal_repo: JournalRepository,
        master_repo: MasterRepository,
        output_dir: Path
    ):
        """
        Initialize journal compiler

        Args:
            llama_client: LLM client for article generation
            journal_repo: Journal draft repository
            master_repo: Master memory repository
            output_dir: Directory for exported markdown files
        """
        self.llm = llama_client
        self.journal_repo = journal_repo
        self.master_repo = master_repo
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def compile(self) -> Optional[Journal]:
        """
        Compile all draft entries into a journal article

        Returns:
            Compiled Journal object, or None if no entries
        """
        # Get all draft entries
        entries = await self.journal_repo.get_all_entries()

        if not entries:
            logger.info("No entries to compile")
            return None

        logger.info(f"Compiling {len(entries)} journal entries")

        # Format entries for LLM
        formatted = self._format_entries(entries)

        # Generate article via LLM
        prompt = f"""Based on these journal entries from a maker/developer, write a cohesive journal article.

The entries are raw logs with timestamps, text notes, and image descriptions.

=== JOURNAL ENTRIES ===
{formatted}
=== END ENTRIES ===

Generate:
1. A concise, descriptive title (without quotes or "Title:")
2. Relevant tags (comma-separated, lowercase, no hashtags)
3. A well-structured article in markdown format that:
   - Flows naturally as a narrative
   - Preserves the chronological journey
   - Highlights key moments and decisions
   - Maintains the personal voice
   - Is suitable for a blog post

Format your response EXACTLY as:
TITLE: <title>
TAGS: <tag1>, <tag2>, <tag3>
---
<article body in markdown>"""

        result = await self.llm.generate(
            prompt=prompt,
            max_tokens=1024,
            temperature=0.7
        )

        # Parse the LLM output
        title, tags, body = self._parse_output(result)

        if not title or not body:
            logger.error("Failed to parse LLM output for journal")
            # Fallback
            title = f"Journal - {datetime.now().strftime('%Y-%m-%d')}"
            body = formatted
            tags = ["journal"]

        # Export to markdown file
        markdown_path = await self._export_markdown(title, body, entries, tags)

        # Save journal to master database
        journal = await self.master_repo.add_journal(
            title=title,
            body=body,
            source_entry_ids=[e.id for e in entries],
            tags=tags,
            markdown_path=str(markdown_path)
        )

        # Also add as a memory for waifu context
        summary = body[:500] + "..." if len(body) > 500 else body
        await self.master_repo.add_memory(
            type="journal",
            content=f"{title}\n\n{summary}",
            metadata={
                "journal_id": journal.id,
                "tags": tags,
                "entry_count": len(entries)
            }
        )

        # Clear draft entries
        await self.journal_repo.clear_all()

        logger.info(f"Journal compiled: {title}")
        return journal

    def _format_entries(self, entries: list[JournalEntry]) -> str:
        """Format entries for LLM prompt"""
        parts = []

        for entry in entries:
            time_str = entry.timestamp.strftime("%Y-%m-%d %H:%M")

            if entry.type == "text":
                parts.append(f"[{time_str}] {entry.content}")
            elif entry.type == "image":
                desc = entry.image_description or "No description"
                context = entry.content or ""
                parts.append(f"[{time_str}] [IMAGE: {desc}] {context}")

        return "\n\n".join(parts)

    def _parse_output(self, output: str) -> tuple[str, list[str], str]:
        """Parse LLM output into title, tags, body"""
        title = ""
        tags = []
        body = ""

        lines = output.strip().split("\n")

        # Find TITLE line
        for i, line in enumerate(lines):
            if line.startswith("TITLE:"):
                title = line[6:].strip()
                break

        # Find TAGS line
        for i, line in enumerate(lines):
            if line.startswith("TAGS:"):
                tags_str = line[5:].strip()
                tags = [t.strip().lower() for t in tags_str.split(",") if t.strip()]
                break

        # Find body after ---
        body_start = -1
        for i, line in enumerate(lines):
            if line.strip() == "---":
                body_start = i + 1
                break

        if body_start > 0:
            body = "\n".join(lines[body_start:]).strip()

        return title, tags, body

    async def _export_markdown(
        self,
        title: str,
        body: str,
        entries: list[JournalEntry],
        tags: list[str]
    ) -> Path:
        """Export journal to markdown file"""
        # Generate filename
        date_str = datetime.now().strftime("%Y-%m-%d")
        safe_title = re.sub(r'[^\w\s-]', '', title)[:50].strip().replace(' ', '-')
        filename = f"{date_str}-{safe_title}.md"
        filepath = self.output_dir / filename

        # Build markdown content
        content = f"""# {title}

**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Entries:** {len(entries)}
**Tags:** {', '.join(tags)}

---

{body}

---

## Raw Timeline

"""

        # Add raw entries for reference
        for entry in entries:
            time_str = entry.timestamp.strftime("%H:%M")
            if entry.type == "text":
                content += f"### {time_str}\n{entry.content}\n\n"
            elif entry.type == "image":
                desc = entry.image_description or "No description"
                content += f"### {time_str} [Image]\n{desc}\n"
                if entry.content:
                    content += f"Context: {entry.content}\n"
                if entry.media_path:
                    content += f"![Image]({entry.media_path})\n"
                content += "\n"

        # Write file
        filepath.write_text(content, encoding="utf-8")
        logger.info(f"Exported markdown: {filepath}")

        return filepath

    async def get_preview(self) -> Optional[str]:
        """
        Get a preview of what would be compiled

        Returns:
            Preview text, or None if no entries
        """
        entries = await self.journal_repo.get_all_entries()

        if not entries:
            return None

        count = len(entries)
        first = entries[0].timestamp.strftime("%H:%M")
        last = entries[-1].timestamp.strftime("%H:%M")

        text_count = sum(1 for e in entries if e.type == "text")
        image_count = sum(1 for e in entries if e.type == "image")

        preview = f"**Journal Preview**\n"
        preview += f"Entries: {count} ({text_count} text, {image_count} images)\n"
        preview += f"Time: {first} - {last}\n\n"
        preview += "Recent entries:\n"

        for entry in entries[-3:]:
            time_str = entry.timestamp.strftime("%H:%M")
            content = entry.content[:50] + "..." if len(entry.content) > 50 else entry.content
            preview += f"- [{time_str}] {content}\n"

        return preview

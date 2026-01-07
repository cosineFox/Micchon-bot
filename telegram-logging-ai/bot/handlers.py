import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from PIL import Image

import config
from .waifu import Waifu
from .journal_mode import JournalModeManager
from .journal_compiler import JournalCompiler
from .bluesky_client import BlueskyClient
from .llama_client import LlamaClient

logger = logging.getLogger(__name__)


class Handlers:
    """Telegram bot handlers for two-mode system"""

    def __init__(
        self,
        waifu: Waifu,
        journal_manager: JournalModeManager,
        journal_compiler: JournalCompiler,
        bluesky_client: BlueskyClient,
        llm_client: LlamaClient,
        allowed_users: list[int],
        image_dir: Path,
        tts_enabled: bool = True
    ):
        """
        Initialize handlers

        Args:
            waifu: Waifu instance for normal mode
            journal_manager: Journal mode manager
            journal_compiler: Journal compiler
            bluesky_client: Bluesky client
            llm_client: LLM client
            allowed_users: List of allowed Telegram user IDs
            image_dir: Directory to store images
            tts_enabled: Enable voice responses
        """
        self.waifu = waifu
        self.journal = journal_manager
        self.compiler = journal_compiler
        self.bsky = bluesky_client
        self.llm = llm_client
        self.allowed_users = allowed_users
        self.image_dir = image_dir
        self.tts_enabled = tts_enabled

        # Track last response memory ID for rating
        self._last_response: dict[int, str] = {}

        self.image_dir.mkdir(parents=True, exist_ok=True)

    def _is_authorized(self, update: Update) -> bool:
        """Check if chat is authorized"""
        # If no users/chats configured, deny all for security
        if not self.allowed_users:
            return False
            
        if not update.effective_chat:
            return False
            
        return update.effective_chat.id in self.allowed_users

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        if not self._is_authorized(update):
            return

        help_text = """**Welcome!**

I'm your personal AI companion with memory. I remember everything you share with me.

**Commands:**
`/journal start` - Enter journal mode (silent logging)
`/journal done` - Compile journal into article
`/journal cancel` - Exit journal without saving
`/journal status` - Show journal session info

`/bsky <text>` - Post to Bluesky
`/voice` - Toggle voice responses
`/rate <1-5>` - Rate my last response
`/status` - Show system status
`/help` - Show this help

**Modes:**
- **Normal mode**: I respond to everything you say
- **Journal mode**: I silently log your entries for later compilation

Just send me a message to chat!"""

        await update.message.reply_text(help_text, parse_mode="Markdown")

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        await self.cmd_start(update, context)

    async def cmd_journal(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /journal commands"""
        if not self._is_authorized(update):
            return

        user_id = update.effective_user.id
        args = context.args

        if not args:
            # Show current status
            info = await self.journal.get_session_info(user_id)
            if info["active"]:
                await update.message.reply_text(
                    f"Journal mode active\n"
                    f"Entries: {info['entry_count']}\n"
                    f"Duration: {info['duration_minutes']} minutes"
                )
            else:
                await update.message.reply_text(
                    "Not in journal mode. Use `/journal start` to begin.",
                    parse_mode="Markdown"
                )
            return

        action = args[0].lower()

        if action == "start":
            if self.journal.is_journal_mode(user_id):
                await update.message.reply_text("Already in journal mode!")
                return

            self.journal.start_journal(user_id)
            await update.message.reply_text(
                "Journal mode started. Send text and images - I'll log them silently.\n"
                "Use `/journal done` to compile into an article."
            )

        elif action == "done":
            if not self.journal.is_journal_mode(user_id):
                await update.message.reply_text("Not in journal mode!")
                return

            # Check if there are entries
            count = await self.journal.get_entry_count(user_id)
            if count == 0:
                await update.message.reply_text("No entries to compile. Add some first!")
                return

            await update.message.reply_text("Compiling journal...")

            try:
                journal = await self.compiler.compile()
                self.journal.end_journal(user_id)

                if journal:
                    # Send the compiled article
                    response = f"**{journal.title}**\n\n"
                    response += f"Tags: {', '.join(journal.tags)}\n"
                    response += f"Entries: {len(journal.source_entry_ids)}\n\n"
                    response += journal.body[:1000]
                    if len(journal.body) > 1000:
                        response += "...\n\n(Full article saved)"

                    await update.message.reply_text(response, parse_mode="Markdown")

                    # Send markdown file
                    if journal.markdown_path:
                        with open(journal.markdown_path, "rb") as f:
                            await update.message.reply_document(
                                document=f,
                                filename=Path(journal.markdown_path).name
                            )
                else:
                    await update.message.reply_text("Failed to compile journal.")

            except Exception as e:
                logger.error(f"Journal compile failed: {e}")
                await update.message.reply_text(f"Compile failed: {e}")

        elif action == "cancel":
            if not self.journal.is_journal_mode(user_id):
                await update.message.reply_text("Not in journal mode!")
                return

            count = await self.journal.cancel_journal(user_id)
            await update.message.reply_text(f"Journal cancelled. {count} entries discarded.")

        elif action == "status":
            info = await self.journal.get_session_info(user_id)
            if info["active"]:
                await update.message.reply_text(
                    f"**Journal Session**\n"
                    f"Entries: {info['entry_count']}\n"
                    f"Duration: {info['duration_minutes']} min\n"
                    f"Started: {info['started_at'].strftime('%H:%M') if info['started_at'] else 'N/A'}",
                    parse_mode="Markdown"
                )
            else:
                await update.message.reply_text("Not in journal mode.")

        elif action == "preview":
            preview = await self.compiler.get_preview()
            if preview:
                await update.message.reply_text(preview, parse_mode="Markdown")
            else:
                await update.message.reply_text("No entries to preview.")

        else:
            await update.message.reply_text(
                "Unknown action. Use: start, done, cancel, status, preview"
            )

    async def cmd_bsky(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /bsky command - post to Bluesky"""
        if not self._is_authorized(update):
            return

        if self.journal.is_journal_mode(update.effective_user.id):
            await update.message.reply_text("Exit journal mode first to post to Bluesky.")
            return

        if not context.args:
            await update.message.reply_text("Usage: `/bsky <text>`", parse_mode="Markdown")
            return

        raw_text = " ".join(context.args)

        await update.message.reply_text("Posting to Bluesky...")

        try:
            # Polish the text
            polished = await self.bsky.polish_for_post(raw_text, self.llm)

            # Post
            uri = await self.bsky.post(polished)

            if uri:
                await update.message.reply_text(
                    f"Posted to Bluesky!\n\n{polished}\n\nURI: {uri}"
                )
            else:
                await update.message.reply_text("Post failed.")

        except Exception as e:
            logger.error(f"Bluesky post failed: {e}")
            await update.message.reply_text(f"Failed: {e}")

    async def cmd_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /voice command - toggle voice responses"""
        if not self._is_authorized(update):
            return

        self.tts_enabled = not self.tts_enabled
        status = "enabled" if self.tts_enabled else "disabled"
        await update.message.reply_text(f"Voice responses {status}.")

    async def cmd_rate(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /rate command - rate last response"""
        if not self._is_authorized(update):
            return

        user_id = update.effective_user.id

        if not context.args:
            await update.message.reply_text("Usage: `/rate <1-5>`", parse_mode="Markdown")
            return

        try:
            rating = int(context.args[0])
            if rating < 1 or rating > 5:
                raise ValueError("Rating must be 1-5")
        except ValueError:
            await update.message.reply_text("Rating must be a number 1-5.")
            return

        # Get last response memory ID
        memory_id = self._last_response.get(user_id)
        if not memory_id:
            await update.message.reply_text("No recent response to rate.")
            return

        # Add feedback
        await self.waifu.add_feedback(memory_id, rating)
        await update.message.reply_text(f"Thanks for rating {rating}/5!")

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        if not self._is_authorized(update):
            return

        user_id = update.effective_user.id
        session = self.journal.get_session(user_id)

        status = f"**System Status**\n\n"
        status += f"Mode: {session.mode}\n"
        status += f"Voice: {'enabled' if self.tts_enabled else 'disabled'}\n"

        if session.is_journal_mode():
            info = await self.journal.get_session_info(user_id)
            status += f"\n**Journal Session**\n"
            status += f"Entries: {info['entry_count']}\n"
            status += f"Duration: {info['duration_minutes']} min\n"

        await update.message.reply_text(status, parse_mode="Markdown")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages - mode-dependent behavior"""
        if not self._is_authorized(update):
            return

        user_id = update.effective_user.id
        text = update.message.text

        if self.journal.is_journal_mode(user_id):
            # Journal mode: silent logging
            await self.journal.add_text_entry(user_id, text)
            await update.message.reply_text("✓")
        else:
            # Normal mode: waifu responds
            await self._waifu_respond(update, text)

    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle photo messages"""
        if not self._is_authorized(update):
            return

        user_id = update.effective_user.id
        caption = update.message.caption or ""

        # Download image
        photo = update.message.photo[-1]  # Highest resolution
        file = await context.bot.get_file(photo.file_id)

        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = self.image_dir / f"{timestamp}_{photo.file_id[:8]}.jpg"

        await file.download_to_drive(str(image_path))

        # Compress if needed
        image_path = await self._compress_image(image_path)

        if self.journal.is_journal_mode(user_id):
            # Journal mode: describe and log silently
            await update.message.reply_text("Processing...")

            description = await self.waifu.describe_image(image_path, caption)
            await self.journal.add_image_entry(
                user_id, image_path, description, caption
            )

            await update.message.reply_text("✓")
        else:
            # Normal mode: describe and respond
            await update.message.reply_text("Looking at your image...")

            # Check if waifu has OCR capabilities
            if hasattr(self.waifu, 'respond_to_image'):
                response, memory_id = await self.waifu.respond_to_image(
                    image_path, user_id, caption
                )
                voice_path = None  # No TTS in this implementation
            else:
                # Fallback for standard waifu
                description = await self.waifu.describe_image(image_path, caption)
                await update.message.reply_text("Processing...")

                response, memory_id = await self.waifu.respond(
                    f"I just shared an image. {description}",
                    user_id,
                    stream=False
                )
                voice_path = None

            self._last_response[user_id] = memory_id

            await update.message.reply_text(response)

            if voice_path:
                with open(voice_path, "rb") as voice_file:
                    await update.message.reply_voice(voice=voice_file)
                voice_path.unlink(missing_ok=True)

    async def _waifu_respond(self, update: Update, message: str):
        """Generate waifu response"""
        user_id = update.effective_user.id

        # Show typing indicator
        await update.message.chat.send_action("typing")

        try:
            result = await self.waifu.respond(
                message, user_id,
                stream=False,
                with_voice=self.tts_enabled
            )

            if self.tts_enabled:
                response, voice_path, memory_id = result
            else:
                response, memory_id = result
                voice_path = None

            self._last_response[user_id] = memory_id

            # Send text response
            await update.message.reply_text(response)

            # Send voice if available
            if voice_path:
                with open(voice_path, "rb") as voice_file:
                    await update.message.reply_voice(voice=voice_file)
                voice_path.unlink(missing_ok=True)

        except Exception as e:
            logger.error(f"Waifu response failed: {e}")
            await update.message.reply_text("Sorry, I had trouble responding. Try again?")

    async def _compress_image(self, image_path: Path) -> Path:
        """Compress image if too large"""
        max_dim = config.MAX_IMAGE_DIMENSION
        quality = config.JPEG_QUALITY

        try:
            img = Image.open(image_path)

            # Resize if too large
            if max(img.size) > max_dim:
                ratio = max_dim / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            # Convert to RGB if needed
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')

            # Save as JPEG
            compressed_path = image_path.with_suffix('.jpg')
            img.save(compressed_path, 'JPEG', quality=quality, optimize=True)

            # Remove original if different
            if compressed_path != image_path:
                image_path.unlink(missing_ok=True)

            return compressed_path

        except Exception as e:
            logger.warning(f"Image compression failed: {e}")
            return image_path

    async def callback_rate(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle rating button callbacks"""
        query = update.callback_query
        await query.answer()

        data = query.data
        if data.startswith("rate_"):
            parts = data.split("_")
            if len(parts) == 3:
                rating = int(parts[1])
                memory_id = parts[2]

                await self.waifu.add_feedback(memory_id, rating)
                await query.edit_message_text(
                    f"{query.message.text}\n\n_Rated {rating}/5_",
                    parse_mode="Markdown"
                )

    def get_rating_keyboard(self, memory_id: str) -> InlineKeyboardMarkup:
        """Generate rating keyboard for a response"""
        buttons = [
            InlineKeyboardButton(f"{i}⭐", callback_data=f"rate_{i}_{memory_id}")
            for i in range(1, 6)
        ]
        return InlineKeyboardMarkup([buttons])

import asyncio
from pathlib import Path
from typing import AsyncIterator, Optional
import logging

from .llama_client import LlamaClient
from .tts_client import TTSClient
from memory.context import ContextBuilder
from memory.master_repo import MasterRepository
from memory.models import WaifuContext

logger = logging.getLogger(__name__)


class Waifu:
    """AI companion with personality, memory, and voice"""

    def __init__(
        self,
        llama_client: LlamaClient,
        context_builder: ContextBuilder,
        master_repo: MasterRepository,
        personality: str,
        tts_client: Optional[TTSClient] = None,
        tts_enabled: bool = True
    ):
        """
        Initialize waifu

        Args:
            llama_client: LLM client for generation
            context_builder: Context builder for memory
            master_repo: Master repository for storing responses
            personality: System prompt defining personality
            tts_client: TTS client for voice (optional)
            tts_enabled: Enable voice responses
        """
        self.llm = llama_client
        self.context = context_builder
        self.repo = master_repo
        self.personality = personality
        self.tts = tts_client
        self.tts_enabled = tts_enabled and tts_client is not None

    async def respond(
        self,
        message: str,
        user_id: int,
        stream: bool = False,
        with_voice: bool = False
    ) -> str | tuple[str, Optional[Path]]:
        """
        Generate response to user message

        Args:
            message: User's message
            user_id: Telegram user ID
            stream: Enable streaming (returns AsyncIterator)
            with_voice: Also generate voice response

        Returns:
            Response text, or tuple of (text, voice_path) if with_voice
        """
        logger.info(f"Generating response for user {user_id}")

        # Store user message as memory
        await self.repo.add_memory(
            type="chat",
            content=message,
            metadata={"role": "user", "user_id": user_id}
        )

        # Build context from memory
        ctx = await self.context.build_context(message, user_id)

        # Format context for prompt
        context_str = self.context.format_context_for_prompt(ctx)

        # Build system prompt with context
        system_prompt = self._build_system_prompt(context_str)

        # Generate response
        if stream:
            return self._stream_response(message, system_prompt, user_id)
        else:
            response = await self.llm.generate(
                prompt=message,
                system=system_prompt,
                stream=False
            )

            # Store assistant response as memory
            memory = await self.repo.add_memory(
                type="chat",
                content=response,
                metadata={"role": "assistant", "user_id": user_id}
            )

            # Generate voice if requested
            voice_path = None
            if with_voice and self.tts_enabled:
                try:
                    voice_path = await self.tts.text_to_voice_message(response)
                except Exception as e:
                    logger.error(f"TTS failed: {e}")

            if with_voice:
                return response, voice_path, memory.id
            return response, memory.id

    async def _stream_response(
        self,
        message: str,
        system_prompt: str,
        user_id: int
    ) -> AsyncIterator[str]:
        """Generate streaming response"""
        full_response = []

        async for token in await self.llm.generate(
            prompt=message,
            system=system_prompt,
            stream=True
        ):
            full_response.append(token)
            yield token

        # Store complete response
        response_text = "".join(full_response)
        await self.repo.add_memory(
            type="chat",
            content=response_text,
            metadata={"role": "assistant", "user_id": user_id}
        )

    def _build_system_prompt(self, context: str) -> str:
        """Build full system prompt with personality and context"""
        return f"""{self.personality}

=== YOUR MEMORIES ===
{context}

Remember to:
- Reference past conversations and events naturally
- Show you remember what they've shared
- Be genuinely caring, not performatively
- Keep responses warm but concise
"""

    async def describe_image(
        self,
        image_path: Path,
        user_context: Optional[str] = None
    ) -> str:
        """
        Describe an image with optional user context

        Args:
            image_path: Path to image file
            user_context: User's context/caption for the image

        Returns:
            Image description
        """
        prompt = "Describe this image in detail, focusing on what's happening and any notable elements."
        if user_context:
            prompt = f"The user shared this image with context: '{user_context}'. Describe what you see."

        description = await self.llm.describe_image(
            image_path=image_path,
            prompt=prompt
        )

        return description

    async def respond_to_image(
        self,
        image_path: Path,
        user_id: int,
        caption: Optional[str] = None,
        with_voice: bool = False
    ) -> tuple[str, Optional[Path], str]:
        """
        Respond to an image shared by user

        Args:
            image_path: Path to image
            user_id: User ID
            caption: User's caption (optional)
            with_voice: Generate voice response

        Returns:
            Tuple of (response, voice_path, memory_id)
        """
        # Describe the image
        description = await self.describe_image(image_path, caption)

        # Store image as memory
        image_memory = await self.repo.add_memory(
            type="image",
            content=description,
            raw_content=caption,
            media_path=str(image_path),
            metadata={"user_id": user_id}
        )

        # Generate a response about the image
        ctx = await self.context.build_context(
            f"User shared an image: {description}",
            user_id
        )
        context_str = self.context.format_context_for_prompt(ctx)
        system_prompt = self._build_system_prompt(context_str)

        image_prompt = f"I just shared an image. {description}"
        if caption:
            image_prompt = f"I just shared an image with caption '{caption}'. {description}"

        response = await self.llm.generate(
            prompt=image_prompt,
            system=system_prompt,
            stream=False
        )

        # Store response
        response_memory = await self.repo.add_memory(
            type="chat",
            content=response,
            metadata={"role": "assistant", "user_id": user_id, "about_image": image_memory.id}
        )

        # Generate voice if requested
        voice_path = None
        if with_voice and self.tts_enabled:
            try:
                voice_path = await self.tts.text_to_voice_message(response)
            except Exception as e:
                logger.error(f"TTS failed: {e}")

        return response, voice_path, response_memory.id

    async def add_feedback(
        self,
        memory_id: str,
        rating: int,
        context: Optional[str] = None
    ) -> str:
        """
        Add user feedback for a response (for RL fine-tuning)

        Args:
            memory_id: ID of the response memory
            rating: Rating 1-5
            context: Additional context

        Returns:
            Feedback ID
        """
        return await self.repo.add_feedback(
            memory_id=memory_id,
            rating=rating,
            context=context
        )


# Global singleton
_waifu_instance: Optional[Waifu] = None


def get_waifu(
    llama_client: LlamaClient,
    context_builder: ContextBuilder,
    master_repo: MasterRepository,
    personality: str,
    tts_client: Optional[TTSClient] = None,
    tts_enabled: bool = True
) -> Waifu:
    """Get or create global waifu instance"""
    global _waifu_instance

    if _waifu_instance is None:
        _waifu_instance = Waifu(
            llama_client=llama_client,
            context_builder=context_builder,
            master_repo=master_repo,
            personality=personality,
            tts_client=tts_client,
            tts_enabled=tts_enabled
        )

    return _waifu_instance

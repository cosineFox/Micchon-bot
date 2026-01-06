import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging

from atproto import Client, models

import config
from memory.master_repo import MasterRepository

logger = logging.getLogger(__name__)


class BlueskyClient:
    """Bluesky client with memory integration"""

    def __init__(self, master_repo: Optional[MasterRepository] = None):
        """
        Initialize Bluesky client

        Args:
            master_repo: Optional master repository for storing posts as memories
        """
        self.handle = config.BLUESKY_HANDLE
        self.password = config.BLUESKY_APP_PASSWORD
        self.repo = master_repo
        self._client: Optional[Client] = None
        self._last_post_time: Optional[datetime] = None
        self._cooldown = config.BLUESKY_POST_COOLDOWN

    def _get_client(self) -> Client:
        """Get or create authenticated Bluesky client"""
        if self._client is None:
            self._client = Client()
            self._client.login(self.handle, self.password)
            logger.info(f"Logged into Bluesky as {self.handle}")
        return self._client

    async def _ensure_cooldown(self):
        """Ensure rate limit cooldown between posts"""
        if self._last_post_time:
            elapsed = (datetime.now() - self._last_post_time).total_seconds()
            if elapsed < self._cooldown:
                wait_time = self._cooldown - elapsed
                logger.debug(f"Waiting {wait_time:.1f}s for cooldown")
                await asyncio.sleep(wait_time)

    async def post(
        self,
        text: str,
        image_path: Optional[str] = None,
        alt_text: Optional[str] = None,
        store_memory: bool = True
    ) -> Optional[str]:
        """
        Post to Bluesky

        Args:
            text: Post content
            image_path: Optional image to attach
            alt_text: Alt text for image
            store_memory: Store post as memory (default True)

        Returns:
            Post URI if successful, None if failed
        """
        await self._ensure_cooldown()

        try:
            client = self._get_client()

            embed = None
            if image_path:
                image_data = Path(image_path).read_bytes()
                upload = client.upload_blob(image_data)
                embed = models.AppBskyEmbedImages.Main(
                    images=[
                        models.AppBskyEmbedImages.Image(
                            alt=alt_text or "",
                            image=upload.blob
                        )
                    ]
                )

            # Post to Bluesky
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.send_post(text=text, embed=embed)
            )

            self._last_post_time = datetime.now()
            uri = response.uri

            logger.info(f"Posted to Bluesky: {uri}")

            # Store as memory if repository is available
            if store_memory and self.repo:
                await self.repo.add_memory(
                    type="bsky_post",
                    content=text,
                    media_path=image_path,
                    metadata={
                        "bsky_uri": uri,
                        "alt_text": alt_text,
                        "handle": self.handle
                    }
                )
                logger.debug(f"Stored Bluesky post as memory")

            return uri

        except Exception as e:
            logger.error(f"Bluesky post failed: {e}")
            raise Exception(f"Bluesky post failed: {e}")

    async def polish_for_post(
        self,
        raw_text: str,
        llm_client,
        project_name: Optional[str] = None
    ) -> str:
        """
        Use AI to polish text for Bluesky posting

        Args:
            raw_text: Raw user input
            llm_client: LLM client for text generation
            project_name: Optional project name for context

        Returns:
            Polished post text
        """
        prompt = f"""Polish this text for a Bluesky post. Keep it under 300 characters, engaging, and add relevant hashtags.

Raw text: {raw_text}

Rules:
- Be concise but informative
- Add 1-2 relevant hashtags like #buildinpublic #maker #coding
- Maintain the original meaning
- Make it engaging for social media

Return ONLY the polished post text, nothing else."""

        polished = await llm_client.generate(
            prompt=prompt,
            max_tokens=100,
            temperature=0.7
        )

        # Ensure it's not too long
        if len(polished) > 300:
            polished = polished[:297] + "..."

        return polished.strip()

    async def is_authenticated(self) -> bool:
        """Check if client can authenticate"""
        try:
            self._get_client()
            return True
        except Exception as e:
            logger.warning(f"Bluesky auth failed: {e}")
            return False

    async def get_profile(self) -> Optional[dict]:
        """Get current user profile"""
        try:
            client = self._get_client()
            loop = asyncio.get_event_loop()
            profile = await loop.run_in_executor(
                None,
                lambda: client.get_profile(self.handle)
            )
            return {
                "handle": profile.handle,
                "display_name": profile.display_name,
                "followers_count": profile.followers_count,
                "follows_count": profile.follows_count,
                "posts_count": profile.posts_count
            }
        except Exception as e:
            logger.error(f"Failed to get profile: {e}")
            return None


# Global singleton
_bsky_client: Optional[BlueskyClient] = None


def get_bluesky_client(
    master_repo: Optional[MasterRepository] = None
) -> BlueskyClient:
    """Get or create global Bluesky client"""
    global _bsky_client

    if _bsky_client is None:
        _bsky_client = BlueskyClient(master_repo=master_repo)

    return _bsky_client

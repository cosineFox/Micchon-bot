import asyncio
from pathlib import Path
from typing import AsyncIterator, Optional
import logging

from .model_manager import ModelManager

logger = logging.getLogger(__name__)


class LlamaClient:
    """Async wrapper around llama-cpp-python with streaming support"""

    def __init__(self, model_manager: ModelManager, gen_params: dict):
        """
        Initialize Llama client

        Args:
            model_manager: ModelManager instance
            gen_params: Generation parameters (temperature, top_p, etc.)
        """
        self.model_manager = model_manager
        self.gen_params = gen_params

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False
    ) -> str | AsyncIterator[str]:
        """
        Generate text completion

        Args:
            prompt: User prompt
            system: System prompt (personality, instructions)
            max_tokens: Override default max tokens
            temperature: Override default temperature
            stream: Enable streaming (returns AsyncIterator)

        Returns:
            Generated text (or AsyncIterator if streaming)
        """
        model = await self.model_manager.get_model()

        # Build generation parameters
        params = self.gen_params.copy()
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if temperature is not None:
            params["temperature"] = temperature
        params["stream"] = stream

        # Format prompt with system message if provided
        if system:
            full_prompt = f"{system}\n\nUser: {prompt}\nAssistant:"
        else:
            full_prompt = prompt

        logger.debug(f"Generating completion (stream={stream})")

        if stream:
            return self._stream_generate(model, full_prompt, params)
        else:
            return await self._complete_generate(model, full_prompt, params)

    async def _complete_generate(self, model, prompt: str, params: dict) -> str:
        """Generate complete response (non-streaming)"""
        loop = asyncio.get_event_loop()

        # Run blocking generation in thread pool
        response = await loop.run_in_executor(
            None,
            lambda: model(prompt, **params)
        )

        # Extract text from response
        text = response["choices"][0]["text"]
        logger.debug(f"Generated {len(text)} chars")
        return text.strip()

    async def _stream_generate(self, model, prompt: str, params: dict) -> AsyncIterator[str]:
        """Generate streaming response"""
        loop = asyncio.get_event_loop()

        # Create generator in thread pool
        def _blocking_stream():
            return model(prompt, **params)

        stream = await loop.run_in_executor(None, _blocking_stream)

        # Yield tokens from stream
        for chunk in stream:
            text = chunk["choices"][0]["text"]
            if text:
                yield text

    async def chat(
        self,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False
    ) -> str | AsyncIterator[str]:
        """
        Chat completion with message history

        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}
            max_tokens: Override default max tokens
            temperature: Override default temperature
            stream: Enable streaming

        Returns:
            Generated response (or AsyncIterator if streaming)
        """
        model = await self.model_manager.get_model()

        # Build generation parameters
        params = self.gen_params.copy()
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if temperature is not None:
            params["temperature"] = temperature
        params["stream"] = stream

        logger.debug(f"Chat completion with {len(messages)} messages (stream={stream})")

        if stream:
            return self._stream_chat(model, messages, params)
        else:
            return await self._complete_chat(model, messages, params)

    async def _complete_chat(self, model, messages: list[dict], params: dict) -> str:
        """Complete chat response (non-streaming)"""
        loop = asyncio.get_event_loop()

        # Run blocking chat in thread pool
        response = await loop.run_in_executor(
            None,
            lambda: model.create_chat_completion(messages=messages, **params)
        )

        # Extract text from response
        text = response["choices"][0]["message"]["content"]
        logger.debug(f"Generated {len(text)} chars")
        return text.strip()

    async def _stream_chat(self, model, messages: list[dict], params: dict) -> AsyncIterator[str]:
        """Streaming chat response"""
        loop = asyncio.get_event_loop()

        # Create generator in thread pool
        def _blocking_stream():
            return model.create_chat_completion(messages=messages, **params)

        stream = await loop.run_in_executor(None, _blocking_stream)

        # Yield tokens from stream
        for chunk in stream:
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                text = delta["content"]
                if text:
                    yield text

    async def describe_image(
        self,
        image_path: Path,
        prompt: str = "Describe this image."
    ) -> str:
        """
        Generate image description using LLaVA/multimodal model
        """
        # Check for dedicated vision model first
        vision_model = await self.model_manager.get_vision_model()
        
        if vision_model:
            model = vision_model
            logger.info("Using dedicated vision model")
        else:
            model = await self.model_manager.get_model()
            logger.info("Using main model for vision")

        try:
            # Helper to run in thread
            def _generate():
                # Convert image path to base64 data URI format for llama-cpp
                import base64
                
                with open(image_path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode("utf-8")
                
                image_url = f"data:image/jpeg;base64,{img_b64}"

                response = model.create_chat_completion(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": image_url}}
                            ]
                        }
                    ],
                    max_tokens=self.gen_params.get("max_tokens", 512),
                    temperature=0.7
                )
                return response["choices"][0]["message"]["content"]

            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _generate)

        except Exception as e:
            logger.error(f"Image description failed: {e}")
            return f"[Error processing image: {e}]"

    async def count_tokens(self, text: str) -> int:
        """
        Count tokens in text

        Args:
            text: Input text

        Returns:
            Token count
        """
        model = await self.model_manager.get_model()

        loop = asyncio.get_event_loop()
        tokens = await loop.run_in_executor(
            None,
            lambda: model.tokenize(text.encode("utf-8"))
        )

        return len(tokens)

    async def format_chat_history(
        self,
        messages: list[dict],
        max_tokens: int = 1500
    ) -> list[dict]:
        """
        Truncate chat history to fit within token budget

        Args:
            messages: List of message dicts
            max_tokens: Maximum tokens for history

        Returns:
            Truncated message list
        """
        # Keep system message if present
        system_msg = None
        if messages and messages[0]["role"] == "system":
            system_msg = messages[0]
            messages = messages[1:]

        # Count tokens for each message
        total_tokens = 0
        truncated = []

        # Start from most recent and work backwards
        for msg in reversed(messages):
            content = msg.get("content", "")
            tokens = await self.count_tokens(content)

            if total_tokens + tokens > max_tokens:
                break

            truncated.insert(0, msg)
            total_tokens += tokens

        # Add system message back
        if system_msg:
            truncated.insert(0, system_msg)

        logger.debug(f"Truncated history: {len(messages)} → {len(truncated)} messages ({total_tokens} tokens)")
        return truncated

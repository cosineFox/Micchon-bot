import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Optional
import logging
import torch

from sentence_transformers import SentenceTransformer
from cachetools import TTLCache
import numpy as np

logger = logging.getLogger(__name__)

# Thread pool for blocking embedding operations
_embed_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="embedder")


class Embedder:
    """
    Embedding generator optimized for RAG performance.

    Uses GPU when available, aggressive caching, and async execution.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        model_path: Optional[str] = None,  # Path to GGUF model
        dimension: int = 384,
        task: str = "search result",
        cache_size: int = 500,
        cache_ttl: int = 3600,
        use_gpu: bool = True
    ):
        """
        Initialize embedder

        Args:
            model_name: Name (for logic/logging)
            model_path: Path to GGUF file (if using llama.cpp)
            ...
        """
        self.model_name = model_name
        self.model_path = model_path
        self.dimension = dimension
        self.task = task
        self.cache_size = cache_size
        self.use_gpu = use_gpu
        self._model = None  # Union[SentenceTransformer, Llama]
        self._is_gguf = model_path is not None and str(model_path).endswith(".gguf")

        # TTL cache
        self._cache = TTLCache(maxsize=cache_size, ttl=cache_ttl)

        # Detect device (for sentence-transformers)
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        logger.info(f"Embedder initialized (GGUF: {self._is_gguf})")

    def _normalize_text(self, text: str) -> str:
        """Normalize text for cache key"""
        return text.strip().lower()[:500]  # Limit cache key size

    def _format_text(self, text: str) -> str:
        """Format text for embedding model"""
        return text.strip()

    def _get_model(self):
        """Lazy load model"""
        if self._model is None:
            if self._is_gguf:
                from llama_cpp import Llama
                logger.info(f"Loading GGUF embedding model: {self.model_path}")
                self._model = Llama(
                    model_path=str(self.model_path),
                    embedding=True,
                    n_gpu_layers=-1 if self.use_gpu else 0,
                    verbose=False
                )
            else:
                logger.info(f"Loading SentenceTransformer: {self.model_name}")
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    trust_remote_code=True
                )
            
            logger.info("Embedding model loaded")
        return self._model

    def _embed_sync(self, text: str) -> list[float]:
        """Synchronous embedding generation"""
        model = self._get_model()
        formatted = self._format_text(text)

        if self._is_gguf:
            # llama-cpp-python embedding
            # Note: create_embedding returns response dict or list depending on version
            # We assume standard usage: model.create_embedding(text) -> dict
            try:
                response = model.create_embedding(formatted)
                # Extract embedding vector
                if isinstance(response, dict) and "data" in response:
                    embedding = response["data"][0]["embedding"]
                else:
                    embedding = response # fallback
            except Exception as e:
                logger.error(f"GGUF embedding failed: {e}")
                return [0.0] * self.dimension
        else:
            # SentenceTransformer
            embedding = model.encode(formatted, convert_to_numpy=True, show_progress_bar=False)
            if len(embedding) > self.dimension:
                embedding = embedding[:self.dimension]
            embedding = embedding.tolist()

        return embedding

    async def embed(self, text: str) -> list[float]:
        """
        Generate embedding for text with caching and async execution

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector
        """
        # Check cache first
        cache_key = self._normalize_text(text)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Run blocking encode in thread pool
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(_embed_executor, self._embed_sync, text)

        # Cache result
        self._cache[cache_key] = embedding
        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts in batch (GPU-optimized)

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Check cache for already-embedded texts
        results = [None] * len(texts)
        texts_to_embed = []
        indices_to_embed = []

        for i, text in enumerate(texts):
            cache_key = self._normalize_text(text)
            if cache_key in self._cache:
                results[i] = self._cache[cache_key]
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)

        # Batch embed remaining texts
        if texts_to_embed:
            def _batch_encode():
                model = self._get_model()
                formatted = [self._format_text(t) for t in texts_to_embed]
                embeddings = model.encode(
                    formatted,
                    convert_to_numpy=True,
                    batch_size=32,
                    show_progress_bar=False
                )
                # Truncate dimensions if needed
                if embeddings.shape[1] > self.dimension:
                    embeddings = embeddings[:, :self.dimension]
                return embeddings.tolist()

            loop = asyncio.get_event_loop()
            new_embeddings = await loop.run_in_executor(_embed_executor, _batch_encode)

            # Store results and update cache
            for idx, emb, text in zip(indices_to_embed, new_embeddings, texts_to_embed):
                results[idx] = emb
                cache_key = self._normalize_text(text)
                self._cache[cache_key] = emb

        return results

    def clear_cache(self):
        """Clear the embedding cache"""
        self._cache.clear()
        logger.info("Embedding cache cleared")

    def unload_model(self):
        """Unload model from memory to free VRAM/RAM"""
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Embedding model unloaded")

    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        return {
            "cache_size": len(self._cache),
            "max_size": self._cache.maxsize,
            "model_loaded": self._model is not None,
            "device": self.device
        }


# Global singleton instance (lazy-loaded)
_embedder_instance: Optional[Embedder] = None


def get_embedder(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    model_path: Optional[str] = None,
    dimension: int = 384,
    task: str = "search result",
    use_gpu: bool = True
) -> Embedder:
    """
    Get or create the global embedder instance
    """
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = Embedder(
            model_name=model_name,
            model_path=model_path,
            dimension=dimension,
            task=task,
            use_gpu=use_gpu
        )
    return _embedder_instance

from functools import lru_cache
from typing import Optional
from sentence_transformers import SentenceTransformer
import numpy as np


class Embedder:
    """Embedding generator using EmbeddingGemma with caching"""

    def __init__(
        self,
        model_name: str = "google/embeddinggemma-300m",
        dimension: int = 256,
        task: str = "search result",
        cache_size: int = 100
    ):
        """
        Initialize embedder with EmbeddingGemma

        Args:
            model_name: HuggingFace model name
            dimension: Embedding dimension (256, 512, or 768 via MRL)
            task: Task prompt for embeddings ("search result", "classification", etc.)
            cache_size: LRU cache size for repeated queries
        """
        self.model_name = model_name
        self.dimension = dimension
        self.task = task
        self.cache_size = cache_size
        self._model: Optional[SentenceTransformer] = None

    def _get_model(self) -> SentenceTransformer:
        """Lazy load the model on first use"""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _format_text(self, text: str) -> str:
        """Format text with task prompt for EmbeddingGemma"""
        return f"task: {self.task} | query: {text}"

    @lru_cache(maxsize=100)
    def _embed_cached(self, text: str) -> tuple[float, ...]:
        """Cached embedding generation (tuple for hashability)"""
        model = self._get_model()
        formatted = self._format_text(text)

        # Generate embedding
        embedding = model.encode(formatted, convert_to_numpy=True)

        # Truncate to desired dimension if using MRL
        if len(embedding) > self.dimension:
            embedding = embedding[:self.dimension]

        return tuple(embedding.tolist())

    async def embed(self, text: str) -> list[float]:
        """
        Generate embedding for text (async wrapper)

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector
        """
        # Use cached version (converted back to list)
        return list(self._embed_cached(text))

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts in batch

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        model = self._get_model()
        formatted_texts = [self._format_text(t) for t in texts]

        # Batch encode
        embeddings = model.encode(formatted_texts, convert_to_numpy=True, batch_size=32)

        # Truncate to desired dimension if using MRL
        if embeddings.shape[1] > self.dimension:
            embeddings = embeddings[:, :self.dimension]

        return embeddings.tolist()

    def clear_cache(self):
        """Clear the embedding cache"""
        self._embed_cached.cache_clear()

    def unload_model(self):
        """Unload model from memory to free VRAM/RAM"""
        if self._model is not None:
            del self._model
            self._model = None
            # Clear cache when unloading
            self.clear_cache()


# Global singleton instance (lazy-loaded)
_embedder_instance: Optional[Embedder] = None


def get_embedder(
    model_name: str = "google/embeddinggemma-300m",
    dimension: int = 256,
    task: str = "search result"
) -> Embedder:
    """Get or create the global embedder instance"""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = Embedder(
            model_name=model_name,
            dimension=dimension,
            task=task
        )
    return _embedder_instance

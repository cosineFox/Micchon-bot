import asyncio
import logging
from typing import Optional
from uuid import UUID

from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct

logger = logging.getLogger(__name__)

# Collection name for memories
COLLECTION_NAME = "memories"


class QdrantMemoryStore:
    """Qdrant vector store for memory embeddings"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        embedding_dimension: int = 384,
        distance: Distance = Distance.COSINE
    ):
        """
        Initialize Qdrant client

        Args:
            host: Qdrant server host
            port: Qdrant server port
            embedding_dimension: Vector dimension
            distance: Distance metric (COSINE, EUCLID, DOT)
        """
        self.host = host
        self.port = port
        self.embedding_dimension = embedding_dimension
        self.distance = distance
        self._client: Optional[QdrantClient] = None
        self._initialized = False

    async def _get_client(self) -> QdrantClient:
        """Get or create Qdrant client"""
        if self._client is None:
            loop = asyncio.get_event_loop()
            self._client = await loop.run_in_executor(
                None,
                lambda: QdrantClient(host=self.host, port=self.port)
            )
        return self._client

    async def init(self):
        """Initialize Qdrant collection if it doesn't exist"""
        if self._initialized:
            return

        client = await self._get_client()

        try:
            loop = asyncio.get_event_loop()

            def _init_collection():
                # Check if collection exists
                collections = client.get_collections().collections
                exists = any(c.name == COLLECTION_NAME for c in collections)

                if not exists:
                    logger.info(f"Creating Qdrant collection: {COLLECTION_NAME}")
                    client.create_collection(
                        collection_name=COLLECTION_NAME,
                        vectors_config=VectorParams(
                            size=self.embedding_dimension,
                            distance=self.distance
                        )
                    )
                    logger.info(f"Collection created with {self.embedding_dimension}d vectors")
                else:
                    logger.info(f"Qdrant collection '{COLLECTION_NAME}' already exists")

            await loop.run_in_executor(None, _init_collection)
            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            raise

    async def add_embedding(
        self,
        memory_id: str,
        embedding: list[float],
        metadata: Optional[dict] = None
    ) -> bool:
        """
        Add embedding for a memory

        Args:
            memory_id: Memory UUID
            embedding: Vector embedding
            metadata: Optional payload metadata

        Returns:
            Success status
        """
        await self.init()
        client = await self._get_client()

        try:
            loop = asyncio.get_event_loop()

            def _upsert():
                # Convert string UUID to int for Qdrant point ID
                point_id = self._uuid_to_int(memory_id)

                payload = metadata or {}
                payload["memory_id"] = memory_id  # Store original UUID

                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=[
                        PointStruct(
                            id=point_id,
                            vector=embedding,
                            payload=payload
                        )
                    ]
                )

            await loop.run_in_executor(None, _upsert)
            logger.debug(f"Added embedding for memory {memory_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to add embedding: {e}")
            return False

    async def search_similar(
        self,
        query_embedding: list[float],
        limit: int = 10,
        score_threshold: float = 0.0
    ) -> list[tuple[str, float]]:
        """
        Search for similar memories

        Args:
            query_embedding: Query vector
            limit: Max results
            score_threshold: Minimum similarity score

        Returns:
            List of (memory_id, score) tuples
        """
        await self.init()
        client = await self._get_client()

        try:
            loop = asyncio.get_event_loop()

            def _search():
                results = client.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=query_embedding,
                    limit=limit,
                    score_threshold=score_threshold
                )
                return [
                    (r.payload.get("memory_id", str(r.id)), r.score)
                    for r in results
                ]

            return await loop.run_in_executor(None, _search)

        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return []

    async def delete_embedding(self, memory_id: str) -> bool:
        """Delete embedding for a memory"""
        await self.init()
        client = await self._get_client()

        try:
            loop = asyncio.get_event_loop()

            def _delete():
                point_id = self._uuid_to_int(memory_id)
                client.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=models.PointIdsList(
                        points=[point_id]
                    )
                )

            await loop.run_in_executor(None, _delete)
            logger.debug(f"Deleted embedding for memory {memory_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete embedding: {e}")
            return False

    async def delete_old_embeddings(self, memory_ids: list[str]) -> int:
        """
        Delete embeddings for multiple memories

        Args:
            memory_ids: List of memory UUIDs to delete

        Returns:
            Number of deleted embeddings
        """
        await self.init()
        client = await self._get_client()

        try:
            loop = asyncio.get_event_loop()

            def _batch_delete():
                point_ids = [self._uuid_to_int(mid) for mid in memory_ids]
                client.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=models.PointIdsList(
                        points=point_ids
                    )
                )
                return len(point_ids)

            count = await loop.run_in_executor(None, _batch_delete)
            logger.info(f"Deleted {count} old embeddings")
            return count

        except Exception as e:
            logger.error(f"Failed to batch delete embeddings: {e}")
            return 0

    async def get_collection_info(self) -> dict:
        """Get collection statistics"""
        await self.init()
        client = await self._get_client()

        try:
            loop = asyncio.get_event_loop()

            def _info():
                info = client.get_collection(COLLECTION_NAME)
                return {
                    "vectors_count": info.vectors_count,
                    "points_count": info.points_count,
                    "status": info.status.name,
                    "dimension": self.embedding_dimension
                }

            return await loop.run_in_executor(None, _info)

        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {"error": str(e)}

    def _uuid_to_int(self, uuid_str: str) -> int:
        """Convert UUID string to integer for Qdrant point ID"""
        try:
            return UUID(uuid_str).int % (2**63)  # Keep within signed 64-bit range
        except ValueError:
            # If not a valid UUID, hash the string
            return hash(uuid_str) % (2**63)

    async def close(self):
        """Close Qdrant client connection"""
        if self._client is not None:
            self._client.close()
            self._client = None
            logger.info("Qdrant client closed")


# Global singleton
_qdrant_store: Optional[QdrantMemoryStore] = None


def get_qdrant_store(
    host: str = "localhost",
    port: int = 6333,
    embedding_dimension: int = 384
) -> QdrantMemoryStore:
    """Get or create global Qdrant store"""
    global _qdrant_store

    if _qdrant_store is None:
        _qdrant_store = QdrantMemoryStore(
            host=host,
            port=port,
            embedding_dimension=embedding_dimension
        )

    return _qdrant_store

"""Vector stores behind one tiny ``add`` / ``search`` surface.

- :class:`VectorStore` — an in-memory store (default). Vectors are L2-normalised
  on the way in, so one matrix-vector dot product gives cosine similarity for
  every chunk at once. Fast, dependency-free, and rebuilt on each startup.
- :class:`QdrantVectorStore` — a real, persistent vector database you run in
  Docker (``VECTOR_STORE=qdrant``). Talks to Qdrant's REST API over httpx, so it
  needs no extra Python dependency.

:func:`get_vector_store` picks between them from the ``VECTOR_STORE`` env var.
Because the interface is small, swapping to FAISS / pgvector later stays local.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import ClassVar

import httpx
import numpy as np


@dataclass
class Chunk:
    id: int
    text: str
    source: str


@dataclass
class SearchHit:
    chunk: Chunk
    score: float


@dataclass
class VectorStore:
    name: ClassVar[str] = "memory"
    dim: int
    _chunks: list[Chunk] = field(default_factory=list)
    _matrix: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        if len(chunks) == 0:
            return
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vectors.shape[1]}")
        self._chunks.extend(chunks)
        vectors = vectors.astype(np.float32)
        self._matrix = vectors if self._matrix is None else np.vstack([self._matrix, vectors])

    def search(self, query_vector: np.ndarray, k: int = 4) -> list[SearchHit]:
        if self._matrix is None or len(self._chunks) == 0:
            return []
        scores = self._matrix @ query_vector.astype(np.float32)
        k = min(k, len(self._chunks))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [SearchHit(chunk=self._chunks[i], score=float(scores[i])) for i in top]

    def next_id(self) -> int:
        return len(self._chunks)


class QdrantVectorStore:
    """A persistent vector store backed by Qdrant (run locally via Docker).

    Uses Qdrant's REST API directly (no client library). By default the
    collection is recreated on startup so a fresh ingest is deterministic, matching
    the in-memory store's behaviour; set ``QDRANT_RECREATE=false`` to keep data
    across restarts.
    """

    name: ClassVar[str] = "qdrant"

    def __init__(
        self,
        dim: int,
        url: str | None = None,
        collection: str | None = None,
        recreate: bool | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.dim = dim
        self.url = (url or os.getenv("QDRANT_URL", "http://localhost:6333")).rstrip("/")
        self.collection = collection or os.getenv("QDRANT_COLLECTION", "rag_docs")
        if recreate is None:
            recreate = os.getenv("QDRANT_RECREATE", "true").lower() not in ("false", "0", "no")
        self.timeout = timeout
        self._n = 0
        self._client = httpx.Client(base_url=self.url, timeout=timeout)
        self._ensure_collection(recreate)

    def _ensure_collection(self, recreate: bool) -> None:
        if recreate:
            self._client.delete(f"/collections/{self.collection}")
        exists = self._client.get(f"/collections/{self.collection}").status_code == 200
        if not exists:
            resp = self._client.put(
                f"/collections/{self.collection}",
                json={"vectors": {"size": self.dim, "distance": "Cosine"}},
            )
            resp.raise_for_status()
        else:
            self._n = self._count()

    def _count(self) -> int:
        resp = self._client.post(f"/collections/{self.collection}/points/count", json={"exact": True})
        resp.raise_for_status()
        return int(resp.json()["result"]["count"])

    def __len__(self) -> int:
        return self._n

    def next_id(self) -> int:
        return self._n

    def add(self, chunks: list["Chunk"], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        if len(chunks) == 0:
            return
        points = [
            {
                "id": chunk.id,
                "vector": vectors[i].astype(np.float32).tolist(),
                "payload": {"text": chunk.text, "source": chunk.source},
            }
            for i, chunk in enumerate(chunks)
        ]
        resp = self._client.put(f"/collections/{self.collection}/points", params={"wait": "true"}, json={"points": points})
        resp.raise_for_status()
        self._n += len(points)

    def search(self, query_vector: np.ndarray, k: int = 4) -> list[SearchHit]:
        resp = self._client.post(
            f"/collections/{self.collection}/points/search",
            json={"vector": query_vector.astype(np.float32).tolist(), "limit": k, "with_payload": True},
        )
        resp.raise_for_status()
        hits: list[SearchHit] = []
        for item in resp.json()["result"]:
            payload = item.get("payload", {})
            chunk = Chunk(id=int(item["id"]), text=payload.get("text", ""), source=payload.get("source", ""))
            hits.append(SearchHit(chunk=chunk, score=float(item["score"])))
        return hits


def get_vector_store(dim: int):
    """Return the vector store selected by the ``VECTOR_STORE`` env var."""
    kind = os.getenv("VECTOR_STORE", "memory").strip().lower()
    if kind in ("", "memory", "inmemory"):
        return VectorStore(dim=dim)
    if kind == "qdrant":
        return QdrantVectorStore(dim=dim)
    raise ValueError(f"Unknown VECTOR_STORE={kind!r}. Use: memory, qdrant.")

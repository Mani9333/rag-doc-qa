"""Pluggable text embeddings.

The default :class:`HashingEmbedder` needs **no model download and no network**
— it uses the feature-hashing trick (signed hashing of word tokens into a fixed
vector) so cosine similarity approximates term overlap. That keeps the project
instant to run and the tests deterministic.

For real semantic search, set ``EMBEDDINGS=sentence-transformers`` (local, free,
downloads a small model once) or ``EMBEDDINGS=openai`` (hosted). All three return
L2-normalised vectors, so the vector store treats them identically.
"""

from __future__ import annotations

import hashlib
import os
import re
from abc import ABC, abstractmethod

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(ABC):
    name: str = "embedder"
    dim: int

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalised vectors."""
        raise NotImplementedError

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class HashingEmbedder(Embedder):
    name = "hashing"

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in _TOKEN.findall(text.lower()):
                digest = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
                index = digest % self.dim
                sign = 1.0 if (digest >> 8) & 1 == 0 else -1.0
                vecs[i, index] += sign
        return _normalize(vecs)


class SentenceTransformerEmbedder(Embedder):
    """Real semantic embeddings via sentence-transformers (optional dep)."""

    name = "sentence-transformers"

    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer  # lazy: optional dependency

        self.model_name = model_name or os.getenv("ST_MODEL", "all-MiniLM-L6-v2")
        self._model = SentenceTransformer(self.model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.asarray(self._model.encode(texts, normalize_embeddings=True), dtype=np.float32)
        return vecs


class OpenAIEmbedder(Embedder):
    """Hosted embeddings via OpenAI's /embeddings endpoint (optional)."""

    name = "openai"

    def __init__(self, model: str | None = None) -> None:
        import httpx  # lazy

        self._httpx = httpx
        self.model = model or os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set (needed for EMBEDDINGS=openai).")
        self.dim = 1536

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self._httpx.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": texts},
            timeout=60.0,
        )
        resp.raise_for_status()
        vecs = np.asarray([d["embedding"] for d in resp.json()["data"]], dtype=np.float32)
        return _normalize(vecs)


def get_embedder() -> Embedder:
    kind = os.getenv("EMBEDDINGS", "hashing").strip().lower()
    if kind in ("", "hashing"):
        return HashingEmbedder(dim=int(os.getenv("HASHING_DIM", "512")))
    if kind in ("sentence-transformers", "st"):
        return SentenceTransformerEmbedder()
    if kind == "openai":
        return OpenAIEmbedder()
    raise ValueError(f"Unknown EMBEDDINGS={kind!r}. Use: hashing, sentence-transformers, openai.")


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms

"""Integration test for the optional Qdrant vector store.

Auto-skips unless a Qdrant instance is reachable, so the default `pytest` run
stays hermetic. To run it:

    docker run -d --name rag-qdrant -p 6333:6333 qdrant/qdrant
    pytest tests/test_qdrant.py
"""

import os

import httpx
import numpy as np
import pytest

from rag.pipeline import RagPipeline
from rag.vector_store import Chunk, QdrantVectorStore

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")


def _qdrant_reachable() -> bool:
    try:
        return httpx.get(f"{QDRANT_URL}/readyz", timeout=1.0).status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _qdrant_reachable(),
    reason="Qdrant not reachable — run `docker run -p 6333:6333 qdrant/qdrant`",
)


def _unit(vec):
    vec = np.asarray(vec, dtype=np.float32)
    return vec / np.linalg.norm(vec)


def test_qdrant_add_and_search():
    store = QdrantVectorStore(dim=3, url=QDRANT_URL, collection="rag_test_unit", recreate=True)
    chunks = [Chunk(id=i, text=f"c{i}", source="s") for i in range(3)]
    vectors = np.vstack([_unit([1, 0, 0]), _unit([0, 1, 0]), _unit([0.9, 0.1, 0])])
    store.add(chunks, vectors)
    assert len(store) == 3
    hits = store.search(_unit([1, 0, 0]), k=2)
    assert hits[0].chunk.id == 0
    assert hits[0].score >= hits[1].score


def test_pipeline_end_to_end_on_qdrant(monkeypatch):
    monkeypatch.setenv("VECTOR_STORE", "qdrant")
    monkeypatch.setenv("QDRANT_COLLECTION", "rag_test_pipeline")
    rag = RagPipeline(chunk_size=400, overlap=80, top_k=3)
    assert rag.ingest_dir("data/sample_docs") > 0
    # Same discriminative query as the in-memory pipeline test: Qdrant should
    # rank identically since both use cosine over the same vectors.
    answer = rag.ask("How does cosine similarity rank vector search results?")
    assert answer.citations
    assert "vector-search" in answer.citations[0].source

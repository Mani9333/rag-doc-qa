import numpy as np

from rag.vector_store import Chunk, VectorStore


def _unit(vec):
    vec = np.asarray(vec, dtype=np.float32)
    return vec / np.linalg.norm(vec)


def test_search_returns_top_k_in_order():
    store = VectorStore(dim=3)
    chunks = [Chunk(id=i, text=f"c{i}", source="s") for i in range(3)]
    vectors = np.vstack([_unit([1, 0, 0]), _unit([0, 1, 0]), _unit([0.9, 0.1, 0])])
    store.add(chunks, vectors)

    hits = store.search(_unit([1, 0, 0]), k=2)
    assert len(hits) == 2
    assert hits[0].chunk.id == 0
    assert hits[1].chunk.id == 2  # closest to [1,0,0] after the exact match
    assert hits[0].score >= hits[1].score


def test_empty_store_returns_nothing():
    assert VectorStore(dim=4).search(_unit([1, 0, 0, 0]), k=3) == []


def test_dim_mismatch_raises():
    store = VectorStore(dim=3)
    import pytest

    with pytest.raises(ValueError):
        store.add([Chunk(id=0, text="x", source="s")], np.zeros((1, 4), dtype=np.float32))

import numpy as np

from rag.embeddings import HashingEmbedder


def test_shape_and_normalisation():
    emb = HashingEmbedder(dim=128)
    vecs = emb.embed(["hello world", "another document about cats"])
    assert vecs.shape == (2, 128)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_similar_text_is_more_similar_than_unrelated():
    emb = HashingEmbedder(dim=512)
    q = emb.embed_one("cosine similarity ranks vector search results")
    related = emb.embed_one("vector search ranks results by cosine similarity")
    unrelated = emb.embed_one("bananas are a good source of potassium")
    assert float(q @ related) > float(q @ unrelated)


def test_deterministic():
    emb = HashingEmbedder(dim=64)
    a = emb.embed_one("repeatable embeddings")
    b = emb.embed_one("repeatable embeddings")
    assert np.array_equal(a, b)

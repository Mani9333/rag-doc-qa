from rag.pipeline import RagPipeline


def _pipeline():
    # Default embedder (hashing) + default model (mock) → fully offline.
    return RagPipeline(chunk_size=400, overlap=80, top_k=3)


def test_ingest_and_retrieve_relevant_chunk():
    rag = _pipeline()
    n = rag.ingest_dir("data/sample_docs")
    assert n > 0
    assert rag.num_chunks == n

    hits = rag.retrieve("How does cosine similarity rank vector search results?")
    assert hits
    # The vector-search document should be the top hit for this question.
    assert "vector-search" in hits[0].chunk.source


def test_ask_returns_grounded_answer_with_citations():
    rag = _pipeline()
    rag.ingest_dir("data/sample_docs")
    answer = rag.ask("What is retrieval-augmented generation?")
    assert answer.answer and answer.answer != "No documents have been ingested yet."
    assert "[1]" in answer.answer  # mock cites the top source
    assert len(answer.citations) == 3
    assert all(0.0 <= c.score <= 1.0001 for c in answer.citations)


def test_ask_with_no_documents():
    rag = _pipeline()
    answer = rag.ask("anything?")
    assert answer.citations == []

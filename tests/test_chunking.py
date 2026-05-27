from rag.chunking import chunk_text


def test_empty_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_short_text_is_one_chunk():
    assert chunk_text("Just a short sentence.", 800, 150) == ["Just a short sentence."]


def test_long_text_splits_and_stays_within_bound():
    paras = "\n\n".join(f"Paragraph {i}. " + "word " * 40 for i in range(20))
    chunks = chunk_text(paras, chunk_size=300, overlap=60)
    assert len(chunks) > 3
    assert all(0 < len(c) <= 300 for c in chunks)


def test_oversized_paragraph_is_hard_split():
    chunks = chunk_text("x" * 2000, chunk_size=500, overlap=100)
    assert len(chunks) >= 4
    assert all(len(c) <= 500 for c in chunks)


def test_overlap_must_be_smaller_than_chunk_size():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("hello world", chunk_size=100, overlap=100)

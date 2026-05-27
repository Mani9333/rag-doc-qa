"""The RAG pipeline: ingest → chunk → embed → store, and ask → retrieve → answer.

This class wires the pluggable pieces (embedder, vector store, chat model)
together and is what the API and the tests drive. Keeping ingestion and querying
in one small object makes the whole data flow readable top-to-bottom.
"""

from __future__ import annotations

from dataclasses import dataclass

from .chunking import chunk_text
from .embeddings import Embedder, get_embedder
from .llm import ChatModel, get_chat_model
from .loaders import load_documents
from .prompts import build_messages
from .vector_store import Chunk, SearchHit, get_vector_store


@dataclass
class Citation:
    source: str
    score: float
    snippet: str


@dataclass
class Answer:
    answer: str
    citations: list[Citation]


class RagPipeline:
    def __init__(
        self,
        embedder: Embedder | None = None,
        model: ChatModel | None = None,
        *,
        chunk_size: int = 800,
        overlap: int = 150,
        top_k: int = 4,
    ) -> None:
        self.embedder = embedder or get_embedder()
        self.model = model or get_chat_model()
        self.store = get_vector_store(self.embedder.dim)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k

    # -- ingestion ---------------------------------------------------------
    def ingest_text(self, text: str, source: str) -> int:
        pieces = chunk_text(text, self.chunk_size, self.overlap)
        if not pieces:
            return 0
        chunks = [Chunk(id=self.store.next_id() + i, text=p, source=source) for i, p in enumerate(pieces)]
        vectors = self.embedder.embed([c.text for c in chunks])
        self.store.add(chunks, vectors)
        return len(chunks)

    def ingest_dir(self, directory: str) -> int:
        total = 0
        for source, text in load_documents(directory):
            total += self.ingest_text(text, source)
        return total

    # -- querying ----------------------------------------------------------
    def retrieve(self, question: str, k: int | None = None) -> list[SearchHit]:
        query_vec = self.embedder.embed_one(question)
        return self.store.search(query_vec, k or self.top_k)

    def ask(self, question: str, k: int | None = None) -> Answer:
        hits = self.retrieve(question, k)
        if not hits:
            return Answer(answer="No documents have been ingested yet.", citations=[])
        messages = build_messages(question, hits)
        answer_text = self.model.complete(messages)
        citations = [
            Citation(source=h.chunk.source, score=round(h.score, 4), snippet=_snippet(h.chunk.text))
            for h in hits
        ]
        return Answer(answer=answer_text, citations=citations)

    @property
    def num_chunks(self) -> int:
        return len(self.store)


def _snippet(text: str, limit: int = 200) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"

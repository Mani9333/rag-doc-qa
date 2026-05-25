"""Build the grounded prompt sent to the model.

The context chunks are numbered so the model can cite them as ``[1]``, ``[2]``,
… and the system instruction constrains it to answer *only* from that context —
the core guardrail against hallucination in RAG.
"""

from __future__ import annotations

from .llm import Message
from .vector_store import SearchHit

SYSTEM = (
    "You are a precise assistant. Answer the question using ONLY the numbered "
    "context below. Cite the sources you use with bracketed numbers like [1]. "
    "If the answer is not contained in the context, say you don't know."
)


def build_messages(question: str, hits: list[SearchHit]) -> list[Message]:
    context = "\n\n".join(f"[{i}] (source: {h.chunk.source})\n{h.chunk.text}" for i, h in enumerate(hits, 1))
    user = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer with citations:"
    return [Message("system", SYSTEM), Message("user", user)]

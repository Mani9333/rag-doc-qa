"""Split documents into overlapping chunks for embedding + retrieval.

Strategy: pack whole paragraphs up to ``chunk_size`` characters, carrying a small
``overlap`` tail into the next chunk so context isn't lost at boundaries. Any
single paragraph larger than ``chunk_size`` is hard-split with a sliding window.
Character-based keeps it dependency-free and predictable; a token-based splitter
(tiktoken) would be the swap for tight model context budgets.
"""

from __future__ import annotations

import re

DEFAULT_CHUNK_SIZE = 800
DEFAULT_OVERLAP = 150

_PARA = re.compile(r"\n\s*\n")


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    paragraphs: list[str] = []
    for para in _PARA.split(text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= chunk_size:
            paragraphs.append(para)
        else:
            paragraphs.extend(_window(para, chunk_size, overlap))

    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if not current:
            current = para
        elif len(current) + 2 + len(para) <= chunk_size:
            current = f"{current}\n\n{para}"
        else:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            # Carry an overlap tail into the next chunk, but only if it still
            # fits — this keeps every emitted chunk within chunk_size.
            if tail and len(tail) + 2 + len(para) <= chunk_size:
                current = f"{tail}\n\n{para}"
            else:
                current = para
    if current:
        chunks.append(current)
    return chunks


def _window(text: str, size: int, overlap: int) -> list[str]:
    step = size - overlap
    return [text[i : i + size] for i in range(0, len(text), step)]

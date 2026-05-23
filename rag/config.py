"""Environment-driven settings (all have sensible offline defaults)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from .chunking import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP


@dataclass(frozen=True)
class Settings:
    ingest_dir: str = os.getenv("INGEST_DIR", "data/sample_docs")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", str(DEFAULT_CHUNK_SIZE)))
    overlap: int = int(os.getenv("CHUNK_OVERLAP", str(DEFAULT_OVERLAP)))
    top_k: int = int(os.getenv("TOP_K", "4"))

    @staticmethod
    def load() -> "Settings":
        return Settings()

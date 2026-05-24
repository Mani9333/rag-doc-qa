"""Load plain-text documents from a directory tree."""

from __future__ import annotations

from pathlib import Path

TEXT_EXTENSIONS = {".md", ".txt", ".rst", ".py", ".markdown"}


def load_documents(directory: str | Path) -> list[tuple[str, str]]:
    """Return ``(source, text)`` pairs for every supported file under ``directory``."""
    base = Path(directory)
    if not base.exists():
        raise FileNotFoundError(f"no such directory: {directory}")
    docs: list[tuple[str, str]] = []
    for path in sorted(base.rglob("*")):
        if path.is_file() and path.suffix.lower() in TEXT_EXTENSIONS:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                docs.append((str(path.relative_to(base)), text))
    return docs

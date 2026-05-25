"""FastAPI app exposing the RAG pipeline.

Routes:
  GET  /            → minimal web UI (static/index.html)
  GET  /healthz     → liveness + which model/embedder/how many chunks
  POST /ask         → {question, top_k?} → grounded answer + citations
  POST /ingest      → {dir} or {text, source} → add documents at runtime
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import Settings
from .pipeline import RagPipeline

load_dotenv()

_STATIC = Path(__file__).resolve().parent.parent / "static"


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)


class IngestRequest(BaseModel):
    dir: str | None = None
    text: str | None = None
    source: str = "inline"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings.load()
    pipeline = RagPipeline(chunk_size=settings.chunk_size, overlap=settings.overlap, top_k=settings.top_k)
    try:
        pipeline.ingest_dir(settings.ingest_dir)
    except FileNotFoundError:
        pass  # nothing to preload; documents can be added via /ingest
    app.state.pipeline = pipeline
    app.state.settings = settings
    yield


app = FastAPI(title="rag-doc-qa", version="0.1.0", lifespan=lifespan)

if _STATIC.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


def _pipeline(app: FastAPI) -> RagPipeline:
    return app.state.pipeline


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


@app.get("/healthz")
async def healthz() -> dict:
    pipeline = _pipeline(app)
    return {
        "status": "ok",
        "provider": pipeline.model.name,
        "embedder": pipeline.embedder.name,
        "vector_store": pipeline.store.name,
        "chunks": pipeline.num_chunks,
    }


@app.post("/ask")
async def ask(req: AskRequest) -> dict:
    answer = _pipeline(app).ask(req.question, req.top_k)
    return {
        "answer": answer.answer,
        "citations": [c.__dict__ for c in answer.citations],
    }


@app.post("/ingest")
async def ingest(req: IngestRequest) -> dict:
    pipeline = _pipeline(app)
    if req.text:
        added = pipeline.ingest_text(req.text, req.source)
    elif req.dir:
        try:
            added = pipeline.ingest_dir(req.dir)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        raise HTTPException(status_code=400, detail="provide either 'text' or 'dir'")
    return {"ingested": added, "total": pipeline.num_chunks}

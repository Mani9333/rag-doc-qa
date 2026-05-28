# rag-doc-qa — grounded Q&A over your own documents

A small **Retrieval-Augmented Generation** service: point it at a folder of
documents and ask questions in a browser. Answers are **grounded in the retrieved
text and cite their sources**, so you can see exactly where each answer came
from.

It's built to show the *mechanics* of RAG — chunking, embeddings, vector search,
prompt construction, citations — with clean, swappable parts rather than a wall
of framework glue. It runs **offline with zero API keys and no model downloads**
(a hashing embedder + a deterministic extractive model), and upgrades to real
embeddings and a real LLM through environment variables.

## Architecture

```
 Ingest ──▶  load ─▶ chunk ─▶ embed ─▶ ┌───────────────┐
 (folder)                              │  vector store  │  (in-memory, cosine)
                                       └──────┬────────┘
                                              │ top-k
 Ask ────▶ embed query ──────────────────────┤
 (browser / API)                              ▼
                          build grounded prompt (numbered context)
                                              │
                                              ▼
                              ChatModel  ─▶  answer + [citations]
                     (mock | openai | anthropic | ollama)
```

Two things make it trustworthy: the prompt constrains the model to answer **only
from the numbered context**, and every response returns the **source + score**
of each retrieved chunk.

## Repository layout

```
rag/
├── loaders.py       # read .md/.txt/.py from a folder
├── chunking.py      # paragraph-packing splitter with overlap (bounded chunks)
├── embeddings.py    # Embedder: HashingEmbedder (default) | sentence-transformers | openai
├── vector_store.py  # in-memory cosine top-k, or Qdrant via Docker (VECTOR_STORE)
├── prompts.py       # grounded prompt with numbered, citable context
├── pipeline.py      # ingest → chunk → embed → store; ask → retrieve → answer  ← the core
├── main.py          # FastAPI: / (UI), /healthz, /ask, /ingest
└── llm/             # tiny text-in/text-out model layer (shared design)
static/index.html    # minimal web UI
data/sample_docs/    # a few docs so it works out of the box
tests/               # hermetic pytest suite
```

## Quick start (local, no keys)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

uvicorn rag.main:app --reload --port 8000
# open http://localhost:8000  → ask a question, see cited sources
```

On startup it ingests `data/sample_docs/`. Ask from the API too:

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"How does the vector store rank results?"}' | python -m json.tool

# add your own docs at runtime
curl -s localhost:8000/ingest -H 'content-type: application/json' \
  -d '{"dir":"/path/to/your/notes"}'
```

## Local setup (all options)

Everything is chosen by environment variables — no code changes. The defaults
(`LLM_PROVIDER=mock`, `EMBEDDINGS=hashing`, `VECTOR_STORE=memory`) need **no
keys, no downloads, and no Docker**. Mix and match the pieces below; all settings
live in [`.env.example`](.env.example) and load automatically from a `.env`.

**Answer model** — pick one:

```bash
# a) Offline (default): deterministic, extractive, zero setup
export LLM_PROVIDER=mock

# b) Ollama — local & free. Install from https://ollama.com, then:
ollama pull llama3.1
export LLM_PROVIDER=ollama OLLAMA_MODEL=llama3.1
#   ...or run Ollama in Docker instead of installing it:
#   docker run -d --name ollama -p 11434:11434 ollama/ollama
#   docker exec -it ollama ollama pull llama3.1
#   export LLM_PROVIDER=ollama OLLAMA_HOST=http://localhost:11434

# c) Hosted API (needs a key)
export LLM_PROVIDER=openai    OPENAI_API_KEY=sk-...          # or:
export LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-ant-...
```

**Embeddings** — default is offline; upgrade for real semantic search:

```bash
export EMBEDDINGS=hashing                                   # default, no download
# real semantic vectors (downloads a ~90 MB model once):
pip install -r requirements-optional.txt
export EMBEDDINGS=sentence-transformers ST_MODEL=all-MiniLM-L6-v2
# or hosted embeddings:
export EMBEDDINGS=openai OPENAI_API_KEY=sk-... OPENAI_EMBED_MODEL=text-embedding-3-small
```

**Vector store** — default is in-memory; use a real vector DB via Docker:

```bash
# start Qdrant (persistent vector database) in Docker:
docker run -d --name rag-qdrant -p 6333:6333 qdrant/qdrant

export VECTOR_STORE=qdrant QDRANT_URL=http://localhost:6333
uvicorn rag.main:app --reload --port 8000     # now backed by Qdrant
```

The Qdrant store uses Qdrant's REST API directly (no extra Python dependency) and
recreates its collection on startup so a fresh ingest is deterministic
(`QDRANT_RECREATE=false` to persist). Everything else is unchanged.

> Example — real stack end to end: `LLM_PROVIDER=ollama` +
> `EMBEDDINGS=sentence-transformers` + `VECTOR_STORE=qdrant` gives fully local,
> free, semantic RAG with a persistent database and no API keys.

## How retrieval works

- **Chunking** packs whole paragraphs up to `CHUNK_SIZE` characters with a small
  `CHUNK_OVERLAP` carried across boundaries; oversized paragraphs are
  sliding-window split. Every chunk stays within `CHUNK_SIZE`.
- **Embeddings** map text to L2-normalised vectors. The default `HashingEmbedder`
  uses the feature-hashing trick (no downloads, deterministic); swap in
  sentence-transformers or OpenAI for true semantic similarity.
- **Search** is cosine similarity. Because vectors are normalised, ranking all
  chunks is a single matrix-vector product plus a top-k selection.

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

Hermetic and offline: unit tests for chunking, embeddings, and the vector store;
an end-to-end pipeline test over the sample docs; and FastAPI endpoint tests via
`TestClient`. No network, deterministic results.

There's also an **integration test** for the Qdrant store that **auto-skips**
unless Qdrant is reachable — start it first to run it:

```bash
docker run -d --name rag-qdrant -p 6333:6333 qdrant/qdrant
pytest tests/test_qdrant.py
```

## Design decisions & tradeoffs

- **Pluggable embedder + vector store behind tiny interfaces.** The default path
  (hashing + in-memory NumPy) has zero heavy dependencies and runs instantly; a
  real path (sentence-transformers/OpenAI embeddings + a **Qdrant** database in
  Docker) is a pure env-var switch. The `Embedder.embed` and vector store
  `add`/`search` surfaces are small, so adding FAISS/pgvector later stays local.
- **Extractive mock model.** So the project is genuinely useful offline: it
  returns the most relevant sentences from the top source with a citation. It's
  not fluent — set `LLM_PROVIDER` to a real model for that — but it exercises the
  full retrieve-then-cite path and keeps tests deterministic.
- **Citations are first-class.** Every answer carries the source and similarity
  score of each retrieved chunk; the UI renders them. Grounding + attribution is
  the whole point of RAG.
- **Character chunking, not tokens.** Dependency-free and predictable; a
  token-aware splitter (tiktoken) is the swap when you need to fit a tight model
  context budget exactly.

## Notes

A focused iteration to demonstrate RAG internals end-to-end. Deliberately out of
scope: re-ranking, streaming responses, approximate-index tuning, auth, and
PDF/HTML loaders.

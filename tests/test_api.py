from fastapi.testclient import TestClient

from rag.main import app


def test_healthz_and_ask_flow():
    # The context manager triggers startup, which ingests data/sample_docs.
    with TestClient(app) as client:
        health = client.get("/healthz").json()
        assert health["status"] == "ok"
        assert health["chunks"] > 0
        assert health["embedder"] == "hashing"

        resp = client.post("/ask", json={"question": "Why chunk documents before embedding?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"]
        assert len(data["citations"]) >= 1
        assert "source" in data["citations"][0]


def test_ingest_inline_text():
    with TestClient(app) as client:
        before = client.get("/healthz").json()["chunks"]
        resp = client.post("/ingest", json={"text": "Zebras are striped mammals.", "source": "zoo"})
        assert resp.status_code == 200
        assert resp.json()["total"] > before

        answer = client.post("/ask", json={"question": "Tell me about zebras"}).json()
        assert any(c["source"] == "zoo" for c in answer["citations"])


def test_ask_requires_question():
    with TestClient(app) as client:
        assert client.post("/ask", json={}).status_code == 422

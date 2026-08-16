from pathlib import Path

from fastapi.testclient import TestClient

from docrag.api import app
from docrag.evaluate import evaluate_strategies, make_queries_from_docs
from docrag.loaders import load_documents
from docrag.pipeline import RagEngine

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "samples"


def test_auto_queries_and_strategy_report(tmp_path):
    docs = load_documents(SAMPLE)
    cases = make_queries_from_docs(docs, limit=8)
    assert cases
    engine = RagEngine(index_dir=tmp_path)
    engine.ingest(SAMPLE)
    report = evaluate_strategies(engine, docs, k=3)
    names = {row["strategy"] for row in report["strategies"]}
    assert names == set(engine.stores)
    assert report["winner"] in names
    assert report["query_count"] >= 1
    best = next(r for r in report["strategies"] if r["strategy"] == report["winner"])
    assert 0 <= best["hit_at_k"] <= 1
    assert best["chunk_count"] > 0


def test_dashboard_and_evaluate_samples():
    client = TestClient(app)
    home = client.get("/")
    assert home.status_code == 200
    assert b"DocRAG eval dashboard" in home.content
    res = client.post("/evaluate-samples?k=3")
    assert res.status_code == 200
    body = res.json()
    assert body["winner"]
    assert len(body["strategies"]) == 5
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert metrics.json()["winner"] == body["winner"]

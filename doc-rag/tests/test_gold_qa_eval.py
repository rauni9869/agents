from pathlib import Path

from fastapi.testclient import TestClient

from docrag.api import app
from docrag.evaluate import evaluate_strategies, load_qa
from docrag.loaders import load_documents
from docrag.pipeline import RagEngine

GOLD = Path(__file__).resolve().parents[1] / "data" / "gold"
SAMPLE = Path(__file__).resolve().parents[1] / "data" / "samples"


def test_load_gold_qa():
    cases = load_qa(GOLD / "qa.json")
    assert len(cases) == 10
    assert any("paternity" in c.question.lower() for c in cases)
    assert any("AuthBridge" in c.must_include for c in cases)


def test_gold_qa_eval_on_handbook(tmp_path):
    docs = load_documents(GOLD)
    engine = RagEngine(index_dir=tmp_path)
    engine.ingest(GOLD)
    report = evaluate_strategies(engine, docs, k=3, cases=load_qa(GOLD / "qa.json"))
    assert report["mode"] == "gold_qa"
    assert report["query_count"] == 10
    assert report["winner"] in engine.stores
    assert len(report["per_question"]) == 10
    # At least one chunker should recover most gold answers on this handbook.
    best = max(r["hit_at_k"] for r in report["strategies"])
    assert best >= 0.7


def test_dashboard_gold_eval():
    client = TestClient(app)
    res = client.post("/evaluate-samples?k=3")
    assert res.status_code == 200
    body = res.json()
    assert body["mode"] == "gold_qa"
    assert body["query_count"] == 10
    assert "BitLocker" in body["per_question"][5]["answer"]

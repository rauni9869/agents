from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from docrag.evaluate import evaluate_strategies, hit_at_k, load_qa
from docrag.loaders import load_documents
from docrag.pipeline import RagEngine

STATIC = Path(__file__).resolve().parent / "static"
ROOT = Path(__file__).resolve().parents[2]
UPLOADS = ROOT / ".docrag" / "uploads"
DOCS_DIR = UPLOADS / "docs"
QA_DIR = UPLOADS / "qa"
GOLD = ROOT / "data" / "gold"

app = FastAPI(title="DocRAG", description="Upload a document and gold Q&A, compare chunkers")
engine = RagEngine(index_dir=ROOT / ".docrag" / "index")
engine.load()
LAST_EVAL: dict = {}
LAST_DOCS: list[tuple[str, str]] = []

app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def dashboard():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "stores": list(engine.stores),
        "docs": [src for src, _ in LAST_DOCS],
    }


@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)) -> dict:
    if UPLOADS.exists():
        shutil.rmtree(UPLOADS)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)
    saved_docs: list[str] = []
    saved_qa: list[str] = []
    for item in files:
        name = Path(item.filename or "upload.txt").name
        suffix = Path(name).suffix.lower()
        dest_dir = QA_DIR if suffix in {".json", ".jsonl", ".csv"} else DOCS_DIR
        dest = dest_dir / name
        dest.write_bytes(await item.read())
        (saved_qa if dest_dir == QA_DIR else saved_docs).append(name)
    return {"docs": saved_docs, "qa": saved_qa, "count": len(saved_docs) + len(saved_qa)}


def _run(docs_path: Path, k: int, qa_path: Path | None = None) -> dict:
    global LAST_EVAL, LAST_DOCS
    LAST_DOCS = load_documents(docs_path)
    engine.ingest(docs_path)
    cases = []
    if qa_path and Path(qa_path).exists():
        cases = load_qa(qa_path)
    elif docs_path.exists():
        cases = load_qa(docs_path)
    LAST_EVAL = evaluate_strategies(engine, LAST_DOCS, k=k, cases=cases or None)
    LAST_EVAL["documents"] = [src for src, _ in LAST_DOCS]
    return LAST_EVAL


@app.post("/evaluate")
def run_evaluate(k: int = 3) -> dict:
    if DOCS_DIR.exists() and any(DOCS_DIR.iterdir()):
        qa = QA_DIR if QA_DIR.exists() and any(QA_DIR.iterdir()) else None
        return _run(DOCS_DIR, k, qa)
    return _run(GOLD, k, GOLD)


@app.post("/evaluate-samples")
def evaluate_samples(k: int = 3) -> dict:
    return _run(GOLD, k, GOLD)


@app.get("/metrics")
def metrics() -> dict:
    if not LAST_EVAL:
        return JSONResponse({"error": "No evaluation yet. Upload a doc and Q&A, then Evaluate."}, status_code=404)
    return LAST_EVAL


@app.get("/query")
def query(
    q: str = Query(..., min_length=2),
    strategy: str = "recursive",
    k: int = 4,
) -> dict:
    hits = engine.query(q, strategy, k=k)
    return {
        "question": q,
        "strategy": strategy,
        "hits": [
            {"score": round(h.score, 4), "source": h.chunk.source, "text": h.chunk.text}
            for h in hits
        ],
    }


@app.get("/compare")
def compare(q: str = Query(..., min_length=2), k: int = 3) -> dict:
    return {
        name: [{"score": round(h.score, 4), "text": h.chunk.text[:500], "source": h.chunk.source} for h in hits]
        for name, hits in engine.compare(q, k=k).items()
    }


@app.get("/eval")
def eval_chunkers(k: int = 3) -> dict[str, float]:
    return hit_at_k(engine, k=k)

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from docrag.evaluate import evaluate_strategies, hit_at_k
from docrag.loaders import load_documents
from docrag.pipeline import RagEngine

STATIC = Path(__file__).resolve().parent / "static"
ROOT = Path(__file__).resolve().parents[2]
UPLOADS = ROOT / ".docrag" / "uploads"
SAMPLES = ROOT / "data" / "samples"

app = FastAPI(title="DocRAG", description="Upload docs, compare chunkers, view eval metrics")
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
    UPLOADS.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for item in files:
        name = Path(item.filename or "upload.txt").name
        dest = UPLOADS / name
        dest.write_bytes(await item.read())
        saved.append(name)
    return {"saved": saved, "count": len(saved)}


def _run(source: Path, k: int) -> dict:
    global LAST_EVAL, LAST_DOCS
    LAST_DOCS = load_documents(source)
    engine.ingest(source)
    LAST_EVAL = evaluate_strategies(engine, LAST_DOCS, k=k)
    LAST_EVAL["documents"] = [src for src, _ in LAST_DOCS]
    return LAST_EVAL


@app.post("/evaluate")
def run_evaluate(k: int = 3) -> dict:
    source = UPLOADS if UPLOADS.exists() and any(UPLOADS.iterdir()) else SAMPLES
    return _run(source, k)


@app.post("/evaluate-samples")
def evaluate_samples(k: int = 3) -> dict:
    return _run(SAMPLES, k)


@app.get("/metrics")
def metrics() -> dict:
    if not LAST_EVAL:
        return JSONResponse({"error": "No evaluation yet. Upload a doc and click Evaluate."}, status_code=404)
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

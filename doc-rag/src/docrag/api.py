from fastapi import FastAPI, Query

from docrag.evaluate import hit_at_k
from docrag.pipeline import RagEngine

app = FastAPI(title="DocRAG", description="Document RAG with multiple chunking strategies")
engine = RagEngine()
engine.load()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "stores": ",".join(engine.stores) or "empty"}


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
        name: [{"score": round(h.score, 4), "text": h.chunk.text[:500]} for h in hits]
        for name, hits in engine.compare(q, k=k).items()
    }


@app.get("/eval")
def eval_chunkers(k: int = 3) -> dict[str, float]:
    return hit_at_k(engine, k=k)

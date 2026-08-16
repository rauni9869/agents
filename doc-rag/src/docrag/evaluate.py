from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from docrag.chunking.base import split_sentences
from docrag.pipeline import RagEngine


@dataclass
class EvalCase:
    question: str
    must_include: str


SAMPLE_CASES = [
    EvalCase("What is the return window for Nexus headphones?", "30 days"),
    EvalCase("Who is the support email for Nimbus Cloud?", "support@nimbus.example"),
    EvalCase("What embedding model does Atlas RAG use by default?", "all-MiniLM-L6-v2"),
]


def hit_at_k(engine: RagEngine, cases: list[EvalCase] | None = None, k: int = 3) -> dict[str, float]:
    cases = cases or SAMPLE_CASES
    if not engine.stores:
        engine.load()
    scores: dict[str, float] = {}
    for strategy in engine.stores:
        hits = 0
        for case in cases:
            retrieved = engine.query(case.question, strategy, k=k)
            blob = " ".join(item.chunk.text for item in retrieved).lower()
            if case.must_include.lower() in blob:
                hits += 1
        scores[strategy] = hits / max(len(cases), 1)
    return scores


def make_queries_from_docs(docs: list[tuple[str, str]], limit: int = 12) -> list[EvalCase]:
    """Turn document sentences into retrieval probes (works for any uploaded file)."""
    cases: list[EvalCase] = []
    for _source, text in docs:
        for sent in split_sentences(text):
            clean = " ".join(sent.strip().lstrip("#-").strip().split())
            if len(clean) < 40 or len(clean) > 220:
                continue
            needle = clean[:80]
            cases.append(EvalCase(question=clean, must_include=needle))
    # Prefer diverse probes: even spread
    if len(cases) > limit:
        step = max(1, len(cases) // limit)
        cases = cases[::step][:limit]
    return cases


def _chunk_stats(engine: RagEngine, strategy: str) -> dict[str, float | int]:
    lengths = [len(c.text) for c in engine.stores[strategy].chunks]
    arr = np.array(lengths, dtype=np.float32) if lengths else np.array([0.0])
    p95_idx = min(len(arr) - 1, int(round(0.95 * (len(arr) - 1))))
    return {
        "chunk_count": len(lengths),
        "avg_chunk_len": round(float(arr.mean()), 1),
        "p95_chunk_len": round(float(sorted(arr)[p95_idx]), 1),
    }


def evaluate_strategies(
    engine: RagEngine,
    docs: list[tuple[str, str]] | None = None,
    k: int = 3,
    limit_queries: int = 12,
) -> dict:
    if not engine.stores:
        engine.load()
    cases = make_queries_from_docs(docs or [], limit=limit_queries)
    if not cases:
        cases = SAMPLE_CASES

    reports: list[dict] = []
    for strategy in engine.stores:
        stats = _chunk_stats(engine, strategy)
        hits1 = hits3 = 0
        rr_sum = 0.0
        top1_sum = 0.0
        t0 = time.perf_counter()
        for case in cases:
            retrieved = engine.query(case.question, strategy, k=max(k, 5))
            needle = case.must_include.lower()
            texts = [item.chunk.text.lower() for item in retrieved]
            if texts:
                top1_sum += retrieved[0].score
            if any(needle in t for t in texts[:1]):
                hits1 += 1
            if any(needle in t for t in texts[:k]):
                hits3 += 1
            rank = next((i + 1 for i, t in enumerate(texts) if needle in t), None)
            rr_sum += (1.0 / rank) if rank else 0.0
        elapsed_ms = (time.perf_counter() - t0) * 1000
        n = max(len(cases), 1)
        reports.append(
            {
                "strategy": strategy,
                **stats,
                "queries": len(cases),
                "hit_at_1": round(hits1 / n, 4),
                "hit_at_k": round(hits3 / n, 4),
                "mrr": round(rr_sum / n, 4),
                "mean_top1_score": round(top1_sum / n, 4),
                "latency_ms": round(elapsed_ms / n, 2),
            }
        )

    ranked = sorted(reports, key=lambda r: (r["hit_at_k"], r["mrr"], r["mean_top1_score"]), reverse=True)
    winner = ranked[0]["strategy"] if ranked else None
    return {
        "k": k,
        "query_count": len(cases),
        "queries": [c.question for c in cases],
        "winner": winner,
        "strategies": reports,
    }

from __future__ import annotations

from dataclasses import dataclass

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

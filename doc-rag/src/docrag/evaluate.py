from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from docrag.chunking.base import split_sentences
from docrag.pipeline import RagEngine

QA_SUFFIXES = {".json", ".jsonl", ".csv"}


@dataclass
class EvalCase:
    question: str
    must_include: str
    answer: str = ""
    case_id: str = ""


SAMPLE_CASES = [
    EvalCase("What is the return window for Nexus headphones?", "30 days", "30 days", "sample-1"),
]


def _norm(text: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text)
    return " ".join(cleaned.split())


def answer_in_text(haystack: str, needle: str) -> bool:
    return _norm(needle) in _norm(haystack) if needle.strip() else False


def load_qa(path: str | Path) -> list[EvalCase]:
    """Load gold Q&A from .json, .jsonl, or .csv (columns: question, answer)."""
    root = Path(path)
    files: list[Path]
    if root.is_file():
        files = [root]
    else:
        files = sorted(p for p in root.rglob("*") if p.suffix.lower() in QA_SUFFIXES)
    cases: list[EvalCase] = []
    for file_path in files:
        cases.extend(_parse_qa_file(file_path))
    return cases


def _parse_qa_file(path: Path) -> list[EvalCase]:
    suffix = path.suffix.lower()
    rows: list[dict]
    if suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        rows = raw if isinstance(raw, list) else raw.get("items") or raw.get("qa") or []
    elif suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    cases: list[EvalCase] = []
    for i, row in enumerate(rows):
        question = str(row.get("question") or row.get("q") or "").strip()
        answer = str(row.get("answer") or row.get("a") or "").strip()
        needle = str(row.get("must_include") or answer).strip()
        if not question or not needle:
            continue
        cases.append(
            EvalCase(
                question=question,
                must_include=needle,
                answer=answer or needle,
                case_id=str(row.get("id") or f"{path.stem}-{i+1}"),
            )
        )
    return cases


def hit_at_k(engine: RagEngine, cases: list[EvalCase] | None = None, k: int = 3) -> dict[str, float]:
    cases = cases or SAMPLE_CASES
    if not engine.stores:
        engine.load()
    scores: dict[str, float] = {}
    for strategy in engine.stores:
        hits = 0
        for case in cases:
            retrieved = engine.query(case.question, strategy, k=k)
            blob = " ".join(item.chunk.text for item in retrieved)
            if answer_in_text(blob, case.must_include):
                hits += 1
        scores[strategy] = hits / max(len(cases), 1)
    return scores


def make_queries_from_docs(docs: list[tuple[str, str]], limit: int = 12) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for _source, text in docs:
        for sent in split_sentences(text):
            clean = " ".join(sent.strip().lstrip("#-").strip().split())
            if len(clean) < 40 or len(clean) > 220:
                continue
            needle = clean[:80]
            cases.append(EvalCase(question=clean, must_include=needle, answer=needle))
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
    cases: list[EvalCase] | None = None,
) -> dict:
    if not engine.stores:
        engine.load()
    mode = "gold_qa"
    if cases is None:
        cases = make_queries_from_docs(docs or [], limit=limit_queries)
        mode = "auto_sentence"
    if not cases:
        cases = SAMPLE_CASES
        mode = "sample_fallback"

    reports: list[dict] = []
    per_question: list[dict] = []

    for case in cases:
        row = {
            "id": case.case_id,
            "question": case.question,
            "answer": case.answer or case.must_include,
            "by_strategy": {},
        }
        for strategy in engine.stores:
            retrieved = engine.query(case.question, strategy, k=max(k, 5))
            texts = [item.chunk.text for item in retrieved]
            rank = next(
                (i + 1 for i, text in enumerate(texts) if answer_in_text(text, case.must_include)),
                None,
            )
            row["by_strategy"][strategy] = {
                "hit_at_1": bool(rank == 1),
                "hit_at_k": bool(rank is not None and rank <= k),
                "rank": rank,
                "top1_score": round(retrieved[0].score, 4) if retrieved else 0.0,
                "top_chunk": (retrieved[0].chunk.text[:280] if retrieved else ""),
            }
        per_question.append(row)

    for strategy in engine.stores:
        stats = _chunk_stats(engine, strategy)
        hits1 = hits3 = 0
        rr_sum = 0.0
        top1_sum = 0.0
        t0 = time.perf_counter()
        for case in cases:
            retrieved = engine.query(case.question, strategy, k=max(k, 5))
            texts = [item.chunk.text for item in retrieved]
            if retrieved:
                top1_sum += retrieved[0].score
            rank = next(
                (i + 1 for i, text in enumerate(texts) if answer_in_text(text, case.must_include)),
                None,
            )
            if rank == 1:
                hits1 += 1
            if rank is not None and rank <= k:
                hits3 += 1
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
        "mode": mode,
        "query_count": len(cases),
        "queries": [c.question for c in cases],
        "winner": winner,
        "strategies": reports,
        "per_question": per_question,
    }

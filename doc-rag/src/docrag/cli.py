from __future__ import annotations

import argparse
import json

from docrag.chunking import ALL_STRATEGIES
from docrag.evaluate import hit_at_k
from docrag.pipeline import RagEngine


def _parse_strategies(raw: str | None) -> list[str]:
    if not raw:
        return list(ALL_STRATEGIES)
    return [part.strip() for part in raw.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="docrag", description="RAG search with multiple chunkers")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ingest = sub.add_parser("ingest", help="Index documents")
    ingest.add_argument("path")
    ingest.add_argument("--strategies", default=",".join(ALL_STRATEGIES))
    ingest.add_argument("--index", default=".docrag")

    query = sub.add_parser("query", help="Search one or all chunkers")
    query.add_argument("question")
    query.add_argument("--strategy", default="recursive")
    query.add_argument("--compare", action="store_true")
    query.add_argument("--k", type=int, default=3)
    query.add_argument("--index", default=".docrag")

    ev = sub.add_parser("eval", help="Hit-rate per chunker on sample questions")
    ev.add_argument("--index", default=".docrag")
    ev.add_argument("--k", type=int, default=3)

    serve = sub.add_parser("serve", help="Start FastAPI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    engine = RagEngine(index_dir=getattr(args, "index", ".docrag"))

    if args.cmd == "ingest":
        counts = engine.ingest(args.path, _parse_strategies(args.strategies))
        print(json.dumps({"chunks": counts}, indent=2))
        return

    if args.cmd == "query":
        engine.load(_parse_strategies(None) if args.compare else [args.strategy])
        if args.compare:
            result = {
                name: [{"score": round(h.score, 4), "text": h.chunk.text[:400], "source": h.chunk.source}
                       for h in hits]
                for name, hits in engine.compare(args.question, k=args.k).items()
            }
            print(json.dumps(result, indent=2))
            return
        print(engine.answer(args.question, args.strategy, k=args.k))
        return

    if args.cmd == "eval":
        engine.load()
        print(json.dumps(hit_at_k(engine, k=args.k), indent=2))
        return

    if args.cmd == "serve":
        import uvicorn
        from docrag.api import app

        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

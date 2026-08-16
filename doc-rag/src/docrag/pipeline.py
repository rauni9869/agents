from __future__ import annotations

from pathlib import Path

from docrag.chunking import ALL_STRATEGIES, build_chunker
from docrag.embeddings import Embedder, TfidfEmbedder
from docrag.loaders import load_documents
from docrag.store import VectorStore
from docrag.types import ScoredChunk

DEFAULT_INDEX = Path(".docrag")


class RagEngine:
    def __init__(self, embedder: Embedder | None = None, index_dir: str | Path = DEFAULT_INDEX):
        self.embedder = embedder or TfidfEmbedder()
        self.index_dir = Path(index_dir)
        self.stores: dict[str, VectorStore] = {}

    def ingest(self, data_path: str | Path, strategies: list[str] | None = None) -> dict[str, int]:
        strategies = strategies or list(ALL_STRATEGIES)
        docs = load_documents(data_path)
        by_strategy: dict[str, list] = {}
        corpus: list[str] = [text for _, text in docs]
        for name in strategies:
            chunker = build_chunker(name, self.embedder)
            chunks = []
            for source, text in docs:
                chunks.extend(chunker.split(text, source))
            by_strategy[name] = chunks
            corpus.extend(c.text for c in chunks)

        self.embedder.fit(corpus)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(self.embedder, "save"):
            self.embedder.save(self.index_dir)

        counts: dict[str, int] = {}
        for name, chunks in by_strategy.items():
            vectors = self.embedder.embed([c.text for c in chunks])
            store = VectorStore()
            store.add(chunks, vectors)
            store.save(self.index_dir / name)
            self.stores[name] = store
            counts[name] = len(chunks)
        return counts

    def load(self, strategies: list[str] | None = None) -> None:
        if hasattr(self.embedder, "load") and (self.index_dir / "tfidf_vocab.json").exists():
            self.embedder.load(self.index_dir)
        strategies = strategies or list(ALL_STRATEGIES)
        for name in strategies:
            path = self.index_dir / name
            if path.exists():
                self.stores[name] = VectorStore.load(path)

    def query(self, question: str, strategy: str, k: int = 4) -> list[ScoredChunk]:
        if strategy not in self.stores:
            self.load([strategy])
        store = self.stores[strategy]
        qvec = self.embedder.embed([question])[0]
        return store.search(qvec, k=k)

    def compare(self, question: str, k: int = 4) -> dict[str, list[ScoredChunk]]:
        if not self.stores:
            self.load()
        return {name: self.query(question, name, k=k) for name in self.stores}

    def answer(self, question: str, strategy: str, k: int = 4) -> str:
        hits = self.query(question, strategy, k=k)
        if not hits:
            return "No matching chunks."
        context = "\n\n".join(f"[{h.chunk.source} #{h.chunk.index}] {h.chunk.text}" for h in hits)
        return (
            f"Question: {question}\n"
            f"Chunker: {strategy}\n\n"
            f"Retrieved context:\n{context}\n"
        )

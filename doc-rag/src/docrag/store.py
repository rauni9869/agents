from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from docrag.embeddings import cosine_scores
from docrag.types import Chunk, ScoredChunk


class VectorStore:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        self.chunks.extend(chunks)
        if self.matrix.size == 0:
            self.matrix = embeddings.astype(np.float32)
        else:
            self.matrix = np.vstack([self.matrix, embeddings.astype(np.float32)])

    def search(self, query_vec: np.ndarray, k: int = 4) -> list[ScoredChunk]:
        if not self.chunks:
            return []
        scores = cosine_scores(query_vec.astype(np.float32), self.matrix)
        k = min(k, len(self.chunks))
        top = np.argsort(scores)[::-1][:k]
        return [ScoredChunk(chunk=self.chunks[i], score=float(scores[i])) for i in top]

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "source": c.source,
                "strategy": c.strategy,
                "index": c.index,
                "metadata": c.metadata,
            }
            for c in self.chunks
        ]
        (path / "chunks.json").write_text(json.dumps(payload, indent=2))
        np.save(path / "embeddings.npy", self.matrix)

    @classmethod
    def load(cls, directory: str | Path) -> VectorStore:
        path = Path(directory)
        payload = json.loads((path / "chunks.json").read_text())
        store = cls()
        store.chunks = [Chunk(**item) for item in payload]
        store.matrix = np.load(path / "embeddings.npy")
        return store

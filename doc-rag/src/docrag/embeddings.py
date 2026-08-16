from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import json

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def cosine_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.array([])
    qn = np.linalg.norm(query)
    mn = np.linalg.norm(matrix, axis=1)
    dots = matrix @ query
    denom = np.clip(mn * qn, 1e-12, None)
    return dots / denom


class Embedder(ABC):
    name: str

    @abstractmethod
    def fit(self, texts: list[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class TfidfEmbedder(Embedder):
    """Offline bag-of-words embeddings. Good for demos and tests with no API key."""

    name = "tfidf"

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray = np.array([])

    def fit(self, texts: list[str]) -> None:
        df: dict[str, int] = {}
        for text in texts:
            for token in set(_tokenize(text)):
                df[token] = df.get(token, 0) + 1
        self.vocab = {tok: i for i, tok in enumerate(sorted(df))}
        n_docs = max(len(texts), 1)
        idf = np.zeros(len(self.vocab), dtype=np.float32)
        for tok, idx in self.vocab.items():
            idf[idx] = np.log((1 + n_docs) / (1 + df[tok])) + 1.0
        self.idf = idf

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tfidf_vocab.json").write_text(json.dumps(self.vocab))
        np.save(path / "tfidf_idf.npy", self.idf)

    def load(self, directory: str | Path) -> None:
        path = Path(directory)
        self.vocab = {k: int(v) for k, v in json.loads((path / "tfidf_vocab.json").read_text()).items()}
        self.idf = np.load(path / "tfidf_idf.npy")

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self.vocab:
            self.fit(texts)
        matrix = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[int, int] = {}
            tokens = _tokenize(text)
            for tok in tokens:
                idx = self.vocab.get(tok)
                if idx is not None:
                    counts[idx] = counts.get(idx, 0) + 1
            if not tokens:
                continue
            for idx, count in counts.items():
                matrix[row, idx] = (count / len(tokens)) * self.idf[idx]
        return matrix


def _tokenize(text: str) -> list[str]:
    return [part for part in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if part]

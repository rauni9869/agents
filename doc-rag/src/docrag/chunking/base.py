from __future__ import annotations

import re
from abc import ABC, abstractmethod

from docrag.types import Chunk


class Chunker(ABC):
    name: str

    @abstractmethod
    def split(self, text: str, source: str) -> list[Chunk]:
        raise NotImplementedError


def _chunks(texts: list[str], source: str, strategy: str) -> list[Chunk]:
    out: list[Chunk] = []
    for i, raw in enumerate(texts):
        text = raw.strip()
        if not text:
            continue
        out.append(
            Chunk(
                chunk_id=f"{strategy}:{source}:{i}",
                text=text,
                source=source,
                strategy=strategy,
                index=i,
            )
        )
    return out


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]

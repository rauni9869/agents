from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    strategy: str
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float

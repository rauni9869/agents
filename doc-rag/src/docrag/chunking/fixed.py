from docrag.chunking.base import Chunker, _chunks
from docrag.types import Chunk


class FixedChunker(Chunker):
    name = "fixed"

    def __init__(self, size: int = 500, overlap: int = 80):
        if overlap >= size:
            raise ValueError("overlap must be smaller than size")
        self.size = size
        self.overlap = overlap

    def split(self, text: str, source: str) -> list[Chunk]:
        pieces: list[str] = []
        start = 0
        step = self.size - self.overlap
        while start < len(text):
            pieces.append(text[start : start + self.size])
            start += step
        return _chunks(pieces, source, self.name)

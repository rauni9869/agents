from docrag.chunking.base import Chunker, _chunks
from docrag.types import Chunk

SEPARATORS = ["\n\n", "\n", ". ", " "]


class RecursiveChunker(Chunker):
    name = "recursive"

    def __init__(self, size: int = 500, overlap: int = 60):
        self.size = size
        self.overlap = overlap

    def split(self, text: str, source: str) -> list[Chunk]:
        parts = self._split(text, SEPARATORS)
        merged = self._merge(parts)
        return _chunks(merged, source, self.name)

    def _split(self, text: str, seps: list[str]) -> list[str]:
        if len(text) <= self.size or not seps:
            return [text]
        sep = seps[0]
        bits = text.split(sep) if sep else list(text)
        out: list[str] = []
        for bit in bits:
            if len(bit) > self.size:
                out.extend(self._split(bit, seps[1:]))
            else:
                out.append(bit)
        return out

    def _merge(self, parts: list[str]) -> list[str]:
        chunks: list[str] = []
        buf = ""
        for part in parts:
            candidate = f"{buf} {part}".strip() if buf else part
            if len(candidate) <= self.size:
                buf = candidate
                continue
            if buf:
                chunks.append(buf)
            if self.overlap and chunks:
                tail = chunks[-1][-self.overlap :]
                buf = f"{tail} {part}".strip()
            else:
                buf = part
        if buf:
            chunks.append(buf)
        return chunks

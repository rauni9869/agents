from docrag.chunking.base import Chunker, _chunks, split_sentences
from docrag.types import Chunk


class SentenceChunker(Chunker):
    name = "sentence"

    def __init__(self, size: int = 500):
        self.size = size

    def split(self, text: str, source: str) -> list[Chunk]:
        sentences = split_sentences(text)
        packed: list[str] = []
        buf: list[str] = []
        length = 0
        for sent in sentences:
            extra = len(sent) + (1 if buf else 0)
            if buf and length + extra > self.size:
                packed.append(" ".join(buf))
                buf = [sent]
                length = len(sent)
            else:
                buf.append(sent)
                length += extra
        if buf:
            packed.append(" ".join(buf))
        return _chunks(packed, source, self.name)

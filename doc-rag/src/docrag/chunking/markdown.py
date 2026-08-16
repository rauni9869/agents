from __future__ import annotations

import re

from docrag.chunking.base import Chunker, _chunks
from docrag.types import Chunk

HEADING = re.compile(r"(?m)^(#{1,6})\s+.+$")


class MarkdownChunker(Chunker):
    name = "markdown"

    def split(self, text: str, source: str) -> list[Chunk]:
        indices = [m.start() for m in HEADING.finditer(text)]
        if not indices:
            return _chunks([text], source, self.name)
        if indices[0] != 0:
            indices = [0, *indices]
        indices.append(len(text))
        pieces = [text[indices[i] : indices[i + 1]] for i in range(len(indices) - 1)]
        return _chunks(pieces, source, self.name)

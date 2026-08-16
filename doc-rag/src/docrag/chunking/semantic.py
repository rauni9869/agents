from docrag.chunking.base import Chunker, _chunks, split_sentences
from docrag.embeddings import Embedder, cosine
from docrag.types import Chunk


class SemanticChunker(Chunker):
    name = "semantic"

    def __init__(self, embedder: Embedder, threshold: float = 0.25, max_size: int = 700):
        self.embedder = embedder
        self.threshold = threshold
        self.max_size = max_size

    def split(self, text: str, source: str) -> list[Chunk]:
        sentences = split_sentences(text)
        if len(sentences) <= 1:
            return _chunks(sentences or [text], source, self.name)

        vectors = self.embedder.embed(sentences)
        groups: list[list[str]] = [[sentences[0]]]
        for i, sent in enumerate(sentences[1:]):
            similar = cosine(vectors[i], vectors[i + 1]) >= self.threshold
            too_long = sum(len(s) for s in groups[-1]) + len(sent) > self.max_size
            if similar and not too_long:
                groups[-1].append(sent)
            else:
                groups.append([sent])
        return _chunks([" ".join(g) for g in groups], source, self.name)

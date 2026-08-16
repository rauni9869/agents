from docrag.chunking.fixed import FixedChunker
from docrag.chunking.markdown import MarkdownChunker
from docrag.chunking.recursive import RecursiveChunker
from docrag.chunking.semantic import SemanticChunker
from docrag.chunking.sentence import SentenceChunker
from docrag.embeddings import Embedder

ALL_STRATEGIES = ("fixed", "recursive", "sentence", "semantic", "markdown")


def build_chunker(name: str, embedder: Embedder):
    if name == "fixed":
        return FixedChunker()
    if name == "recursive":
        return RecursiveChunker()
    if name == "sentence":
        return SentenceChunker()
    if name == "semantic":
        return SemanticChunker(embedder)
    if name == "markdown":
        return MarkdownChunker()
    raise ValueError(f"Unknown chunker: {name}. Choose from {ALL_STRATEGIES}")

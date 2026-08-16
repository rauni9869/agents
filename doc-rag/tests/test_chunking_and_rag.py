from pathlib import Path

from docrag.chunking.fixed import FixedChunker
from docrag.chunking.markdown import MarkdownChunker
from docrag.chunking.recursive import RecursiveChunker
from docrag.chunking.sentence import SentenceChunker
from docrag.embeddings import TfidfEmbedder, cosine
from docrag.evaluate import hit_at_k
from docrag.pipeline import RagEngine

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "samples"


def test_fixed_and_sentence_chunkers():
    text = "A. " * 80
    fixed = FixedChunker(size=40, overlap=10).split(text, "t.txt")
    assert len(fixed) > 1
    assert fixed[0].strategy == "fixed"
    sent = SentenceChunker(size=30).split("Hello there. More words. End.", "t.txt")
    assert all(c.text for c in sent)


def test_recursive_and_markdown():
    md = "# Title\n\nIntro text.\n\n## Next\n\nBody paragraph here."
    rec = RecursiveChunker(size=40, overlap=5).split(md, "a.md")
    assert rec
    heads = MarkdownChunker().split(md, "a.md")
    assert len(heads) >= 2
    assert heads[0].text.startswith("# Title")


def test_tfidf_cosine_and_pipeline(tmp_path):
    embedder = TfidfEmbedder()
    embedder.fit(["return window 30 days", "cloud storage plan"])
    a, b = embedder.embed(["30 days return", "unrelated zebra"])
    assert cosine(a, a) > 0.99
    assert cosine(a, b) < cosine(a, embedder.embed(["return 30 days"])[0])

    engine = RagEngine(embedder=TfidfEmbedder(), index_dir=tmp_path)
    counts = engine.ingest(SAMPLE)
    assert counts["markdown"] >= 3
    hits = engine.query("What is the return window for Nexus headphones?", "markdown", k=3)
    blob = " ".join(h.chunk.text for h in hits)
    assert "30 days" in blob

    scores = hit_at_k(engine, k=3)
    assert scores["markdown"] >= 0.6
    assert set(scores) == set(counts)

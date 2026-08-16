# DocRAG — document search with multiple chunkers

Ingest PDF / Markdown / TXT files, chunk them in more than one way, embed, and run RAG search. Compare chunking side by side.

## Chunkers

| Name | What it does |
| --- | --- |
| `fixed` | Sliding window (size + overlap) |
| `recursive` | Split by paragraph → line → sentence → space |
| `sentence` | Pack sentences until a size cap |
| `semantic` | New chunk when neighboring sentence similarity drops |
| `markdown` | Split on `#` headings |

Default embeddings are **TF-IDF** (no API key, no model download). Optional: `sentence-transformers`.

## Setup

```bash
cd doc-rag
python3 -m pip install -e ".[dev]"
docrag ingest data/samples --strategies fixed,recursive,sentence,semantic,markdown
docrag query "What is the return window for Nexus headphones?" --compare --k 3
docrag eval
```

Open the dashboard:

```bash
python3 -m docrag serve --host 0.0.0.0 --port 8000
```

Then open `http://127.0.0.1:8000`

1. Upload PDF / Markdown / TXT (or click **Use sample docs**)
2. **Run evaluation** — indexes 5 chunkers and scores **hit@1**, **hit@k**, **MRR**, latency
3. Search a question across all chunkers

API:

```bash
# GET http://127.0.0.1:8000/compare?q=return%20window
```

## Resume keywords

Python, RAG, document Q&A, chunking, embeddings, TF-IDF, vector search, cosine similarity, FastAPI, PDF ingestion, retrieval evaluation

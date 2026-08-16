# Atlas RAG intern notes

Atlas RAG indexes PDFs and Markdown for internal search.

Default embedding model: all-MiniLM-L6-v2 (384 dimensions).
Vector store: FAISS IndexFlatIP with cosine similarity.

Chunking experiments:
- Fixed windows leak tables across chunks.
- Markdown heading splits keep policy sections intact.
- Semantic splits help long FAQ pages.

Query latency target is under 200 ms for top-5 retrieval on 10k chunks.

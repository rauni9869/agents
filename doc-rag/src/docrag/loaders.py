from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

SUPPORTED = {".txt", ".md", ".markdown", ".pdf"}


def load_documents(path: str | Path) -> list[tuple[str, str]]:
    """Return (source_name, text) pairs from a file or directory."""
    root = Path(path)
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED)
    docs: list[tuple[str, str]] = []
    for file_path in files:
        text = _read_file(file_path)
        if text.strip():
            docs.append((str(file_path), text))
    if not docs:
        raise FileNotFoundError(f"No readable documents in {root}")
    return docs


def _read_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="replace")

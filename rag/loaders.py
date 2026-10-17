"""Load PDF / HTML / Markdown into Documents made of (page, text) sections.
Sections are chunked separately so a chunk never straddles a PDF page and `page` metadata is exact."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

SUPPORTED = {".pdf": "pdf", ".html": "html", ".htm": "html", ".md": "md", ".markdown": "md"}


@dataclass
class Document:
    doc_id: str                         # path relative to the corpus root: stable identity
    source: str
    doc_type: str
    sections: list[tuple[int | None, str]]

    @property
    def text(self) -> str:
        return "\n".join(t for _, t in self.sections)


def _load_pdf(path: Path):
    from pypdf import PdfReader
    return [(i + 1, (p.extract_text() or "")) for i, p in enumerate(PdfReader(str(path)).pages)]


def _load_html(path: Path):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(path.read_text(errors="ignore"), "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    return [(None, soup.get_text("\n"))]


def _load_md(path: Path):
    return [(None, path.read_text(errors="ignore"))]


def load_document(path: Path, root: Path) -> Document | None:
    kind = SUPPORTED.get(path.suffix.lower())
    if kind is None:
        return None
    sections = {"pdf": _load_pdf, "html": _load_html, "md": _load_md}[kind](path)
    sections = [(p, t) for p, t in sections if t and t.strip()]
    rel = str(path.relative_to(root)) if path.is_relative_to(root) else path.name
    return Document(doc_id=rel, source=rel, doc_type=kind, sections=sections)


def iter_documents(root: Path):
    if root.is_file():
        d = load_document(root, root.parent)
        if d:
            yield d
        return
    for p in sorted(root.rglob("*")):
        if p.is_file():
            d = load_document(p, root)
            if d:
                yield d

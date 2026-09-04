"""Document loading and chunking.

Chunking is the least glamorous part of RAG and the one that most often decides
whether it works.  Two decisions are made here and both are argued rather than
assumed.

DECISION 1: split on structure first, characters second.

Every document in this corpus is Markdown with headed sections, and a heading
is a far better boundary than any character count, because the author already
did the semantic segmentation for us.  So the splitter walks the heading tree
and only falls back to character windows when a single section is too long.

The failure this avoids is specific and severe for GraphRAG.  A fixed 900
character window cuts through the middle of a sentence like "Meridian purchases
the copper-clad laminate from Formosa Substrate Materials", leaving "Meridian
purchases the copper-clad" in one chunk and "laminate from Formosa Substrate
Materials" in the next.  Neither chunk states the relationship.  The extractor
sees no relationship, so no edge is created, so the multi-hop query returns
nothing - and the bug looks like a retrieval problem three layers away from
where it happened.

DECISION 2: keep the document title in every chunk.

A chunk that begins "Corrective action requested: move floor-life tracking..."
is unattributable on its own.  Prefixing the heading path costs about 15 tokens
and makes both the embedding and the extractor aware of what they are reading.
This is sometimes called contextual retrieval, and on a corpus like this one it
is the cheapest quality improvement available.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_FRONT_MATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


@dataclass
class Document:
    doc_id: str
    title: str
    doc_type: str
    published: str
    body: str
    source_path: str


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    ord: int
    text: str
    heading: str = ""
    embedding: list[float] = field(default_factory=list)


def load_documents(directory: Path) -> list[Document]:
    """Read every .md file and parse its front matter.

    Front matter is parsed with a regex and a two-line key/value split rather
    than a YAML library, because the format here is fixed and flat.  If the
    corpus ever gained nested front matter this should switch to yaml.safe_load
    - noted rather than pre-built, because unused generality is its own bug.
    """
    documents: list[Document] = []
    for path in sorted(directory.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        match = _FRONT_MATTER.match(raw)
        if not match:
            raise ValueError(
                f"{path.name} has no front matter block. Every document needs "
                "one: it carries doc_id, title, doc_type and published, and "
                "citations are built from those fields."
            )
        meta: dict[str, str] = {}
        for line in match.group(1).splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                meta[key.strip()] = value.strip()
        body = raw[match.end():].strip()
        documents.append(
            Document(
                doc_id=meta["doc_id"],
                title=meta.get("title", path.stem),
                doc_type=meta.get("doc_type", "unknown"),
                published=meta.get("published", ""),
                body=body,
                source_path=path.name,
            )
        )
    if not documents:
        raise ValueError(f"No .md documents found in {directory}")
    return documents


def _sections(body: str) -> list[tuple[str, str]]:
    """Split a Markdown body into (heading, text) pairs.

    Text before the first heading is kept under an empty heading rather than
    discarded - in this corpus that is where the document's opening summary
    lives, and it is often the most quotable sentence in the file.
    """
    matches = list(_HEADING.finditer(body))
    if not matches:
        return [("", body)]

    sections: list[tuple[str, str]] = []
    preamble = body[: matches[0].start()].strip()
    if preamble:
        sections.append(("", preamble))

    for i, match in enumerate(matches):
        heading = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        text = body[start:end].strip()
        if text:
            sections.append((heading, text))
    return sections


def _split_long(text: str, size: int, overlap: int) -> list[str]:
    """Character-window fallback for an oversized section.

    Breaks are pushed to the nearest paragraph, then sentence, then space, so
    the window never lands mid-word.  The overlap exists so a statement
    straddling a boundary survives in at least one chunk intact.
    """
    if len(text) <= size:
        return [text]

    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            for separator in ("\n\n", "\n", ". ", " "):
                cut = window.rfind(separator)
                # Only accept a break in the last third of the window, or we
                # would produce tiny chunks whenever an early newline appears.
                if cut > size * 0.6:
                    end = start + cut + len(separator)
                    break
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [p for p in pieces if p]


def chunk_document(doc: Document, *, chunk_size: int, chunk_overlap: int,
                   min_chunk_chars: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for heading, text in _sections(doc.body):
        for piece in _split_long(text, chunk_size, chunk_overlap):
            if len(piece) < min_chunk_chars and chunks:
                # Too small to stand alone.  Append to the previous chunk
                # rather than dropping it: a two-line section is usually a
                # continuation, and dropping content silently is never right.
                chunks[-1].text += "\n\n" + piece
                continue
            ordinal = len(chunks)
            prefix = f"{doc.title}"
            if heading:
                prefix += f" > {heading}"
            chunks.append(
                Chunk(
                    # Deterministic id: doc_id plus position.  Re-running
                    # ingestion on unchanged input produces identical ids, so
                    # MERGE updates rather than duplicates.  A random uuid here
                    # would make every re-ingest double the graph.
                    chunk_id=f"{doc.doc_id}::{ordinal:03d}",
                    doc_id=doc.doc_id,
                    ord=ordinal,
                    heading=heading,
                    text=f"[{prefix}]\n{piece}",
                )
            )
    return chunks


def chunk_documents(docs: list[Document], *, chunk_size: int,
                    chunk_overlap: int, min_chunk_chars: int) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        out.extend(
            chunk_document(
                doc,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                min_chunk_chars=min_chunk_chars,
            )
        )
    return out

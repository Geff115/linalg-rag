"""
Phase 0, part 1: parse the Linear Algebra course book into tagged chunks.

This module turns the PDF into a list of LangChain Documents, each carrying
metadata that lets an answer cite its source: unit, section, and printed page.

It deliberately does NOT embed or upload anything. Run it standalone to
inspect the output and confirm the tagging is correct before Pinecone is
involved. Ingestion (embedding + upsert) lives in ingest.py and imports
load_chunks() from here.

Key facts verified against the real PDF (not assumed from the table of
contents):
  - The book has 154 physical pages.
  - Printed page number = physical index (0-based) minus 1. This constant
    offset was confirmed at six points across the whole book, and again by
    the printed page-number in each page footer.
  - Unit and section start pages below are the real physical indices where
    each heading appears, found by scanning the extracted text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Default location in the repo. Override with load_chunks(pdf_path=...).
DEFAULT_PDF = Path("data/course_book.pdf")

# The body of the book runs from the Unit 1 divider to just before the
# backmatter. Physical indices are 0-based. Everything outside this range
# (cover, masthead, TOC, learning-objective front matter, references,
# figure lists) is excluded as non-content.
BODY_START = 12   # phys index of "UNIT 1 FOUNDATIONS" divider
BODY_END = 148    # phys index of "BACKMATTER" divider (exclusive upper bound)


@dataclass(frozen=True)
class Marker:
    """The start of a unit or section, at a known physical page index."""
    phys_start: int
    unit: int
    unit_title: str
    section: str
    section_title: str


# Ordered by phys_start. Each body page is assigned to the LAST marker whose
# phys_start is <= that page. Section "N.0" is the unit divider / study goals.
# Titles come from the table of contents (full, untruncated); the physical
# start pages come from scanning the real text.
MARKERS: list[Marker] = [
    Marker(12, 1, "Foundations", "1.0", "Unit Introduction"),
    Marker(13, 1, "Foundations", "1.1", "Systems of Linear Equations"),
    Marker(14, 1, "Foundations", "1.2", "Matrices: Basic Terms"),
    Marker(18, 1, "Foundations", "1.3", "Matrix Algebra"),
    Marker(27, 1, "Foundations", "1.4",
           "Matrices as Compact Representations of Systems of Linear Equations"),
    Marker(33, 1, "Foundations", "1.5", "Inverse and Trace"),
    Marker(42, 2, "Vector Spaces", "2.0", "Unit Introduction"),
    Marker(43, 2, "Vector Spaces", "2.1", "Definition"),
    Marker(46, 2, "Vector Spaces", "2.2",
           "Linear Combination and Linear Dependence"),
    Marker(53, 2, "Vector Spaces", "2.3", "Basis, Linear Envelope, and Rank"),
    Marker(64, 3, "Linear and Affine Mapping", "3.0", "Unit Introduction"),
    Marker(65, 3, "Linear and Affine Mapping", "3.1",
           "Matrix Representation of Linear Mappings"),
    Marker(75, 3, "Linear and Affine Mapping", "3.2", "Image and Kernel"),
    Marker(79, 3, "Linear and Affine Mapping", "3.3",
           "Affine Spaces and Subspaces"),
    Marker(81, 3, "Linear and Affine Mapping", "3.4", "Affine Mappings"),
    Marker(84, 4, "Analytical Geometry", "4.0", "Unit Introduction"),
    Marker(86, 4, "Analytical Geometry", "4.1", "Norm"),
    Marker(93, 4, "Analytical Geometry", "4.2", "Scalar Product"),
    Marker(96, 4, "Analytical Geometry", "4.3", "Orthogonal Projections"),
    Marker(108, 4, "Analytical Geometry", "4.4", "Outlook: Complex Numbers"),
    Marker(118, 5, "Matrix Decomposition", "5.0", "Unit Introduction"),
    Marker(119, 5, "Matrix Decomposition", "5.1", "Determinant"),
    Marker(124, 5, "Matrix Decomposition", "5.2",
           "Eigenvalues and Eigenvectors"),
    Marker(131, 5, "Matrix Decomposition", "5.3", "Cholesky Decomposition"),
    Marker(136, 5, "Matrix Decomposition", "5.4",
           "Eigenvalue Decomposition and Diagonalization"),
    Marker(139, 5, "Matrix Decomposition", "5.5",
           "Singular Value Decomposition"),
]


def marker_for_page(phys_index: int) -> Marker:
    """Return the unit/section that owns a given physical page.

    A page is owned by the last marker that starts on or before it. A page
    that straddles a section boundary is assigned to the earlier section;
    that is a deliberate, minor approximation, since detecting the exact
    line where one section ends and the next begins is not reliable.
    """
    owner = MARKERS[0]
    for m in MARKERS:
        if m.phys_start <= phys_index:
            owner = m
        else:
            break
    return owner


def printed_page(phys_index: int) -> int:
    """Convert a 0-based physical index to the page number printed in the PDF."""
    return phys_index - 1


_BARE_PAGENUM = re.compile(r"^\s*\d{1,3}\s*$")


def clean_page_text(text: str) -> str:
    """Strip footer page-number noise and normalize whitespace.

    We only remove a trailing line that is a bare number (the printed page
    footer). We do not try to repair equation-font contamination in prose;
    that needs a vision model and is out of scope for this phase.
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    # Drop a trailing bare page-number line if present.
    while lines and _BARE_PAGENUM.match(lines[-1]):
        lines.pop()
    cleaned = "\n".join(lines)
    # Collapse 3+ newlines into a paragraph break.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def load_chunks(
    pdf_path: Path | str = DEFAULT_PDF,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    min_chunk_chars: int = 50,
) -> list[Document]:
    """Parse the PDF into tagged, chunked Documents ready for embedding.

    Chunking is done per page so that every chunk maps to exactly one printed
    page for citation. chunk_size and chunk_overlap are the two main knobs we
    will tune in Phase 2 once we can measure retrieval quality.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Course book not found at {pdf_path}. "
            "Place the PDF at data/course_book.pdf (it is gitignored)."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    documents: list[Document] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for phys_index in range(BODY_START, BODY_END):
            raw = pdf.pages[phys_index].extract_text() or ""
            text = clean_page_text(raw)
            if len(text) < min_chunk_chars:
                continue  # near-empty page (e.g., a full-page figure)

            m = marker_for_page(phys_index)
            page = printed_page(phys_index)

            for i, piece in enumerate(splitter.split_text(text)):
                if len(piece.strip()) < min_chunk_chars:
                    continue
                # Deterministic id: re-running ingest overwrites instead of
                # creating duplicates. This avoids the duplicate-key upsert
                # problem entirely.
                chunk_id = f"u{m.unit}-s{m.section}-p{page}-c{i}"
                documents.append(
                    Document(
                        id=chunk_id,
                        page_content=piece,
                        metadata={
                            "chunk_id": chunk_id,
                            "unit": m.unit,
                            "unit_title": m.unit_title,
                            "section": m.section,
                            "section_title": m.section_title,
                            "page": page,
                            "phys_index": phys_index,
                            "source": pdf_path.name,
                        },
                    )
                )
    return documents


if __name__ == "__main__":
    import argparse
    from collections import Counter

    ap = argparse.ArgumentParser(description="Inspect chunking output.")
    ap.add_argument("--pdf", default=str(DEFAULT_PDF))
    ap.add_argument("--samples", type=int, default=3)
    args = ap.parse_args()

    chunks = load_chunks(args.pdf)
    print(f"Total chunks: {len(chunks)}\n")

    per_unit = Counter(c.metadata["unit"] for c in chunks)
    print("Chunks per unit:")
    for unit in sorted(per_unit):
        title = next(c.metadata["unit_title"] for c in chunks
                     if c.metadata["unit"] == unit)
        print(f"  Unit {unit} ({title}): {per_unit[unit]}")

    print(f"\nFirst {args.samples} chunks:\n")
    for c in chunks[:args.samples]:
        md = c.metadata
        print(f"[{md['chunk_id']}] Unit {md['unit']} "
              f"| {md['section']} {md['section_title']} | p.{md['page']}")
        preview = c.page_content[:200].replace("\n", " ")
        print(f"    {preview}...\n")
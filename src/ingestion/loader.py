"""
ingestion/loader.py

Loads PDF files, extracts text with layout awareness,
attaches rich metadata, and returns LlamaIndex Documents.
"""

import os
import re
import hashlib
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF
import pdfplumber
from tqdm import tqdm

from llama_index.core import Document


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def _detect_filing_type(filename: str) -> str:
    """Infer SEC filing type from filename."""
    fn = filename.lower()
    if "10-k" in fn or "10k" in fn:
        return "10-K"
    if "10-q" in fn or "10q" in fn:
        return "10-Q"
    if "earnings" in fn or "transcript" in fn:
        return "earnings_transcript"
    if "fomc" in fn or "federal_reserve" in fn:
        return "fomc_minutes"
    return "unknown"


def _extract_year(filename: str) -> Optional[str]:
    """Pull a 4-digit year from the filename if present."""
    match = re.search(r"(20\d{2}|19\d{2})", filename)
    return match.group(1) if match else None


def _extract_ticker(filename: str) -> Optional[str]:
    """
    Naively pull a known ticker from the filename.
    Extend this list as you add more companies.
    """
    known = ["JPM", "V", "GS", "MS", "BAC", "WFC", "C", "AXP", "MA"]
    fn = filename.upper()
    for ticker in known:
        if ticker in fn:
            return ticker
    return None


def _doc_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:10]


# ---------------------------------------------------------------------------
# Section-header detection
# ---------------------------------------------------------------------------

_SECTION_PATTERNS = [
    re.compile(r"^(ITEM\s+\d+[A-Z]?\.?\s+.{3,60})$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^([A-Z][A-Z\s\-]{4,50})$", re.MULTILINE),   # ALL-CAPS headings
]

def _detect_section(text_block: str) -> Optional[str]:
    for pattern in _SECTION_PATTERNS:
        m = pattern.search(text_block[:200])   # only check beginning of block
        if m:
            return m.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def load_pdfs(
    data_dir: str = "data/raw",
    verbose: bool = True,
) -> List[Document]:
    """
    Walk data_dir, load every PDF, and return a flat list of
    LlamaIndex Documents — one per page — with rich metadata.
    """
    data_path = Path(data_dir)
    pdf_files = sorted(data_path.glob("**/*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {data_dir}")

    documents: List[Document] = []

    for pdf_path in tqdm(pdf_files, desc="Loading PDFs", disable=not verbose):
        filename = pdf_path.name
        filing_type = _detect_filing_type(filename)
        year = _extract_year(filename)
        ticker = _extract_ticker(filename)

        try:
            docs = _load_single_pdf(pdf_path, filing_type, year, ticker)
            documents.extend(docs)
        except Exception as e:
            print(f"  [WARN] Failed to load {filename}: {e}")
            continue

    print(f"\nLoaded {len(documents)} pages from {len(pdf_files)} PDFs.")
    return documents


def _load_single_pdf(
    pdf_path: Path,
    filing_type: str,
    year: Optional[str],
    ticker: Optional[str],
) -> List[Document]:
    """
    Extract text page-by-page using PyMuPDF.
    Falls back to pdfplumber for pages that return empty text
    (common in scanned/image-heavy filings).
    """
    docs = []
    filename = pdf_path.name

    # Primary: PyMuPDF (fast, layout-aware)
    with fitz.open(str(pdf_path)) as pdf:
        total_pages = len(pdf)

        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            # Fallback: pdfplumber (better for tables)
            if len(text) < 50:
                text = _pdfplumber_page(pdf_path, page_num - 1)

            if not text:
                continue   # skip empty/image pages

            # Clean up common PDF artifacts
            text = _clean_text(text)

            section = _detect_section(text)

            metadata = {
                "filename": filename,
                "filing_type": filing_type,
                "ticker": ticker or "unknown",
                "year": year or "unknown",
                "page": page_num,
                "total_pages": total_pages,
                "section": section or "unknown",
                "doc_hash": _doc_hash(text),
            }

            docs.append(
                Document(
                    text=text,
                    metadata=metadata,
                    id_=f"{filename}_p{page_num}",
                )
            )

    return docs


def _pdfplumber_page(pdf_path: Path, page_index: int) -> str:
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            page = pdf.pages[page_index]
            return page.extract_text() or ""
    except Exception:
        return ""


def _clean_text(text: str) -> str:
    """Remove common PDF noise."""
    # Collapse excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Remove page numbers standing alone
    text = re.sub(r"^\s*\d{1,3}\s*$", "", text, flags=re.MULTILINE)
    return text.strip()

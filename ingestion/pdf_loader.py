"""
PDF -> text extraction (Ingestion Pipeline, step 1).

Reads an uploaded PDF page by page and returns structured page data
(text + page number + source filename) for the next ingestion step
(chunking, Phase 3+).

Out of scope for this module: OCR. A scanned/image-only PDF page will
simply come back with empty text — that's expected and handled, not a bug.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

from pypdf import PdfReader


class PDFLoadError(ValueError):
    """Raised when a file cannot be read as a valid, non-corrupted PDF."""


@dataclass
class PageData:
    """One extracted page, ready for chunking."""

    text: str
    page_number: int  # 1-indexed, matches how a human would cite the page
    source_filename: str


def extract_pages(file_path: str) -> List[PageData]:
    """
    Extract text from every page of a PDF.

    Args:
        file_path: path to a .pdf file.

    Returns:
        One PageData per page, in order (page_number starting at 1).
        A page with no extractable text (blank page, or a scanned/image
        page — OCR is not implemented) comes back with text="" rather
        than being skipped, so page numbering stays intact.

    Raises:
        FileNotFoundError: file_path does not point to an existing file.
        PDFLoadError: the file is not a .pdf, or is corrupted/unreadable
            (including password-protected PDFs, which are unsupported).
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    if not path.is_file():
        raise PDFLoadError(f"Path is not a file: {file_path}")
    if path.suffix.lower() != ".pdf":
        raise PDFLoadError(f"Not a PDF file (expected .pdf extension): {file_path}")

    try:
        reader = PdfReader(str(path))
    except Exception as e:
        raise PDFLoadError(f"Corrupted or unreadable PDF: {file_path} ({e})") from e

    if reader.is_encrypted:
        try:
            # Some PDFs are "encrypted" with an empty owner password and
            # can still be opened; anything else, we treat as unsupported.
            reader.decrypt("")
        except Exception as e:
            raise PDFLoadError(
                f"PDF is password-protected and cannot be read: {file_path}"
            ) from e

    try:
        raw_pages = reader.pages
        page_count = len(raw_pages)
    except Exception as e:
        raise PDFLoadError(f"Corrupted or unreadable PDF: {file_path} ({e})") from e

    pages: List[PageData] = []
    for i in range(page_count):
        try:
            text = raw_pages[i].extract_text() or ""
        except Exception:
            # A single unreadable page shouldn't fail the whole document —
            # treat it like a page with no extractable text.
            text = ""
        pages.append(
            PageData(text=text.strip(), page_number=i + 1, source_filename=path.name)
        )

    return pages


def _preview(text: str, length: int = 120) -> str:
    """Short, single-line preview of a page's text for the verification script."""
    collapsed = " ".join(text.split())
    if not collapsed:
        return "(empty — no extractable text, e.g. a blank or scanned/image page)"
    if len(collapsed) <= length:
        return collapsed
    return collapsed[:length] + "..."


def main() -> None:
    """
    Manual verification script (Phase 2 requirement #7).

    Usage:
        python ingestion/pdf_loader.py path/to/file.pdf
    """
    import sys

    try:
        # Extracted text (e.g. Hebrew course material, or stray emoji/symbols)
        # can contain characters the Windows console's active code page can't
        # encode, which would otherwise crash the print() calls below.
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass  # sys.stdout has no reconfigure() on some environments; best effort.

    if len(sys.argv) != 2:
        print("Usage: python ingestion/pdf_loader.py <path-to-pdf>")
        sys.exit(1)

    file_path = sys.argv[1]

    try:
        pages = extract_pages(file_path)
    except (FileNotFoundError, PDFLoadError) as e:
        print(f"Failed to load PDF: {e}")
        sys.exit(1)

    filename = pages[0].source_filename if pages else Path(file_path).name
    print(f"File: {filename}")
    print(f"Total pages: {len(pages)}\n")

    for page in pages:
        print(f"Page {page.page_number}: {len(page.text)} characters extracted")
        print(f"  Preview: {_preview(page.text)}\n")


if __name__ == "__main__":
    main()

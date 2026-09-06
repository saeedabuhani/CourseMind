"""
Text -> chunks (Ingestion Pipeline, step 2).

Splits the PageData objects produced by ingestion.pdf_loader into
overlapping, token-sized chunks suitable for embedding and RAG retrieval —
while keeping every chunk traceable back to its exact source file and page
number. That page-number link is never allowed to break; it's what Source
Tracking (citing "file X, page Y" in an answer) depends on later.

Chunking strategy
------------------
A chunk never spans two pages — each page is chunked independently, then
the per-page chunks are concatenated. This keeps the chunk -> page_number
mapping exact, at the cost of (rarely) creating a slightly-undersized final
chunk at a page boundary — an acceptable MVP tradeoff given how central
Source Tracking is to this product.

Within a page, splitting is done on whitespace-separated words rather than
raw token slicing. Token counts (via tiktoken) are used to decide how many
words fit in a chunk, but a chunk boundary always falls between two words.
This matters for Hebrew (and any non-ASCII text): slicing a token id
sequence directly can cut a multi-byte character in half and produce
corrupted text, since a token is a raw byte fragment, not necessarily a
whole character. Splitting on word boundaries avoids that entirely while
still respecting a token-based size budget.

Token counting is best-effort: tiktoken's encoder data is fetched over the
network on first use. If that fetch fails (no internet, blocked/expired
certs — exactly the kind of thing that must not break a live demo), this
module falls back to a simple length-based approximation instead of
crashing. See _get_encoder() below.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import warnings

import tiktoken

if __package__ in (None, ""):
    # Allows `python ingestion/chunking.py ...` to work directly (no -m),
    # matching how ingestion/pdf_loader.py is run in Phase 2.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ingestion.pdf_loader import PageData
else:
    from .pdf_loader import PageData

# Defaults for the MVP, defined once here rather than hardcoded wherever
# chunking happens — callers can override them via ChunkingConfig instead.
DEFAULT_CHUNK_SIZE_TOKENS = 500
DEFAULT_CHUNK_OVERLAP_TOKENS = 50
DEFAULT_ENCODING_NAME = "cl100k_base"  # matches GPT-4o family and text-embedding-3-*


@dataclass
class ChunkingConfig:
    """Tunable chunking parameters. Pass a custom instance to chunk_pages()."""

    chunk_size_tokens: int = DEFAULT_CHUNK_SIZE_TOKENS
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS
    encoding_name: str = DEFAULT_ENCODING_NAME

    def __post_init__(self) -> None:
        if self.chunk_size_tokens <= 0:
            raise ValueError("chunk_size_tokens must be positive")
        if self.chunk_overlap_tokens < 0:
            raise ValueError("chunk_overlap_tokens cannot be negative")
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError(
                "chunk_overlap_tokens must be smaller than chunk_size_tokens"
            )


class _ApproxEncoder:
    """
    Offline fallback used only when tiktoken's encoder data can't be
    fetched. Approximates 1 token ~= 4 characters (a common rule of thumb
    for English; still a reasonable, conservative proxy for Hebrew, since
    it tends to overestimate rather than underestimate token count). Only
    .encode() is used by this module (for length, not real tokens), so
    that's all this fallback needs to provide.
    """

    def encode(self, text: str) -> list:
        return [0] * max(1, (len(text) + 3) // 4)


def _get_encoder(encoding_name: str):
    """Real tiktoken encoder when available, else the offline approximation."""
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception as e:
        warnings.warn(
            f"Could not load tiktoken encoding '{encoding_name}' ({e}); "
            "falling back to an approximate, offline token-count estimate.",
            RuntimeWarning,
        )
        return _ApproxEncoder()


@dataclass
class ChunkData:
    """One chunk, ready for embedding, with full source metadata attached."""

    text: str
    source_filename: str
    page_number: int
    chunk_index: int  # 0-indexed position of this chunk within the whole document
    document_id: str
    token_count: int
    character_count: int


def chunk_pages(
    pages: List[PageData],
    document_id: Optional[str] = None,
    config: Optional[ChunkingConfig] = None,
) -> List[ChunkData]:
    """
    Split a document's extracted pages into overlapping, metadata-tagged chunks.

    Args:
        pages: PageData list from ingestion.pdf_loader.extract_pages().
        document_id: identifier shared by every chunk from this document.
            Defaults to the source filename (taken from the first page)
            when not given.
        config: chunking parameters. Defaults to ChunkingConfig() (500
            token chunks, 50 token overlap).

    Returns:
        ChunkData list in document order. Pages with no text (empty,
        blank, or scanned/image pages) contribute zero chunks — they are
        skipped, not an error.
    """
    config = config or ChunkingConfig()
    if document_id is None:
        document_id = pages[0].source_filename if pages else "unknown"

    encoding = _get_encoder(config.encoding_name)

    chunks: List[ChunkData] = []
    chunk_index = 0

    for page in pages:
        if not page.text:
            continue

        for chunk_text in _split_page_into_chunks(page.text, config, encoding):
            chunks.append(
                ChunkData(
                    text=chunk_text,
                    source_filename=page.source_filename,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    document_id=document_id,
                    token_count=len(encoding.encode(chunk_text)),
                    character_count=len(chunk_text),
                )
            )
            chunk_index += 1

    return chunks


def _split_page_into_chunks(text: str, config: ChunkingConfig, encoding) -> List[str]:
    """
    Split one page's text into word-boundary chunks within a token budget.

    Pre-tokenizes each word once, then slides a window over the word list,
    growing it until adding the next word would exceed chunk_size_tokens,
    then stepping back by roughly chunk_overlap_tokens worth of words for
    the next chunk's start. A single word longer than chunk_size_tokens on
    its own (rare — e.g. a long URL) is still emitted whole rather than
    dropped, to avoid data loss.
    """
    words = text.split()
    if not words:
        return []

    chunk_size = config.chunk_size_tokens
    overlap = config.chunk_overlap_tokens
    word_token_counts = [len(encoding.encode(word)) for word in words]

    chunks: List[str] = []
    start = 0
    n = len(words)

    while start < n:
        end = start
        token_total = 0
        while end < n and (token_total + word_token_counts[end] <= chunk_size or end == start):
            token_total += word_token_counts[end]
            end += 1

        chunks.append(" ".join(words[start:end]))

        if end >= n:
            break

        # Step the next chunk's start back by ~overlap tokens worth of words.
        back = end
        back_tokens = 0
        while back > start and back_tokens < overlap:
            back -= 1
            back_tokens += word_token_counts[back]
        start = back if back > start else end

    return chunks


def main() -> None:
    """
    Manual verification script (Phase 3 requirement #9).

    Loads a real PDF through the Phase 2 pdf_loader, chunks it, and prints
    the total chunk count plus a handful of sample chunks with metadata.

    Usage:
        python ingestion/chunking.py path/to/file.pdf [num_samples]
    """
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    if len(sys.argv) not in (2, 3):
        print("Usage: python ingestion/chunking.py <path-to-pdf> [num_samples]")
        sys.exit(1)

    file_path = sys.argv[1]
    num_samples = int(sys.argv[2]) if len(sys.argv) == 3 else 5

    if __package__ in (None, ""):
        from ingestion.pdf_loader import PDFLoadError, extract_pages
    else:
        from .pdf_loader import PDFLoadError, extract_pages

    try:
        pages = extract_pages(file_path)
    except (FileNotFoundError, PDFLoadError) as e:
        print(f"Failed to load PDF: {e}")
        sys.exit(1)

    chunks = chunk_pages(pages)

    print(f"Total pages: {len(pages)}")
    print(f"Total chunks: {len(chunks)}\n")

    for i, chunk in enumerate(chunks[:num_samples], start=1):
        preview = " ".join(chunk.text.split())
        if len(preview) > 200:
            preview = preview[:200] + "..."
        print(f"Chunk {i}")
        print(f"Source: {chunk.source_filename}")
        print(f"Page: {chunk.page_number}")
        print(f"Chunk index: {chunk.chunk_index}")
        print(f"Tokens: {chunk.token_count} | Characters: {chunk.character_count}")
        print(f"Text: {preview}\n")


if __name__ == "__main__":
    main()

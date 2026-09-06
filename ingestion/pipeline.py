"""
Full ingestion pipeline orchestration (Phase 4 wiring + verification).

This module does not contain PDF parsing, chunking, embedding, or vector
store logic itself — it only calls into pdf_loader, chunking, embeddings,
and vectorstore, in order:

    PDF -> extract_pages -> chunk_pages -> embed_chunks -> save_chunks

Keeping this glue in its own file (instead of bolting it onto vectorstore.py
or embeddings.py) is what keeps each of those modules single-purpose.

This module also contains the Phase 3/4 safety check called for in the
Phase 4 spec: Phase 3's token_count can come from a real tokenizer OR an
offline, English-calibrated approximation (see
ingestion.chunking._get_encoder) when tiktoken's vocabulary data can't be
downloaded. That approximation can undercount tokens for Hebrew and other
non-Latin scripts. This module does not redesign or "fix" that — it only
detects when the fallback was used and flags unusually large chunks, so
the person running it knows the token counts are estimates.
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb

if __package__ in (None, ""):
    # Allows `python ingestion/pipeline.py ...` to work directly (no -m),
    # matching how the other ingestion/ modules are run.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ingestion.chunking import ChunkData, chunk_pages
    from ingestion.embeddings import EmbeddingError, embed_chunks
    from ingestion.pdf_loader import PageData, PDFLoadError, extract_pages
    from ingestion.vectorstore import (
        VectorStoreError,
        count,
        get_by_document_id,
        get_client,
        save_chunks,
    )
else:
    from .chunking import ChunkData, chunk_pages
    from .embeddings import EmbeddingError, embed_chunks
    from .pdf_loader import PageData, PDFLoadError, extract_pages
    from .vectorstore import (
        VectorStoreError,
        count,
        get_by_document_id,
        get_client,
        save_chunks,
    )

# A conservative character-count ceiling used only to flag chunks worth a
# second look. OpenAI's embedding models accept far more (thousands of
# tokens) per input, so this is not an API limit — it exists purely
# because the Phase 3 fallback token estimate can be wrong, and a chunk
# this large would be surprising regardless of which tokenizer produced
# its token_count.
SUSPICIOUS_CHUNK_CHARACTERS = 6000


def _chunk_with_fallback_notice(pages: List[PageData]) -> Dict[str, Any]:
    """
    Run chunk_pages() while detecting whether ingestion.chunking's offline
    token-count fallback fired (via the RuntimeWarning it emits), without
    modifying chunking.py at all.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chunks = chunk_pages(pages)
        fallback_messages = [
            str(w.message) for w in caught if "falling back to an approximate" in str(w.message)
        ]
    return {
        "chunks": chunks,
        "used_fallback": bool(fallback_messages),
        "fallback_message": fallback_messages[0] if fallback_messages else None,
    }


def inspect_chunks_before_embedding(chunks: List[ChunkData]) -> List[str]:
    """
    Flag chunks whose character_count looks large relative to their
    recorded token_count — a possible sign of an underestimated token
    count (see module docstring). Returns human-readable warning strings;
    does not modify or drop any chunk.
    """
    findings = []
    for c in chunks:
        if c.character_count > SUSPICIOUS_CHUNK_CHARACTERS:
            findings.append(
                f"Chunk {c.chunk_index} (page {c.page_number}) is "
                f"{c.character_count} characters but recorded as only "
                f"{c.token_count} tokens. That token count may be an "
                "estimate, not an exact tiktoken count — treat it "
                "accordingly rather than as ground truth."
            )
    return findings


def ingest_document(
    file_path: str,
    client: Optional[chromadb.ClientAPI] = None,
) -> Dict[str, Any]:
    """
    Run the full Phase 4 pipeline for one PDF: extract -> chunk -> embed -> store.

    Args:
        file_path: path to a .pdf file.
        client: an existing Chroma client, or None to open the default
            persistent store (data/vectorstore/).

    Returns:
        {
            "pages": List[PageData],
            "chunks": List[ChunkData],
            "used_fallback_tokenizer": bool,
            "fallback_message": Optional[str],
            "size_warnings": List[str],
            "embeddings": List[List[float]],
            "stored_count": int,
        }

    Raises:
        FileNotFoundError / PDFLoadError: from extract_pages().
        EmbeddingError: from embed_chunks() (missing/invalid API key,
            network failure, or another OpenAI API error).
        VectorStoreError: from save_chunks().
    """
    pages = extract_pages(file_path)

    chunk_result = _chunk_with_fallback_notice(pages)
    chunks = chunk_result["chunks"]
    size_warnings = inspect_chunks_before_embedding(chunks)

    embeddings = embed_chunks(chunks) if chunks else []
    stored_count = save_chunks(chunks, embeddings, client=client) if chunks else 0

    return {
        "pages": pages,
        "chunks": chunks,
        "used_fallback_tokenizer": chunk_result["used_fallback"],
        "fallback_message": chunk_result["fallback_message"],
        "size_warnings": size_warnings,
        "embeddings": embeddings,
        "stored_count": stored_count,
    }


def main() -> None:
    """
    Phase 4 manual verification flow.

    PDF -> extract_pages -> chunk_pages -> embed_chunks -> save_chunks,
    then reopens the vector store as a brand-new client to confirm the
    data actually persisted to disk. Prints only safe information: never
    the API key, never a full embedding vector, never other env values.

    Usage:
        python ingestion/pipeline.py path/to/file.pdf
    """
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    if len(sys.argv) != 2:
        print("Usage: python ingestion/pipeline.py <path-to-pdf>")
        sys.exit(1)

    file_path = sys.argv[1]

    try:
        pages = extract_pages(file_path)
    except (FileNotFoundError, PDFLoadError) as e:
        print(f"Failed to load PDF: {e}")
        sys.exit(1)

    filename = pages[0].source_filename if pages else Path(file_path).name
    print(f"Source filename: {filename}")
    print(f"Pages: {len(pages)}")

    chunk_result = _chunk_with_fallback_notice(pages)
    chunks = chunk_result["chunks"]
    print(f"Chunks: {len(chunks)}")

    if chunk_result["used_fallback"]:
        print(f"\nWARNING (from chunking): {chunk_result['fallback_message']}")
        print(
            "NOTE: tiktoken's real tokenizer could not be loaded. Chunk "
            "token_count values are an OFFLINE APPROXIMATION calibrated for "
            "English and may undercount Hebrew/non-Latin text. Treat them "
            "as estimates, not exact counts.\n"
        )

    for w in inspect_chunks_before_embedding(chunks):
        print(f"WARNING: {w}")

    if not chunks:
        print("\nNo extractable text in this PDF — nothing to embed or store.")
        return

    try:
        embeddings = embed_chunks(chunks)
    except EmbeddingError as e:
        print(f"\nFailed to generate embeddings: {e}")
        sys.exit(1)

    dimension = len(embeddings[0]) if embeddings else 0
    print(f"Embedding dimension: {dimension}")

    try:
        write_client = get_client()
        stored = save_chunks(chunks, embeddings, client=write_client)
    except VectorStoreError as e:
        print(f"\nFailed to store chunks: {e}")
        sys.exit(1)

    print(f"Stored records: {stored}")

    sample = chunks[0]
    print("\nSample stored metadata (chunk 0):")
    print(f"  source_filename: {sample.source_filename}")
    print(f"  page_number: {sample.page_number}")
    print(f"  chunk_index: {sample.chunk_index}")
    print(f"  document_id: {sample.document_id}")
    print(f"  token_count: {sample.token_count}")
    print(f"  character_count: {sample.character_count}")

    # --- Persistence verification ---
    # A brand-new client instance pointed at the same on-disk directory
    # simulates a fresh process (e.g. the next `streamlit run`) reopening
    # the store, rather than reusing the client that just wrote the data.
    fresh_client = get_client()
    reloaded = get_by_document_id(sample.document_id, client=fresh_client)
    reloaded_count = len(reloaded.get("ids", []))
    total_count = count(client=fresh_client)

    print("\nPersistence check (reopened the store as a new client):")
    print(f"  Records for this document after reload: {reloaded_count}")
    print(f"  Total records in the collection: {total_count}")


if __name__ == "__main__":
    main()

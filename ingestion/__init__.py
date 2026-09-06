"""
CourseMind ingestion package.

This is NOT an agent tool — it is a fixed pipeline that runs automatically
whenever a student uploads a file, preparing data before the agent ever
gets involved: PDF -> text -> chunks (+ source metadata) -> embeddings ->
vector store.
"""

try:
    # Some Windows antivirus products (AVG, Kaspersky, etc.) and corporate
    # proxies transparently intercept HTTPS ("SSL scanning") and re-sign
    # traffic with their own locally-installed root certificate. Python's
    # default CA bundle (certifi) doesn't know about that root, so calls to
    # tiktoken's encoder download and the OpenAI API fail with
    # SSLCertVerificationError even though the connection itself is fine.
    # truststore uses the OS's own certificate store instead (which already
    # trusts that locally-installed root), fixing this with no configuration.
    # Purely an availability improvement — if it's not installed, everything
    # still works on a normal network; it just won't survive this specific
    # kind of TLS interception.
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

from .pdf_loader import PageData, PDFLoadError, extract_pages
from .chunking import ChunkData, ChunkingConfig, chunk_pages
from .embeddings import EmbeddingError, embed_chunks, embed_texts
from .vectorstore import (
    VectorStoreError,
    get_client,
    get_document_chunks,
    make_chunk_id,
    query,
    save_chunks,
)
from .pipeline import ingest_document

__all__ = [
    "extract_pages",
    "PageData",
    "PDFLoadError",
    "chunk_pages",
    "ChunkData",
    "ChunkingConfig",
    "embed_chunks",
    "embed_texts",
    "EmbeddingError",
    "save_chunks",
    "get_client",
    "make_chunk_id",
    "query",
    "get_document_chunks",
    "VectorStoreError",
    "ingest_document",
]

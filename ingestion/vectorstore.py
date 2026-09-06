"""
Vector store save/query (Ingestion Pipeline, step 4).

Wraps a local, persistent ChromaDB collection under data/vectorstore/.
Not an agent tool — like the rest of ingestion/, this is infrastructure
the pipeline uses after embedding a document's chunks, before the agent
is ever involved.

Storage, the reads needed to verify storage, and raw semantic search
(query()) live here. Turning a raw Chroma query response into a ranked,
metadata-rich result (and deciding what counts as "relevant enough") is
the RAG Retrieval Tool's job (agent/tools/rag_retrieval_tool.py) — this
module only talks to Chroma, nothing about scoring/thresholds/answers.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb

from .chunking import ChunkData

# Persisted on disk under CourseMind/data/vectorstore/ (already git-ignored;
# see .gitignore and the .gitkeep placeholder there).
DEFAULT_PERSIST_DIR = str(Path(__file__).resolve().parent.parent / "data" / "vectorstore")

# Chroma requires collection names to be 3-512 chars from [a-zA-Z0-9._-],
# starting/ending alphanumeric — this name satisfies that.
DEFAULT_COLLECTION_NAME = "coursemind_chunks"


class VectorStoreError(RuntimeError):
    """Raised when the vector store can't be initialized, read, or written to."""


def make_chunk_id(chunk: ChunkData) -> str:
    """
    Deterministic ID for a chunk: same document_id + page + chunk position
    always produces the same ID. This is what lets save_chunks() upsert
    instead of duplicate — re-ingesting the same document overwrites its
    existing records rather than piling up copies.
    """
    return f"{chunk.document_id}::p{chunk.page_number}::c{chunk.chunk_index}"


def get_client(persist_directory: str = DEFAULT_PERSIST_DIR) -> chromadb.ClientAPI:
    """Open (creating if needed) the persistent Chroma store on disk."""
    try:
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=persist_directory)
    except Exception as e:
        raise VectorStoreError(
            f"Could not initialize the vector store at '{persist_directory}': {e}"
        ) from e


def get_collection(
    client: chromadb.ClientAPI, collection_name: str = DEFAULT_COLLECTION_NAME
):
    try:
        return client.get_or_create_collection(name=collection_name)
    except Exception as e:
        raise VectorStoreError(
            f"Could not access collection '{collection_name}': {e}"
        ) from e


def _chunk_metadata(chunk: ChunkData) -> Dict[str, Any]:
    """Every field Source Tracking / later phases will need, as Chroma-safe scalars."""
    return {
        "source_filename": chunk.source_filename,
        "page_number": chunk.page_number,
        "chunk_index": chunk.chunk_index,
        "document_id": chunk.document_id,
        "token_count": chunk.token_count,
        "character_count": chunk.character_count,
    }


def save_chunks(
    chunks: List[ChunkData],
    embeddings: List[List[float]],
    client: Optional[chromadb.ClientAPI] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> int:
    """
    Upsert chunks + their embeddings into the vector store.

    Args:
        chunks: ChunkData list (e.g. from ingestion.chunking.chunk_pages).
        embeddings: one vector per chunk, same order and length as `chunks`
            (e.g. from ingestion.embeddings.embed_chunks).
        client: an existing Chroma client, or None to open the default
            persistent store.
        collection_name: which collection to write into.

    Returns:
        Number of records written. 0 (no-op) if `chunks` is empty.

    Raises:
        VectorStoreError: chunks/embeddings length mismatch, or Chroma
            failed to initialize/write.
    """
    if not chunks:
        return 0
    if len(chunks) != len(embeddings):
        raise VectorStoreError(
            f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
            "counts do not match"
        )

    client = client or get_client()
    collection = get_collection(client, collection_name)

    ids = [make_chunk_id(c) for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [_chunk_metadata(c) for c in chunks]

    try:
        collection.upsert(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )
    except Exception as e:
        raise VectorStoreError(
            f"Failed to write {len(chunks)} chunk(s) to the vector store: {e}"
        ) from e

    return len(chunks)


def get_by_document_id(
    document_id: str,
    client: Optional[chromadb.ClientAPI] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    include_embeddings: bool = False,
) -> Dict[str, Any]:
    """
    Fetch every stored chunk for one document (used later by the Summary
    Tool, which needs all chunks rather than a top-k similarity search;
    used here in Phase 4 to verify that storage actually persisted).
    """
    client = client or get_client()
    collection = get_collection(client, collection_name)
    include = ["documents", "metadatas"] + (["embeddings"] if include_embeddings else [])
    try:
        return collection.get(where={"document_id": document_id}, include=include)
    except Exception as e:
        raise VectorStoreError(f"Vector store fetch failed: {e}") from e


def get_document_chunks(
    document_id: str,
    client: Optional[chromadb.ClientAPI] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> List[Dict[str, Any]]:
    """
    Every stored chunk for one document, as plain per-chunk dicts sorted
    into original document order (page_number, then chunk_index).

    This is deliberately NOT top-k similarity search — it exists for
    consumers (the Summary service) that need the *entire* document
    rather than the most relevant few chunks. It's a thin convenience
    wrapper over get_by_document_id(): no new Chroma access logic, just
    reshaping that call's raw batched-lists response into one sorted
    list of {"text", "source_filename", "page_number", "chunk_index",
    "document_id", "token_count", "character_count"} dicts.

    Returns [] if the document_id has no stored chunks (not an error —
    callers decide whether that's a problem for their use case).
    """
    raw = get_by_document_id(document_id, client=client, collection_name=collection_name)
    documents = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []

    records = [
        {**meta, "text": text} for text, meta in zip(documents, metadatas)
    ]
    records.sort(key=lambda r: (r["page_number"], r["chunk_index"]))
    return records


def query(
    query_embedding: List[float],
    top_k: int = 4,
    document_id: Optional[str] = None,
    client: Optional[chromadb.ClientAPI] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Dict[str, Any]:
    """
    Raw semantic nearest-neighbor search — the RAG Retrieval Tool's
    foundation. Returns Chroma's native response shape (each field is a
    list-of-lists, one inner list per query embedding; here there's
    always exactly one query embedding, so callers read index [0]).

    This collection was created with Chroma's default HNSW space, "l2"
    (squared Euclidean distance): LOWER values mean MORE similar, and
    0.0 means identical vectors. It is NOT a 0-1 similarity score. The
    RAG Retrieval Tool is responsible for interpreting these distances
    (e.g. applying a relevance threshold) — this function just returns
    them as Chroma provides them.

    Args:
        query_embedding: an already-computed embedding vector for the
            search query (this module never calls an embedding API).
        top_k: how many nearest neighbors to return.
        document_id: if given, restricts the search to chunks whose
            document_id metadata matches exactly (keeps a query scoped
            to one uploaded document instead of searching everything
            ever indexed).
        client: an existing Chroma client, or None to open the default
            persistent store.

    Raises:
        VectorStoreError: the query itself failed. An empty collection
            or zero matches is NOT an error — it comes back as empty
            result lists.
    """
    client = client or get_client()
    collection = get_collection(client, collection_name)
    where = {"document_id": document_id} if document_id else None
    try:
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        raise VectorStoreError(f"Vector store query failed: {e}") from e


def count(
    client: Optional[chromadb.ClientAPI] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> int:
    """Total number of records currently stored in the collection."""
    client = client or get_client()
    collection = get_collection(client, collection_name)
    try:
        return collection.count()
    except Exception as e:
        raise VectorStoreError(f"Vector store count failed: {e}") from e

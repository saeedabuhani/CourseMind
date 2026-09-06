"""
Chunks -> embeddings (Ingestion Pipeline, step 3).

Calls the OpenAI embeddings API to turn chunk text into vectors, ready to
be stored in the vector store (ingestion.vectorstore) alongside their
source metadata. Not an agent tool — like the rest of ingestion/, this
runs automatically as part of the pipeline that prepares a document
before the agent is ever involved.

Configuration comes entirely from environment variables (loaded from a
local .env via python-dotenv) — OPENAI_API_KEY and, optionally,
OPENAI_EMBEDDING_MODEL. Nothing here ever hardcodes a key or prints one:
error messages describe *what* went wrong, never the key's value.

Provider-specific by design: this module knows nothing about ChromaDB or
the vector store — that separation is what lets the provider be swapped
without touching ingestion/vectorstore.py or ingestion/pipeline.py.
"""

import os
from typing import List

from dotenv import load_dotenv
from openai import APIConnectionError, APIError, AuthenticationError, OpenAI

from .chunking import ChunkData

load_dotenv()  # populate os.environ from a local .env file, if one exists

# Small, cheap, and good enough for the MVP's RAG use case; overridable
# via OPENAI_EMBEDDING_MODEL so a caller can switch models without a code
# change. Must match .env.example's documented default.
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_BATCH_SIZE = 100  # chunks per API request


class EmbeddingError(RuntimeError):
    """
    Raised for missing configuration or a failed OpenAI embeddings call.

    Messages here describe the failure only — they never include the API
    key's value, so this is safe to log or print as-is.
    """


def _get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EmbeddingError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return OpenAI(api_key=api_key)


def _get_model_name() -> str:
    return os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def embed_texts(texts: List[str], batch_size: int = DEFAULT_BATCH_SIZE) -> List[List[float]]:
    """
    Embed a list of raw strings, in order, batching requests to the API.

    Args:
        texts: strings to embed. An empty list returns [] without calling
            the API (nothing to embed, not an error).
        batch_size: how many strings to send per API request.

    Returns:
        One embedding vector per input string, same order as `texts`.

    Raises:
        EmbeddingError: OPENAI_API_KEY is missing, the key is rejected,
            the API can't be reached, or the API otherwise returns an
            error. The original exception is chained (`from e`) for
            debugging, but the EmbeddingError message itself never
            contains the API key.
    """
    if not texts:
        return []

    client = _get_client()
    model = _get_model_name()
    vectors: List[List[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        try:
            response = client.embeddings.create(model=model, input=batch)
        except AuthenticationError as e:
            raise EmbeddingError(
                "OpenAI rejected the API key (authentication failed). "
                "Check OPENAI_API_KEY in your .env file."
            ) from e
        except APIConnectionError as e:
            raise EmbeddingError(
                "Could not reach the OpenAI API (network/connection error)."
            ) from e
        except APIError as e:
            raise EmbeddingError(f"OpenAI embeddings request failed: {e}") from e

        vectors.extend(item.embedding for item in response.data)

    return vectors


def embed_chunks(chunks: List[ChunkData], batch_size: int = DEFAULT_BATCH_SIZE) -> List[List[float]]:
    """
    Embed ChunkData objects, preserving order: embeddings[i] corresponds
    to chunks[i]. See embed_texts() for error behavior.
    """
    return embed_texts([c.text for c in chunks], batch_size=batch_size)

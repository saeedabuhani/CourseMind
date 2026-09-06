"""
RAG Retrieval Tool — Q&A flow (Phase 5: retrieval only, no answer generation).

Finds evidence for a natural-language question by:
1. embedding the query with the same OpenAI embedding model used at
   ingestion time (ingestion.embeddings.embed_texts) — no embedding logic
   is duplicated here.
2. running a semantic nearest-neighbor search against the existing Chroma
   collection (ingestion.vectorstore.query) — no Chroma setup/storage
   logic is duplicated here either.
3. converting the raw Chroma response into ranked RetrievedChunk objects,
   dropping anything not relevant enough, while preserving every Source
   Tracking field.

Deliberately NOT done here: generating a final answer. This module finds
and ranks evidence; turning that evidence into an answer for the student
is the CrewAI Agent's job (Phase 6+), which will call retrieve() as its
Tool.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

if __package__ in (None, ""):
    # Allows `python agent/tools/rag_retrieval_tool.py ...` to work
    # directly (no -m), matching how the ingestion/ modules are run.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from ingestion.embeddings import embed_texts
    from ingestion.vectorstore import query as vectorstore_query
else:
    from ingestion.embeddings import embed_texts
    from ingestion.vectorstore import query as vectorstore_query

# How many chunks to retrieve when the caller doesn't specify. Kept as one
# named constant (not hardcoded at each call site) so it's easy to tune.
DEFAULT_TOP_K = 4

# The Chroma collection ingestion/vectorstore.py writes to was created
# with Chroma's default HNSW space, "l2" (squared Euclidean distance):
# LOWER means MORE similar, 0.0 means identical. This is NOT a 0-1
# similarity score, so this threshold lives on that same L2 scale.
#
# Chosen empirically for Phase 5 (see verification run / Phase 5 summary),
# against the real 21-page Hebrew JavaScript course PDF used in prior
# phases: two genuinely-relevant queries ("What is a variable in
# JavaScript", "How do you define a function in JavaScript") measured
# distances of ~0.83-0.99 to their top-4 matches, while two deliberately
# unrelated queries (a hummus recipe; the distance to Mars) measured
# ~1.35-1.89. 1.1 sits in the gap between those two observed clusters.
#
# This is a small-sample, MVP-appropriate choice, not a validated
# universal constant — it will need revisiting once more/larger documents
# are indexed, or if it starts admitting irrelevant chunks or rejecting
# relevant ones in practice.
DEFAULT_MAX_DISTANCE = 1.1


class RetrievalError(RuntimeError):
    """
    Raised for retrieval-layer usage errors (empty query, invalid top_k).

    Never includes any API key or secret. Failures from lower layers
    surface as their own exception types instead of being wrapped here:
    EmbeddingError (ingestion.embeddings) for embedding problems,
    VectorStoreError (ingestion.vectorstore) for Chroma problems.
    """


@dataclass
class RetrievedChunk:
    """One piece of evidence, ranked, with full Source Tracking metadata."""

    text: str
    source_filename: str
    page_number: int
    chunk_index: int
    document_id: str
    token_count: int
    character_count: int
    distance: float  # L2 distance to the query — lower is more similar


@dataclass
class RetrievalResult:
    """
    The outcome of one retrieve() call.

    `found` is False (chunks == []) whenever there is nothing relevant
    enough to answer from — the caller (eventually, the Agent) must treat
    that as "say I don't know", never invent an answer from thin air.
    """

    query: str
    document_id: Optional[str]
    chunks: List[RetrievedChunk] = field(default_factory=list)
    found: bool = False


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    document_id: Optional[str] = None,
    max_distance: Optional[float] = DEFAULT_MAX_DISTANCE,
) -> RetrievalResult:
    """
    Find the top_k most relevant chunks for `query`, optionally scoped to
    one document_id, and rank/filter them by relevance.

    This function only finds evidence — it never calls a chat/generation
    model and never produces a final answer.

    Args:
        query: the student's natural-language question. Must be a
            non-empty, non-whitespace-only string.
        top_k: how many nearest-neighbor chunks to consider. Must be a
            positive integer.
        document_id: if given, restricts the search to chunks from that
            document only — this is what lets a student's question stay
            scoped to the currently selected/uploaded document instead of
            searching every document ever indexed.
        max_distance: chunks with a distance greater than this are
            dropped as not relevant enough (see DEFAULT_MAX_DISTANCE for
            how this value was chosen). Pass None to disable filtering
            and return the raw top_k regardless of distance.

    Returns:
        RetrievalResult. If the vector store is empty, has nothing for
        the given document_id, or every candidate is filtered out by
        max_distance, this comes back with chunks=[] and found=False —
        that is the "no useful evidence found" signal, not an error.

    Raises:
        RetrievalError: query is empty/whitespace, or top_k is not a
            positive integer.
        EmbeddingError: from ingestion.embeddings — missing/invalid
            OPENAI_API_KEY, network failure, or another OpenAI API error.
        VectorStoreError: from ingestion.vectorstore — Chroma itself
            failed to initialize or query (an empty store is NOT this;
            it just yields zero results).
    """
    if not query or not query.strip():
        raise RetrievalError("query must be a non-empty string")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise RetrievalError(f"top_k must be a positive integer, got {top_k!r}")

    [query_embedding] = embed_texts([query])

    raw = vectorstore_query(query_embedding, top_k=top_k, document_id=document_id)

    ids = raw.get("ids") or [[]]
    documents = raw.get("documents") or [[]]
    metadatas = raw.get("metadatas") or [[]]
    distances = raw.get("distances") or [[]]

    chunks: List[RetrievedChunk] = []
    for doc_text, meta, distance in zip(documents[0], metadatas[0], distances[0]):
        if max_distance is not None and distance > max_distance:
            continue
        chunks.append(
            RetrievedChunk(
                text=doc_text,
                source_filename=meta["source_filename"],
                page_number=meta["page_number"],
                chunk_index=meta["chunk_index"],
                document_id=meta["document_id"],
                token_count=meta["token_count"],
                character_count=meta["character_count"],
                distance=distance,
            )
        )

    return RetrievalResult(
        query=query, document_id=document_id, chunks=chunks, found=bool(chunks)
    )


class _CourseMaterialSearchInput(BaseModel):
    query: str = Field(
        ...,
        description=(
            "A focused search query about the course material — the key "
            "concept or the student's question itself, e.g. "
            "'What is a variable in JavaScript?'"
        ),
    )


class CourseMaterialSearchTool(BaseTool):
    """
    CrewAI-facing adapter around retrieve() (above). Duplicates none of
    Phase 5's logic — no embedding calls, no Chroma queries, no threshold
    decisions happen here. This class only:
      1. exposes retrieve() to a CrewAI Agent as a Tool it can decide to
         call, with a description telling it when/how to use it, and
      2. turns the structured RetrievalResult into the short text an LLM
         reads, while remembering the last RetrievalResult (see
         `last_result`) so Phase 7 can still get full Source Tracking
         metadata after the Agent finishes — nothing is discarded, it's
         just not shown to the LLM as raw dataclasses.

    `document_id` is set once when the tool is built for a run (see
    agent/crew.py's build_crew()) — it is NOT part of the tool's input
    schema, so the LLM can neither omit nor hallucinate a different one.
    This is what keeps Phase 5's document isolation intact under agent
    control.
    """

    name: str = "search_course_material"
    description: str = (
        "Search the student's uploaded course material for evidence relevant "
        "to a question. ALWAYS use this before answering any question about "
        "the course content — never answer such questions from general "
        "knowledge alone. Input: a focused search query (string). Returns: "
        "relevant excerpts from the material with their source file and page "
        "number, or a clear statement that nothing relevant was found — in "
        "that case, tell the student the material does not cover it instead "
        "of guessing."
    )
    args_schema: Type[BaseModel] = _CourseMaterialSearchInput

    document_id: Optional[str] = None

    _last_result: Optional[RetrievalResult] = PrivateAttr(default=None)

    @property
    def last_result(self) -> Optional[RetrievalResult]:
        """The RetrievalResult from the most recent call, or None if never called."""
        return self._last_result

    def _run(self, query: str) -> str:
        try:
            result = retrieve(query, document_id=self.document_id)
        except Exception as e:
            # Never let a raw exception escape into CrewAI's tool-calling
            # loop with unpredictable results — always hand the agent a
            # clear, safe string it can react to (e.g. tell the student a
            # technical error occurred), consistent with the Grounding
            # Rule's "don't invent, say so" spirit. Never includes a key:
            # EmbeddingError/VectorStoreError/RetrievalError messages are
            # already designed to be safe to surface as-is.
            self._last_result = None
            return f"Course material search failed ({type(e).__name__}): {e}"

        self._last_result = result
        return _format_result_for_agent(result)


def _format_result_for_agent(result: RetrievalResult) -> str:
    """Render a RetrievalResult as the short text a CrewAI Agent reads."""
    if not result.found:
        return (
            "No relevant information was found in the uploaded course "
            "material for this query. Do not invent an answer — tell the "
            "student this specific topic is not covered in their uploaded "
            "material."
        )
    lines = ["Relevant excerpts from the uploaded course material:", ""]
    for i, c in enumerate(result.chunks, start=1):
        lines.append(f"[{i}] Source: {c.source_filename}, page {c.page_number}")
        lines.append(c.text)
        lines.append("")
    return "\n".join(lines)


def _print_result(result: RetrievalResult) -> None:
    print(f"Query: {result.query}")
    if result.document_id:
        print(f"(scoped to document_id: {result.document_id})")
    if not result.found:
        print("Result: NO USEFUL EVIDENCE FOUND (nothing relevant enough)\n")
        return
    for i, c in enumerate(result.chunks, start=1):
        preview = " ".join(c.text.split())
        if len(preview) > 160:
            preview = preview[:160] + "..."
        print(f"Result {i}")
        print(f"  Source: {c.source_filename}")
        print(f"  Page: {c.page_number}")
        print(f"  Chunk index: {c.chunk_index}")
        print(f"  Distance (L2, lower = more similar): {c.distance:.4f}")
        print(f"  Text preview: {preview}")
    print()


def main() -> None:
    """
    Phase 5 manual verification (requirements #14-16): runs four canned
    queries — a factual query, an explanation-style query, a multi-chunk
    query, and a deliberately unrelated query — against an already-indexed
    PDF, plus a document-isolation check using a small synthetic second
    document. Prints only safe information (metadata + distances), never
    an API key or a full embedding vector.

    Usage:
        python agent/tools/rag_retrieval_tool.py
    (Assumes the target PDF has already been indexed via
    `python ingestion/pipeline.py <path>`.)
    """
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    # Queries are in Hebrew to match the indexed PDF's language. This
    # matters: a cross-lingual query (English question against Hebrew
    # course material) was measured during Phase 5 verification to score
    # meaningfully higher L2 distances even for genuinely relevant
    # content — see the Phase 5 summary's "limitations" section. Realistic
    # usage is a student querying in the same language as their material.
    print("=== TEST A: factual query ===")
    _print_result(retrieve("מה זה משתנה בשפת JavaScript?"))

    print("=== TEST B: explanation-style query ===")
    _print_result(retrieve("איך פועלות פונקציות בשפת JavaScript?"))

    print("=== TEST C: multi-chunk query ===")
    _print_result(retrieve("אילו טיפוסי נתונים ואופרטורים יש בשפת JavaScript?", top_k=6))

    print("=== TEST D: unrelated query ===")
    _print_result(retrieve("מה המתכון הכי טוב לעוגת שוקולד?"))


if __name__ == "__main__":
    main()

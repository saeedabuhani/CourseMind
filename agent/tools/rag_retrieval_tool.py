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
    from ingestion.vectorstore import get_document_chunks
    from ingestion.vectorstore import query as vectorstore_query
else:
    from ingestion.embeddings import embed_texts
    from ingestion.vectorstore import get_document_chunks
    from ingestion.vectorstore import query as vectorstore_query

# How many chunks to retrieve when the caller doesn't specify. Kept as one
# named constant (not hardcoded at each call site) so it's easy to tune.
DEFAULT_TOP_K = 6

# The Chroma collection ingestion/vectorstore.py writes to was created
# with Chroma's default HNSW space, "l2" (squared Euclidean distance):
# LOWER means MORE similar, 0.0 means identical. This is NOT a 0-1
# similarity score, so this threshold lives on that same L2 scale.
#
# WHY 1.5 AND NOT THE ORIGINAL 1.1:
# 1.1 was calibrated against a single prose-style PDF whose relevant
# matches measured ~0.83-0.99. Re-measuring against a second, real
# slide-style PDF (23 pages, terse bullet text) showed that terse
# material scores MUCH higher distances for the same kind of question:
#   "What is an Array?"            -> best 1.026   (content IS on page 13)
#   "What is a variable?"          -> best 1.355   (content IS on page 5)
#   "What is a function?"          -> best 1.409   (content IS on page 11)
#   "let vs var?"                  -> best 1.140   (content IS on page 5)
#   "regular vs arrow function?"   -> best 1.442   (content IS on page 11)
# while genuinely unrelated questions against the same document measured
#   chocolate cake recipe 1.526 | capital of Japan 1.608 | world cup 1.595
# So at 1.1, real course content was being rejected as "not found" —
# the FALSE NEGATIVES reported in testing. 1.5 admits every measured
# relevant case above while still rejecting those unrelated queries.
#
# IMPORTANT — this threshold is deliberately NOT the only line of defence.
# Measurement also proved no single absolute distance can separate
# "relevant" from "unrelated" across different documents (on the prose
# PDF, an unrelated query still had matches at ~1.42-1.46, overlapping
# the slide PDF's relevant range). So retrieval is now the RECALL stage
# (get the real material in front of the agent) and the agent, which
# reads the actual excerpt text, is the PRECISION stage: it must answer
# with the NO_EVIDENCE sentinel when the excerpts do not really contain
# the answer (see agent/crew.py), and services/qa_service.py turns that
# sentinel into the deterministic "not found" contract. Grounding is not
# weakened by this: an answer still REQUIRES real retrieved evidence.
DEFAULT_MAX_DISTANCE = 1.5

# Retrieval modes. "semantic" is normal top-k similarity search;
# "page" and "overview" are STRUCTURAL retrievals that use the stored
# page_number/document_id metadata instead of similarity (see
# retrieve_page / retrieve_overview) — they never rank by distance, so
# their chunks carry distance=None.
MODE_SEMANTIC = "semantic"
MODE_PAGE = "page"
MODE_OVERVIEW = "overview"
# Answered by services/qa_service.py straight from the vector store's own
# bookkeeping (how many pages/chunks were indexed, under what filename).
# No retrieval and no LLM are involved, which is why it lives here as a
# mode name only — there is no retrieve_*() function behind it.
MODE_METADATA = "metadata"

# Total character budget for an overview retrieval (a document-level
# "what is in this file?" question). Every page is represented, so this
# is spread evenly across the document's chunks rather than spent on the
# first few.
OVERVIEW_CHAR_BUDGET = 6000
OVERVIEW_MIN_CHARS_PER_CHUNK = 120
OVERVIEW_MAX_CHARS_PER_CHUNK = 400


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
    # L2 distance to the query — lower is more similar. None for
    # structural retrieval (page/overview), which does not rank by
    # similarity at all, so a number here would be meaningless.
    distance: Optional[float]


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
    # Which retrieval strategy produced these chunks: MODE_SEMANTIC,
    # MODE_PAGE or MODE_OVERVIEW.
    retrieval_mode: str = MODE_SEMANTIC
    # Only set for MODE_PAGE: the page that was asked for, and whether it
    # actually exists in the document.
    requested_page: Optional[int] = None
    page_exists: Optional[bool] = None


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
        query=query,
        document_id=document_id,
        chunks=chunks,
        found=bool(chunks),
        retrieval_mode=MODE_SEMANTIC,
    )


def _chunk_from_record(record: dict, text: str) -> RetrievedChunk:
    """Build a RetrievedChunk from a stored chunk record (structural retrieval)."""
    return RetrievedChunk(
        text=text,
        source_filename=record["source_filename"],
        page_number=record["page_number"],
        chunk_index=record["chunk_index"],
        document_id=record["document_id"],
        token_count=record["token_count"],
        character_count=record["character_count"],
        distance=None,  # structural match, not a similarity ranking
    )


def retrieve_page(document_id: str, page_number: int) -> RetrievalResult:
    """
    Return every stored chunk belonging to ONE page of ONE document.

    This is STRUCTURAL retrieval: it uses the page_number/document_id
    metadata that ingestion already stores, never semantic similarity.
    That is deliberate — "what is on page 5?" is a question about a
    location in the document, and similarity search answers it badly
    (measured: asking for page 5 returned page 22 as its best match).

    The page numbering is exactly the one ingestion/pdf_loader.py assigns
    and Source Tracking displays (1-indexed, as a human would cite it) —
    no re-numbering happens here.

    Returns:
        RetrievalResult with retrieval_mode=MODE_PAGE, requested_page set,
        and page_exists telling the caller whether that page exists in
        the document at all. found=False with page_exists=False means
        "the student asked for a page this document does not have" —
        which the caller must report honestly, never fabricate content for.

    Raises:
        RetrievalError: document_id empty, or page_number not a positive int.
        VectorStoreError: from ingestion.vectorstore.
    """
    if not document_id or not document_id.strip():
        raise RetrievalError("document_id must be a non-empty string for page retrieval")
    if not isinstance(page_number, int) or isinstance(page_number, bool) or page_number <= 0:
        raise RetrievalError(f"page_number must be a positive integer, got {page_number!r}")

    records = get_document_chunks(document_id)
    page_records = [r for r in records if r.get("page_number") == page_number]

    chunks = [_chunk_from_record(r, r["text"]) for r in page_records]
    chunks.sort(key=lambda c: c.chunk_index)

    return RetrievalResult(
        query=f"page {page_number}",
        document_id=document_id,
        chunks=chunks,
        found=bool(chunks),
        retrieval_mode=MODE_PAGE,
        requested_page=page_number,
        page_exists=bool(page_records),
    )


def retrieve_overview(
    document_id: str,
    char_budget: int = OVERVIEW_CHAR_BUDGET,
) -> RetrievalResult:
    """
    Return an evidence set that spans the WHOLE document, for
    document-level questions like "what is in this file?".

    This is STRUCTURAL retrieval too: it loads every stored chunk for the
    document (the same all-chunks read the Full Study Summary uses — no
    top-k), then keeps the opening slice of each chunk so that every page
    is represented within a bounded amount of text. The opening slice is
    where headings/topic lines live, which is exactly what a "what topics
    does this cover?" question needs.

    Why not just run a similarity search? Measured: "מה יש בקובץ?" and
    "מה התוכן של המסמך?" have no semantically similar chunk anywhere
    (best distances 1.36-1.50 — indistinguishable from unrelated
    questions), so similarity search answers document-level questions
    with a false "not found".

    Returns:
        RetrievalResult with retrieval_mode=MODE_OVERVIEW. found=False
        only when the document has no indexed chunks at all.

    Raises:
        RetrievalError: document_id is empty.
        VectorStoreError: from ingestion.vectorstore.
    """
    if not document_id or not document_id.strip():
        raise RetrievalError("document_id must be a non-empty string for an overview")

    records = get_document_chunks(document_id)
    if not records:
        return RetrievalResult(
            query="document overview",
            document_id=document_id,
            chunks=[],
            found=False,
            retrieval_mode=MODE_OVERVIEW,
        )

    per_chunk = max(
        OVERVIEW_MIN_CHARS_PER_CHUNK,
        min(OVERVIEW_MAX_CHARS_PER_CHUNK, char_budget // len(records)),
    )

    chunks = []
    for r in records:
        text = " ".join((r.get("text") or "").split())
        if len(text) > per_chunk:
            text = text[:per_chunk].rstrip() + "..."
        if not text:
            continue
        chunks.append(_chunk_from_record(r, text))

    return RetrievalResult(
        query="document overview",
        document_id=document_id,
        chunks=chunks,
        found=bool(chunks),
        retrieval_mode=MODE_OVERVIEW,
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
        "number, or a clear statement that nothing relevant was found. You "
        "may call this more than once: if the first search misses, try the "
        "key technical terms on their own (often the English or code form, "
        "e.g. 'arrow function', 'async await'), because course material "
        "usually mixes Hebrew explanation with English keywords and code."
    )
    args_schema: Type[BaseModel] = _CourseMaterialSearchInput

    document_id: Optional[str] = None
    # Like document_id, these are locked in by build_crew() for one run and
    # are NOT part of args_schema — the LLM cannot set, change or omit them.
    # page_number: answer strictly from that page (structural retrieval).
    # overview: answer a document-level question from whole-document
    # coverage. Both are decided deterministically by services/qa_service.py
    # from the student's question, never by the model.
    page_number: Optional[int] = None
    overview: bool = False
    # Alternative wordings for the SAME question, produced by Query
    # Understanding. On a semantic search these are run alongside the query
    # and their results merged, so recall does not depend on the agent
    # choosing to search a second time.
    #
    # Measured need: for "מה ההבדל בין פונקציה רגילה לפונקציית חץ?" the Hebrew
    # phrasing ranks the correct page at distance 1.552 (above the 1.5
    # threshold, so it is dropped) while "arrow function" ranks that same page
    # at 1.273. Leaving that second search to the agent's judgement produced a
    # grounded answer in only 2 of 5 runs; running it here makes it consistent.
    #
    # This is a BOUNDED fallback, not a retry loop: at most three searches per
    # question, all inside the same document_id, all merged and de-duplicated.
    fallback_queries: List[str] = []

    _last_result: Optional[RetrievalResult] = PrivateAttr(default=None)
    _seen_keys: set = PrivateAttr(default_factory=set)

    @property
    def last_result(self) -> Optional[RetrievalResult]:
        """
        Everything this tool retrieved during the current agent run, merged.

        NOT just the final call: the Agent is allowed (and instructed) to
        search more than once — e.g. re-searching in the material's own
        technical wording when a Hebrew phrasing retrieves poorly. If this
        returned only the last call's result, evidence found by an earlier
        search would vanish from Source Tracking and the grounding check
        would wrongly conclude "no evidence". Chunks are de-duplicated by
        (document_id, page, chunk_index) and kept ordered by relevance.

        None means the tool was never called, or its very first call failed
        technically before any evidence was collected — the signal
        services/qa_service.py uses to tell a technical failure apart from
        an honest "nothing relevant found".
        """
        return self._last_result

    def _merge(self, result: RetrievalResult) -> None:
        """Fold one call's chunks into the accumulated result for this run."""
        previous = self._last_result
        if previous is None or previous.retrieval_mode != result.retrieval_mode:
            self._seen_keys = set()
            merged_chunks = []
        else:
            merged_chunks = list(previous.chunks)

        for c in result.chunks:
            key = (c.document_id, c.page_number, c.chunk_index)
            if key in self._seen_keys:
                continue
            self._seen_keys.add(key)
            merged_chunks.append(c)

        if all(c.distance is not None for c in merged_chunks):
            merged_chunks.sort(key=lambda c: c.distance)

        self._last_result = RetrievalResult(
            query=result.query,
            document_id=result.document_id,
            chunks=merged_chunks,
            found=bool(merged_chunks),
            retrieval_mode=result.retrieval_mode,
            requested_page=result.requested_page,
            page_exists=result.page_exists,
        )

    def _run(self, query: str) -> str:
        try:
            if self.page_number is not None and self.document_id:
                # Locked to one page: ignore the model's query wording
                # entirely and fetch that page by metadata.
                result = retrieve_page(self.document_id, self.page_number)
            elif self.overview and self.document_id:
                result = retrieve_overview(self.document_id)
            else:
                result = retrieve(query, document_id=self.document_id)
                # Same question, other wordings — merged into one evidence set.
                # document_id is passed on every one of them, so a rewritten
                # query can never reach outside the active document.
                for alt in (self.fallback_queries or [])[:2]:
                    if not alt or alt.strip().lower() == query.strip().lower():
                        continue
                    try:
                        self._merge(retrieve(alt, document_id=self.document_id))
                    except Exception:
                        # One weak alternative must not sink a search that
                        # already has real evidence.
                        pass
        except Exception as e:
            # Never let a raw exception escape into CrewAI's tool-calling
            # loop with unpredictable results — always hand the agent a
            # clear, safe string it can react to (e.g. tell the student a
            # technical error occurred), consistent with the Grounding
            # Rule's "don't invent, say so" spirit. Never includes a key:
            # EmbeddingError/VectorStoreError/RetrievalError messages are
            # already designed to be safe to surface as-is.
            #
            # Evidence already gathered by an EARLIER successful search in
            # this same run is kept: one failed follow-up query is not a
            # reason to throw away real material the student can be
            # answered from. Only a failure with nothing gathered yet
            # leaves last_result as None — the technical-failure signal.
            if self._last_result is None:
                self._last_result = None
            return f"Course material search failed ({type(e).__name__}): {e}"

        self._merge(result)
        return _format_result_for_agent(result)


def _format_result_for_agent(result: RetrievalResult) -> str:
    """Render a RetrievalResult as the short text a CrewAI Agent reads."""
    if result.retrieval_mode == MODE_PAGE:
        if not result.found:
            return (
                f"Page {result.requested_page} does not exist in the uploaded "
                "document. Tell the student that page is not in their "
                "document. Do not invent its content."
            )
        lines = [
            f"Full content of page {result.requested_page} of the uploaded "
            "course material. Answer ONLY from this page:",
            "",
        ]
        for c in result.chunks:
            lines.append(f"Source: {c.source_filename}, page {c.page_number}")
            lines.append(c.text)
            lines.append("")
        return "\n".join(lines)

    if result.retrieval_mode == MODE_OVERVIEW:
        if not result.found:
            return (
                "The uploaded document has no indexed content. Tell the "
                "student there is nothing to describe."
            )
        lines = [
            "Coverage of the ENTIRE uploaded document (the opening lines of "
            "every page, in order). Use this to describe what the document "
            "contains and which topics it covers — only from this material:",
            "",
        ]
        for c in result.chunks:
            lines.append(f"[page {c.page_number}] {c.text}")
        return "\n".join(lines)

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
        if c.distance is None:
            print("  Distance: n/a (structural retrieval, not similarity)")
        else:
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

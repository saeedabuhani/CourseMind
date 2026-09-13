"""
Q&A backend service (Phase 7) — the boundary a future UI (Streamlit,
Phase 8+) will call.

Wraps the Phase 6 Agent (agent.run_agent) with:
- a structured response type (QAResponse / SourceReference) instead of
  free text plus a loose dataclass
- STRICT, programmatic grounding enforcement — whether the student gets
  a real answer or the deterministic "not found" message is decided here
  from the structured RetrievalResult, never trusted from the LLM's own
  text
- source deduplication + sorting for a clean, user-facing citation list
- one exception type (QAServiceError) for technical failures, kept
  clearly separate from "no evidence found" (a normal, successful
  QAResponse, not an error)

This module creates no Agents, Tasks, retrieval calls, or embeddings
itself — it only calls agent.run_agent(), Phase 6's public interface.
Duplicating any of that logic here would be exactly what Phase 7's
instructions forbid.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from agent import AgentError, run_agent
from agent.crew import NO_EVIDENCE_SENTINEL
from agent.tools.rag_retrieval_tool import (
    MODE_METADATA,
    MODE_OVERVIEW,
    MODE_PAGE,
    MODE_SEMANTIC,
    retrieve_page,
)
from ingestion.vectorstore import VectorStoreError, get_document_chunks

# Routing now lives in its own module. These names are re-exported so that
# anything already importing them from services.qa_service keeps working.
from services.query_understanding import (  # noqa: F401
    LAST_PAGE,
    ConversationContext,
    DocumentInfo,
    QueryUnderstanding,
    _detect_page_request,
    _looks_like_document_overview,
    _looks_like_document_stats,
    resolve_page_reference,
    understand,
)

# Deterministic "no evidence" text, in the same script as the question.
# This is a simple heuristic (Hebrew Unicode block vs. not) — good enough
# for the MVP. No retrieved evidence is ever translated anywhere in this
# service; only this fixed, evidence-free message is language-switched.
_NOT_FOUND_HE = "המידע לא נמצא בחומר הלימוד שהועלה."
_NOT_FOUND_EN = "The information was not found in the uploaded course material."

_HEBREW_RE = re.compile(r"[֐-׿]")

# Shown when the student asks about a page the document does not have.
# Deterministic, and decided from stored page metadata before the Agent
# is ever run — the model is never given a chance to invent that page.
_PAGE_MISSING_HE = "העמוד המבוקש לא קיים במסמך שהועלה."
_PAGE_MISSING_EN = "The requested page does not exist in the uploaded document."

# Appended (deterministically, not by the model) to a document-level
# overview answer, pointing the student at the fuller feature.
_OVERVIEW_HINT_HE = "לסיכום מפורט של כל המסמך אפשר להשתמש ב-Full Study Summary."
_OVERVIEW_HINT_EN = (
    "For a detailed summary of the whole document, use Full Study Summary."
)

class QAServiceError(RuntimeError):
    """
    Raised for a Q&A-service-level technical failure: empty/whitespace
    question, missing OPENAI_API_KEY, an Agent/CrewAI execution failure,
    or a detected internal RAG Tool failure (see answer_question()'s
    "tool_was_used but retrieval is None" check).

    Distinct from "no evidence found", which is a normal, successful
    QAResponse (found_evidence=False) — not an error. Messages here are
    always safe to show a user: never an API key, stack trace, or
    embedding vector.
    """


@dataclass
class SourceReference:
    """One user-facing citation, built only from real retrieved-chunk metadata."""

    source_filename: str
    page_number: int
    chunk_index: int
    document_id: str
    # None when the evidence came from structural retrieval (a specific
    # page, or whole-document overview) rather than similarity search —
    # there is no meaningful "distance" for an exact metadata match.
    distance: Optional[float]


@dataclass
class QAResponse:
    """The Q&A service's structured result — what a future UI will render."""

    question: str
    answer: str
    document_id: Optional[str]
    found_evidence: bool
    tool_was_used: bool
    sources: List[SourceReference] = field(default_factory=list)
    # Which retrieval strategy answered this question: MODE_SEMANTIC
    # (normal Q&A), MODE_PAGE ("what is on page 5?") or MODE_OVERVIEW
    # ("what is in this document?").
    retrieval_mode: str = MODE_SEMANTIC
    # True when the question had two reasonable readings and the service asked
    # which was meant. found_evidence is False here, but this is NOT the
    # "not found" answer — nothing was searched yet, so a UI should show it as
    # a question back to the student, not as a dead end.
    needs_clarification: bool = False
    # What the session should remember from this turn so a follow-up like
    # "ומה בעמוד הבא?" or "תסביר את זה יותר פשוט" can resolve. Session-scoped
    # and document-scoped; see ConversationContext.
    resolved_page: Optional[int] = None
    topic: Optional[str] = None
    # A one-line, safe summary of how the question was interpreted: intent,
    # normalized wording, page, search wordings. Never contains a prompt, a
    # key, or document text — it exists so a wrong answer can be debugged
    # without guessing which stage misread the question.
    debug: Optional[str] = None


def _looks_hebrew(text: str) -> bool:
    return bool(_HEBREW_RE.search(text))


def _not_found_message(question: str) -> str:
    return _NOT_FOUND_HE if _looks_hebrew(question) else _NOT_FOUND_EN


def _page_missing_message(question: str) -> str:
    return _PAGE_MISSING_HE if _looks_hebrew(question) else _PAGE_MISSING_EN


def _overview_hint(question: str) -> str:
    return _OVERVIEW_HINT_HE if _looks_hebrew(question) else _OVERVIEW_HINT_EN


def _document_stats_answer(
    question: str, filename: str, page_count: int, chunk_count: int
) -> str:
    """
    Build the answer for a document-stats question from real indexed
    counts. Deterministic string assembly - the model is not involved,
    so these numbers can never drift from what was actually stored.
    """
    if _looks_hebrew(question):
        return (
            f'המסמך הפעיל הוא "{filename}".\n\n'
            f"הוא נסרק במלואו ונשמרו ממנו {page_count} עמודים "
            f"ב-{chunk_count} קטעי טקסט (chunks).\n\n"
            "אפשר לשאול אותי על התוכן עצמו, לבקש \"מה יש בקובץ?\" לסקירה "
            "כללית, או לשאול על עמוד מסוים - למשל \"מה יש בעמוד 2?\"."
        )
    return (
        f'The active document is "{filename}".\n\n'
        f"It was fully processed into {page_count} pages "
        f"across {chunk_count} text chunks.\n\n"
        "You can ask about its content, ask \"what is in this document?\" "
        "for an overview, or ask about a specific page."
    )


def _topic_of(understanding) -> Optional[str]:
    """
    What this turn was about, for a later follow-up to inherit.

    The first rewritten search wording is used when there is one — it is the
    topic stated in the material's own words — otherwise the question itself.
    """
    if understanding.referenced_topic:
        return understanding.referenced_topic
    if understanding.retrieval_queries:
        return understanding.retrieval_queries[0]
    if understanding.intent == MODE_SEMANTIC:
        return understanding.original_query
    return None


def _is_no_evidence_answer(answer_text: str) -> bool:
    """
    True when the Agent signalled that the retrieved excerpts do not
    actually answer the question (see agent.crew.NO_EVIDENCE_SENTINEL).

    This is what fixes the false-negative/inconsistency bug where the
    answer text said "not found" while the response still reported
    found_evidence=True and listed sources. It can only ever turn a
    response INTO "not found" — never the other way around.
    """
    return NO_EVIDENCE_SENTINEL in (answer_text or "").upper()


def _is_better(candidate, existing: SourceReference) -> bool:
    """
    Should `candidate` replace `existing` as the single citation for a page?

    Semantic retrieval ranks by L2 distance, so lower wins. Structural
    retrieval (page / overview) has NO distance — every chunk on that page
    is equally "the page" — so the first chunk seen is kept and later ones
    are ignored.

    The None checks are load-bearing, not defensive noise: without them a
    document with more than one chunk on the same page raised
    "TypeError: '<' not supported between instances of 'NoneType' and
    'NoneType'" the moment a page/overview answer cited that page.
    """
    if candidate.distance is None or existing.distance is None:
        return False
    return candidate.distance < existing.distance


def _build_sources(chunks) -> List[SourceReference]:
    """
    Turn retrieved chunks into a clean, deduplicated, sorted citation list.

    Deduplication rule: one SourceReference per unique
    (source_filename, page_number) pair. When multiple retrieved chunks
    land on the same page, only the BEST one for that page is kept — its
    chunk_index is what gets reported. This only shapes the final
    user-facing list; it never mutates or drops anything from the
    underlying RetrievalResult itself.

    "Best" depends on how the evidence was retrieved: semantic results
    rank by distance (lower = better), while structural results (a
    specific page, or a whole-document overview) carry no distance at
    all, so the first chunk seen for a page wins. See _is_better().

    Sort order: ascending distance (lowest first) — since this project
    uses Chroma's L2 distance, lower means more relevant, so the most
    relevant page is listed first.
    """
    best_by_page: Dict[Tuple[str, int], SourceReference] = {}
    for c in chunks:
        key = (c.source_filename, c.page_number)
        existing = best_by_page.get(key)
        if existing is None or _is_better(c, existing):
            best_by_page[key] = SourceReference(
                source_filename=c.source_filename,
                page_number=c.page_number,
                chunk_index=c.chunk_index,
                document_id=c.document_id,
                distance=c.distance,
            )
    # Similarity results sort by relevance (lowest L2 distance first).
    # Structural results (page / overview) have no distance, so they keep
    # the document's own reading order instead.
    values = list(best_by_page.values())
    if any(v.distance is not None for v in values):
        return sorted(
            values,
            key=lambda s: (s.distance is None, s.distance if s.distance is not None else 0.0),
        )
    return sorted(values, key=lambda s: (s.page_number, s.chunk_index))


def answer_question(
    question: str,
    document_id: Optional[str] = None,
    context: Optional[ConversationContext] = None,
) -> QAResponse:
    """
    Answer one student question, grounded in their uploaded course
    material, via the Phase 6 CrewAI Agent (agent.run_agent).

    Args:
        question: the student's natural-language question. Must be
            non-empty, non-whitespace-only.
        document_id: if given, keeps retrieval scoped to that document —
            propagated straight through to run_agent() -> the RAG Tool ->
            retrieve() -> Chroma's `where` filter. Phase 5/6 document
            isolation is preserved end-to-end, not re-implemented here.

    Returns:
        QAResponse. When the Agent genuinely found no supporting
        evidence (searched but nothing was relevant enough, or no
        document matched document_id), this is still a normal,
        successful QAResponse: found_evidence=False, a deterministic
        "not found" answer in the question's language, sources=[]. This
        is NOT an error — a course simply not covering a topic is an
        expected, valid outcome.

    Raises:
        QAServiceError: the question is empty/whitespace, OPENAI_API_KEY
            is missing, the underlying Agent/CrewAI run failed for any
            other reason, or the RAG Tool was invoked but failed
            internally. That last case is detected as tool_was_used=True
            together with retrieval=None: agent.crew's tool wrapper only
            ever leaves retrieval as None when its own call raised an
            exception internally (on success it always sets a
            RetrievalResult, even one with found=False) — so this
            specific combination can only mean a technical failure, never
            "nothing relevant was found". Treating it as a technical
            failure (raising) rather than silently returning "not found"
            is what strict grounding requires: an error-message string
            must never be mistaken for retrieved evidence.
    """
    if not question or not question.strip():
        raise QAServiceError("question must be a non-empty string")

    # ---- Query Understanding -----------------------------------------
    # An LLM reads the question and proposes HOW to search it; deterministic
    # guards in services/query_understanding.py then sanity-check that
    # proposal. What it can NEVER do is decide that no answer exists, or
    # reach a different document — document_id is not even passed to it.
    # See that module's docstring for the measurements behind this design.
    # Conversation context is scoped to the active document: `for_document`
    # returns an empty context the moment a different document is active, so a
    # page or topic remembered from document A can never steer a question
    # about document B.
    ctx = (context or ConversationContext()).for_document(document_id)

    # What the interpreter is told about the document: its human-facing name
    # and how many pages it has. Enough to reject an out-of-range page number,
    # and deliberately NOT the document_id — the interpreter has no way to
    # name a different document because it never receives one.
    doc_info = None
    if document_id:
        try:
            _records = get_document_chunks(document_id)
        except VectorStoreError:
            _records = []
        if _records:
            doc_info = DocumentInfo(
                document_id=document_id,
                filename=_records[0].get("source_filename", document_id),
                page_count=len({r["page_number"] for r in _records}),
            )

    understanding = (
        understand(question, context=ctx, document=doc_info)
        if document_id
        else QueryUnderstanding(original_query=question, intent=MODE_SEMANTIC)
    )

    if understanding.needs_clarification:
        # Two reasonable readings. Ask, rather than answer the wrong one — and
        # rather than the one clearly wrong response, "not found".
        return QAResponse(
            question=question,
            answer=understanding.clarification_question or "",
            document_id=document_id,
            found_evidence=False,
            tool_was_used=False,
            sources=[],
            retrieval_mode=understanding.intent,
            needs_clarification=True,
            debug=understanding.debug_line(),
        )

    retrieval_mode = understanding.intent
    page_number = understanding.page_number

    if document_id and retrieval_mode == MODE_METADATA:
        # Answered straight from the index's own bookkeeping, before any
        # agent/LLM call: exact counts cannot be guessed or drifted.
        try:
            records = get_document_chunks(document_id)
        except VectorStoreError as e:
            raise QAServiceError(
                f"Could not read the uploaded document: {e}"
            ) from e
        if records:
            return QAResponse(
                question=question,
                answer=_document_stats_answer(
                    question,
                    records[0].get("source_filename", document_id),
                    len({r["page_number"] for r in records}),
                    len(records),
                ),
                document_id=document_id,
                found_evidence=True,
                tool_was_used=False,
                # Intentionally empty: this fact is not written on any
                # page, so citing a page would be a false citation.
                sources=[],
                retrieval_mode=MODE_METADATA,
                debug=understanding.debug_line(),
            )
        # No indexed content: fall through to the normal grounded path.
        retrieval_mode = MODE_SEMANTIC

    if document_id and retrieval_mode == MODE_PAGE and page_number == LAST_PAGE:
        # "the last page" is only a number once the document is known.
        try:
            records = get_document_chunks(document_id)
        except VectorStoreError as e:
            raise QAServiceError(f"Could not read the uploaded document: {e}") from e
        pages = {r["page_number"] for r in records} if records else set()
        page_number = max(pages) if pages else None
        if page_number is None:
            retrieval_mode = MODE_SEMANTIC

    if document_id and retrieval_mode == MODE_PAGE and page_number is not None:
        try:
            page_probe = retrieve_page(document_id, page_number)
        except Exception as e:
            raise QAServiceError(
                f"Could not read page {page_number} from the uploaded "
                f"document: {e}"
            ) from e
        if not page_probe.page_exists:
            # Answered without calling the Agent at all: the page simply is
            # not in the document, so there is nothing to ground an answer
            # in and nothing to risk inventing.
            return QAResponse(
                question=question,
                answer=_page_missing_message(question),
                document_id=document_id,
                found_evidence=False,
                tool_was_used=False,
                sources=[],
                retrieval_mode=MODE_PAGE,
                resolved_page=page_number,
                debug=understanding.debug_line(),
            )
    elif retrieval_mode == MODE_PAGE:
        page_number = None
        retrieval_mode = MODE_SEMANTIC

    try:
        result = run_agent(
            question,
            document_id=document_id,
            verbose=False,
            page_number=page_number,
            overview=(retrieval_mode == MODE_OVERVIEW),
            # Extra search wordings from Query Understanding. The agent still
            # answers the ORIGINAL question; these only give its tool better
            # search terms (course material mixes Hebrew prose with English
            # keywords, so a Hebrew-only search regularly misses the page).
            suggested_queries=understanding.retrieval_queries,
        )
    except AgentError as e:
        raise QAServiceError(f"Agent could not answer the question: {e}") from e
    except Exception as e:  # unexpected CrewAI/OpenAI failure not already an AgentError
        raise QAServiceError(f"Unexpected agent failure: {e}") from e

    if result.tool_was_used and result.retrieval is None:
        raise QAServiceError(
            "The course material search failed internally, so no answer "
            "can be safely grounded. Please try again."
        )

    retrieval = result.retrieval
    has_evidence = (
        result.tool_was_used
        and retrieval is not None
        and retrieval.found
        and bool(retrieval.chunks)
    )

    # Second grounding gate: the Agent actually read the retrieved text
    # and reported that it does not contain the answer (or that the
    # question tried to talk it out of using the material at all). Only
    # ever downgrades to "not found" — an answer still requires evidence.
    if has_evidence and _is_no_evidence_answer(result.answer_text):
        has_evidence = False

    if not has_evidence:
        return QAResponse(
            question=question,
            answer=_not_found_message(question),
            document_id=document_id,
            found_evidence=False,
            tool_was_used=result.tool_was_used,
            sources=[],
            retrieval_mode=retrieval_mode,
            resolved_page=page_number,
            topic=_topic_of(understanding),
            debug=understanding.debug_line(),
        )

    answer_text = result.answer_text
    if retrieval_mode == MODE_OVERVIEW:
        # Deterministic pointer to the fuller feature — appended by this
        # service, never generated (or hallucinated) by the model.
        answer_text = f"{answer_text}\n\n{_overview_hint(question)}"

    return QAResponse(
        question=question,
        answer=answer_text,
        document_id=document_id,
        found_evidence=True,
        tool_was_used=True,
        sources=_build_sources(retrieval.chunks),
        retrieval_mode=retrieval_mode,
        resolved_page=page_number,
        topic=_topic_of(understanding),
        debug=understanding.debug_line(),
    )

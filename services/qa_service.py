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

# Deterministic "no evidence" text, in the same script as the question.
# This is a simple heuristic (Hebrew Unicode block vs. not) — good enough
# for the MVP. No retrieved evidence is ever translated anywhere in this
# service; only this fixed, evidence-free message is language-switched.
_NOT_FOUND_HE = "המידע לא נמצא בחומר הלימוד שהועלה."
_NOT_FOUND_EN = "The information was not found in the uploaded course material."

_HEBREW_RE = re.compile(r"[֐-׿]")


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
    distance: float


@dataclass
class QAResponse:
    """The Q&A service's structured result — what a future UI will render."""

    question: str
    answer: str
    document_id: Optional[str]
    found_evidence: bool
    tool_was_used: bool
    sources: List[SourceReference] = field(default_factory=list)


def _looks_hebrew(text: str) -> bool:
    return bool(_HEBREW_RE.search(text))


def _not_found_message(question: str) -> str:
    return _NOT_FOUND_HE if _looks_hebrew(question) else _NOT_FOUND_EN


def _build_sources(chunks) -> List[SourceReference]:
    """
    Turn retrieved chunks into a clean, deduplicated, sorted citation list.

    Deduplication rule: one SourceReference per unique
    (source_filename, page_number) pair. When multiple retrieved chunks
    land on the same page, only the one with the LOWEST distance (best
    match) for that page is kept — its chunk_index is what gets reported.
    This only shapes the final user-facing list; it never mutates or
    drops anything from the underlying RetrievalResult itself.

    Sort order: ascending distance (lowest first) — since this project
    uses Chroma's L2 distance, lower means more relevant, so the most
    relevant page is listed first.
    """
    best_by_page: Dict[Tuple[str, int], SourceReference] = {}
    for c in chunks:
        key = (c.source_filename, c.page_number)
        existing = best_by_page.get(key)
        if existing is None or c.distance < existing.distance:
            best_by_page[key] = SourceReference(
                source_filename=c.source_filename,
                page_number=c.page_number,
                chunk_index=c.chunk_index,
                document_id=c.document_id,
                distance=c.distance,
            )
    return sorted(best_by_page.values(), key=lambda s: s.distance)


def answer_question(question: str, document_id: Optional[str] = None) -> QAResponse:
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

    try:
        result = run_agent(question, document_id=document_id, verbose=False)
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

    if not has_evidence:
        return QAResponse(
            question=question,
            answer=_not_found_message(question),
            document_id=document_id,
            found_evidence=False,
            tool_was_used=result.tool_was_used,
            sources=[],
        )

    return QAResponse(
        question=question,
        answer=result.answer_text,
        document_id=document_id,
        found_evidence=True,
        tool_was_used=True,
        sources=_build_sources(retrieval.chunks),
    )

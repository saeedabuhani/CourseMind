"""
Full-document Summary service (Phase 8).

A separate workflow from Q&A (services/qa_service.py). Where Q&A answers
one question from the top-k most relevant chunks, a summary must cover
the WHOLE document — so this module never calls Phase 5's
retrieve()/top-k search. Instead it loads every stored chunk for a
document_id (ingestion.vectorstore.get_document_chunks — a plain Chroma
`where` filter, not a similarity search) and summarizes all of it via a
hierarchical map-reduce over the OpenAI chat model:

    all chunks (sorted into document order)
        -> batched by a token budget (not top-k, not a chunk-count guess)
        -> each batch summarized independently ("map")
        -> intermediate summaries combined ("reduce"); if there are still
           too many/too-large intermediate summaries for one final call,
           an extra reduction level consolidates them further
        -> one final, structured study summary

This intentionally does NOT go through the Phase 6 CrewAI Agent — the
assignment's Agent requirement is already satisfied by the Q&A
architecture (Phases 5-7); forcing the Agent to repeatedly call a top-k
tool would be the wrong tool for "summarize everything" and wouldn't
change how many OpenAI calls this workflow needs anyway. Calling the
OpenAI chat model directly here is a deliberate, separate workflow, not a
shortcut around the Agent requirement.
"""

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import APIConnectionError, APIError, AuthenticationError, OpenAI

from ingestion.vectorstore import VectorStoreError, get_document_chunks

load_dotenv()

DEFAULT_MODEL = "gpt-4o-mini"

# Token budget per "map" batch. Chosen deliberately smaller than
# gpt-4o-mini's actual context window (128k tokens) — this is not a
# model limitation, it's a design choice to keep each individual
# summarization call focused on a manageable slice of the document
# (mirrors ingestion/chunking.py's own reasoning for using a token
# budget rather than the model's hard limit). Reusing each chunk's
# already-computed token_count metadata (Phase 3) means no re-tokenizing
# happens here.
DEFAULT_BATCH_TOKEN_BUDGET = 3000

# Token budget (estimated, see _estimate_tokens) for how many
# intermediate summaries can be combined in one "reduce" call before an
# extra reduction level is needed first.
DEFAULT_REDUCE_TOKEN_BUDGET = 3000

# Hard safety cap on reduction levels, in case some pathological input
# (e.g. a single summary permanently over budget) would otherwise loop.
MAX_REDUCE_LEVELS = 5

_HEBREW_CHAR_RE = re.compile(r"[֐-׿]")
_LATIN_CHAR_RE = re.compile(r"[A-Za-z]")


class SummaryServiceError(RuntimeError):
    """
    Raised for a Summary-service-level failure: empty/missing
    document_id, no indexed content for that document, missing
    OPENAI_API_KEY, an OpenAI API failure, a ChromaDB read failure,
    malformed stored chunk metadata, an empty intermediate summary, or
    any other unexpected failure in the summarization pipeline.

    Messages here are always safe to show a user: never an API key,
    stack trace, or embedding vector.
    """


@dataclass
class SummaryResponse:
    """The Summary service's structured result."""

    document_id: str
    source_filename: str
    summary: str
    total_chunks: int
    total_pages: int
    batches_processed: int
    language: str


def _detect_dominant_language(chunks: List[Dict]) -> str:
    """
    Simple, MVP-appropriate heuristic: count Hebrew vs. Latin letters
    across the document's actual text and pick whichever script
    dominates. Good enough to choose Hebrew for a Hebrew course PDF (even
    one full of English technical terms like "JavaScript" or "function")
    without needing a real language-detection library.
    """
    sample = " ".join(c["text"] for c in chunks)
    hebrew_count = len(_HEBREW_CHAR_RE.findall(sample))
    latin_count = len(_LATIN_CHAR_RE.findall(sample))
    return "Hebrew" if hebrew_count >= latin_count else "English"


def _make_batches(chunks: List[Dict], max_tokens: int) -> List[List[Dict]]:
    """
    Group already-ordered chunks into batches whose token_count sums
    stay under max_tokens, without ever splitting a single chunk or
    reordering anything. Chunks are assumed pre-sorted (page_number,
    chunk_index) by get_document_chunks(), so document order is
    preserved both across and within batches.
    """
    batches: List[List[Dict]] = []
    current: List[Dict] = []
    current_tokens = 0
    for c in chunks:
        t = c["token_count"]
        if current and current_tokens + t > max_tokens:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(c)
        current_tokens += t
    if current:
        batches.append(current)
    return batches


def _page_range(batch: List[Dict]) -> str:
    pages = [c["page_number"] for c in batch]
    lo, hi = min(pages), max(pages)
    return f"{lo}" if lo == hi else f"{lo}-{hi}"


def _join_batch_text(batch: List[Dict]) -> str:
    return "\n\n".join(f"[Page {c['page_number']}]\n{c['text']}" for c in batch)


def _estimate_tokens(text: str) -> int:
    """Char-based approximation (~4 chars/token) for freshly-generated
    summary text, which has no stored token_count of its own."""
    return max(1, len(text) // 4)


def _group_texts_by_budget(texts: List[str], max_tokens: int) -> List[List[str]]:
    groups: List[List[str]] = []
    current: List[str] = []
    current_tokens = 0
    for t in texts:
        tok = _estimate_tokens(t)
        if current and current_tokens + tok > max_tokens:
            groups.append(current)
            current = []
            current_tokens = 0
        current.append(t)
        current_tokens += tok
    if current:
        groups.append(current)
    return groups


def _map_prompt(material: str, language: str, part_label: str) -> str:
    return (
        "You are helping build a study summary for a university/college "
        "course, from real uploaded course material.\n\n"
        f"Below is {part_label} of a larger document. Write faithful, "
        "well-organized study notes covering ONLY the material in this "
        "part.\n\n"
        "Rules:\n"
        "- Use ONLY the material provided below. Do not invent facts, "
        "examples, or definitions that are not present.\n"
        "- Preserve important technical terminology exactly as written "
        "(e.g. programming keywords, function/variable names) instead of "
        "translating it.\n"
        "- Organize your notes clearly (topics, key concepts, "
        "definitions, rules, examples) — but only include a category if "
        "this part of the material actually supports it.\n"
        "- Be concise but do not omit distinct topics covered in this "
        "part.\n"
        f"- Write in {language}.\n\n"
        "Material:\n---\n"
        f"{material}\n---"
    )


def _final_prompt(material: str, language: str) -> str:
    return (
        "You are creating the FINAL study summary for a university/"
        "college student preparing for an exam, based on faithful notes "
        "already extracted from their real uploaded course material.\n\n"
        "Combine the notes below into ONE coherent, well-structured "
        "study summary using this structure — skip any section the "
        "material does not support; never invent content to fill a "
        "section:\n"
        "- Document title / source\n"
        "- Main topics\n"
        "- Key concepts\n"
        "- Important definitions\n"
        "- Important rules / principles\n"
        "- Important examples (where present)\n"
        "- Important terminology\n"
        "- Exam-study takeaways\n\n"
        "Rules:\n"
        "- Use ONLY the notes provided below. Do not add external facts "
        "or general knowledge.\n"
        "- Preserve important technical terminology exactly as written "
        "(e.g. programming keywords) instead of translating it.\n"
        f"- Write the summary in {language}.\n"
        "- Be comprehensive enough to cover everything in the notes, but "
        "concise enough to study from.\n\n"
        "Notes:\n---\n"
        f"{material}\n---"
    )


def _call_llm(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SummaryServiceError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
    except AuthenticationError as e:
        raise SummaryServiceError(
            "OpenAI rejected the API key (authentication failed). "
            "Check OPENAI_API_KEY in your .env file."
        ) from e
    except APIConnectionError as e:
        raise SummaryServiceError(
            "Could not reach the OpenAI API (network/connection error)."
        ) from e
    except APIError as e:
        raise SummaryServiceError(f"OpenAI chat completion request failed: {e}") from e

    return (response.choices[0].message.content or "").strip()


def summarize_document(
    document_id: str, language: Optional[str] = None
) -> SummaryResponse:
    """
    Produce a full-document study summary for every chunk stored under
    `document_id` — never a top-k subset.

    Args:
        document_id: which indexed document to summarize. Must be
            non-empty.
        language: force the summary's language (e.g. "Hebrew",
            "English"). If None, it's auto-detected from the document's
            own text (see _detect_dominant_language).

    Returns:
        SummaryResponse with the final summary text and coverage
        metadata (total_chunks, total_pages, batches_processed) so a
        caller can show e.g. "Summary generated from 21 pages / 27
        chunks".

    Raises:
        SummaryServiceError: document_id is empty/whitespace, no chunks
            are indexed for it, stored chunk metadata is malformed,
            OPENAI_API_KEY is missing, an OpenAI call failed, the vector
            store read failed, a batch produced an empty summary, or any
            other unexpected pipeline failure. The document is never
            summarized from general knowledge as a fallback — if the
            real material can't be safely processed, this raises instead
            of inventing content.
    """
    if not document_id or not document_id.strip():
        raise SummaryServiceError("document_id must be a non-empty string")

    try:
        chunks = get_document_chunks(document_id)
    except VectorStoreError as e:
        raise SummaryServiceError(
            f"Could not read the document from the vector store: {e}"
        ) from e

    if not chunks:
        raise SummaryServiceError(
            f"No indexed content found for document_id '{document_id}'. "
            "Nothing to summarize."
        )

    try:
        source_filename = chunks[0]["source_filename"]
        total_pages = len({c["page_number"] for c in chunks})
    except KeyError as e:
        raise SummaryServiceError(
            f"Stored chunk metadata is missing an expected field: {e}"
        ) from e

    resolved_language = language or _detect_dominant_language(chunks)
    batches = _make_batches(chunks, DEFAULT_BATCH_TOKEN_BUDGET)

    try:
        if len(batches) == 1:
            # Short document: no map/reduce distinction needed — one
            # direct call straight to the final, fully-structured prompt.
            material = _join_batch_text(batches[0])
            final_summary = _call_llm(_final_prompt(material, resolved_language))
        else:
            intermediate: List[str] = []
            for i, batch in enumerate(batches, start=1):
                part_label = (
                    f"part {i} of {len(batches)} (pages {_page_range(batch)})"
                )
                note = _call_llm(
                    _map_prompt(_join_batch_text(batch), resolved_language, part_label)
                )
                if not note:
                    raise SummaryServiceError(
                        f"Batch {i}/{len(batches)} produced an empty summary — "
                        "aborting rather than silently losing coverage of "
                        f"pages {_page_range(batch)}."
                    )
                intermediate.append(note)

            current_level = intermediate
            for _ in range(MAX_REDUCE_LEVELS):
                if _estimate_tokens("\n\n".join(current_level)) <= DEFAULT_REDUCE_TOKEN_BUDGET:
                    break
                groups = _group_texts_by_budget(current_level, DEFAULT_REDUCE_TOKEN_BUDGET)
                if len(groups) >= len(current_level):
                    break  # no consolidation possible; proceed with what we have
                new_level = []
                for i, group in enumerate(groups, start=1):
                    part_label = f"consolidated notes group {i} of {len(groups)}"
                    note = _call_llm(
                        _map_prompt("\n\n".join(group), resolved_language, part_label)
                    )
                    if not note:
                        raise SummaryServiceError(
                            f"Reduction group {i}/{len(groups)} produced an "
                            "empty summary — aborting rather than silently "
                            "losing coverage."
                        )
                    new_level.append(note)
                current_level = new_level

            final_summary = _call_llm(
                _final_prompt("\n\n".join(current_level), resolved_language)
            )
    except SummaryServiceError:
        raise
    except Exception as e:  # unexpected pipeline failure not already handled above
        raise SummaryServiceError(f"Unexpected summary pipeline failure: {e}") from e

    return SummaryResponse(
        document_id=document_id,
        source_filename=source_filename,
        summary=final_summary,
        total_chunks=len(chunks),
        total_pages=total_pages,
        batches_processed=len(batches),
        language=resolved_language,
    )

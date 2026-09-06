"""
CourseMind evaluation runner (Phase 10).

Exercises the REAL backend end-to-end for every case: services.answer_question()
(which itself calls the real CrewAI Agent -> real RAG Retrieval Tool -> real
ChromaDB -> real OpenAI API) and services.summarize_document() (real
map-reduce over real chunks). Only evals/test_cases.json's EVAL-013
(failure_safety) mocks anything, and it mocks a single internal function to
simulate a technical failure — it does not bypass the Q&A service itself.

Usage:
    python -m evals.run_evals --pdf "path/to/JavaScript_חלק_א_מדריך_לימוד.pdf"

No absolute paths are hardcoded here: the reference PDF's path comes from
--pdf, and the second ("secondary") evaluation document is a small synthetic
fixture built in this file, not a real file on disk, so document-isolation
and English-language checks stay fully portable to another computer.

The dataset's *content* expectations (expected pages, page/chunk counts) are
calibrated to the specific Hebrew JavaScript PDF used throughout this
project's development — see test_cases.json's "_meta" block. Running this
against a different --pdf is fine for the mechanics, but content-specific
checks (keywords, source pages, page/chunk counts) will legitimately fail
since they no longer describe the supplied document.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ingestion.chunking import ChunkData
    from ingestion.embeddings import EmbeddingError, embed_chunks
    from ingestion.pdf_loader import PDFLoadError
    from ingestion.pipeline import ingest_document
    from ingestion.vectorstore import (
        DEFAULT_COLLECTION_NAME,
        VectorStoreError,
        get_client,
        get_collection,
        save_chunks,
    )
    from services import (
        QAServiceError,
        SummaryServiceError,
        answer_question,
        summarize_document,
    )
else:
    from ingestion.chunking import ChunkData
    from ingestion.embeddings import EmbeddingError, embed_chunks
    from ingestion.pdf_loader import PDFLoadError
    from ingestion.pipeline import ingest_document
    from ingestion.vectorstore import (
        DEFAULT_COLLECTION_NAME,
        VectorStoreError,
        get_client,
        get_collection,
        save_chunks,
    )
    from services import (
        QAServiceError,
        SummaryServiceError,
        answer_question,
        summarize_document,
    )

TEST_CASES_PATH = Path(__file__).resolve().parent / "test_cases.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# The secondary evaluation document: a small synthetic fixture (not a real
# file) so isolation/English checks never depend on a private user PDF.
SECONDARY_DOCUMENT_ID = "roman_empire_synthetic.pdf"
_SECONDARY_CHUNK_TEXTS = [
    "The Roman Empire was founded in 27 BC when Octavian became Augustus, the first Roman emperor.",
    "The fall of the Western Roman Empire in 476 AD marked the end of ancient Rome and the start of the medieval period.",
]


class EvalSetupError(RuntimeError):
    """Raised for a fatal setup failure that stops the whole suite cleanly."""


@dataclass
class EvalResult:
    id: str
    type: str
    categories: List[str]
    passed: bool
    checks_passed: int
    checks_total: int
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> float:
        return self.checks_passed / self.checks_total if self.checks_total else 0.0


def _count_keyword_matches(text: str, keywords: List[str]) -> int:
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw.lower() in text_lower)


def _load_test_cases() -> List[Dict[str, Any]]:
    try:
        with open(TEST_CASES_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise EvalSetupError(f"test_cases.json not found at {TEST_CASES_PATH}") from e
    except json.JSONDecodeError as e:
        raise EvalSetupError(f"test_cases.json is not valid JSON: {e}") from e

    cases = data.get("test_cases")
    if not isinstance(cases, list) or not cases:
        raise EvalSetupError("test_cases.json has no non-empty 'test_cases' list")
    return cases


def _index_primary(pdf_path: str) -> str:
    """Index the real reference PDF once. Fatal on failure — nothing else can run."""
    try:
        result = ingest_document(pdf_path)
    except FileNotFoundError as e:
        raise EvalSetupError(f"--pdf file not found: {e}") from e
    except PDFLoadError as e:
        raise EvalSetupError(f"--pdf could not be read as a valid PDF: {e}") from e
    except EmbeddingError as e:
        raise EvalSetupError(f"Could not generate embeddings for --pdf: {e}") from e
    except VectorStoreError as e:
        raise EvalSetupError(f"Could not store --pdf in the vector store: {e}") from e

    if not result["chunks"]:
        raise EvalSetupError("--pdf produced zero extractable chunks — nothing to evaluate")

    return result["chunks"][0].document_id


def _build_secondary_document() -> Optional[str]:
    """
    Index the small synthetic secondary document. Returns its document_id on
    success, or None on failure (a soft failure — only cases scoped to
    'secondary' are affected; the rest of the suite still runs).
    """
    try:
        chunks = [
            ChunkData(
                text=text,
                source_filename=SECONDARY_DOCUMENT_ID,
                page_number=1,
                chunk_index=i,
                document_id=SECONDARY_DOCUMENT_ID,
                token_count=max(1, len(text) // 4),
                character_count=len(text),
            )
            for i, text in enumerate(_SECONDARY_CHUNK_TEXTS)
        ]
        embeddings = embed_chunks(chunks)
        save_chunks(chunks, embeddings)
        return SECONDARY_DOCUMENT_ID
    except Exception as e:
        print(f"WARNING: could not set up the secondary evaluation document: {e}")
        return None


def _cleanup_secondary_document() -> None:
    """Remove the synthetic secondary document's chunks — not a permanent fixture."""
    try:
        client = get_client()
        collection = get_collection(client, DEFAULT_COLLECTION_NAME)
        collection.delete(where={"document_id": SECONDARY_DOCUMENT_ID})
    except Exception as e:
        print(f"WARNING: could not clean up the secondary evaluation document: {e}")


def _run_qa_case(
    case: Dict[str, Any], document_ids: Dict[str, Optional[str]], responses: Dict[str, Any]
) -> EvalResult:
    case_id = case["id"]
    categories = case.get("categories", [])

    reuse_id = case.get("reuse_case_id")
    if reuse_id:
        response = responses.get(reuse_id)
        if response is None:
            return EvalResult(
                case_id, "qa", categories, False, 0, 1,
                reason=f"reuse_case_id '{reuse_id}' has no recorded response (it must run first and succeed)",
            )
    else:
        alias = case["document_alias"]
        document_id = document_ids.get(alias)
        if document_id is None:
            return EvalResult(
                case_id, "qa", categories, False, 0, 1,
                reason=f"document alias '{alias}' is unavailable (setup failed) — case skipped",
            )
        try:
            response = answer_question(case["question"], document_id=document_id)
        except QAServiceError as e:
            return EvalResult(case_id, "qa", categories, False, 0, 1, reason=f"QAServiceError: {e}")
        except Exception as e:
            return EvalResult(case_id, "qa", categories, False, 0, 1, reason=f"Unexpected error: {e}")
        responses[case_id] = response

    checks: List[bool] = []
    details: Dict[str, Any] = {
        "found_evidence": response.found_evidence,
        "tool_was_used": response.tool_was_used,
        "source_pages": [s.page_number for s in response.sources],
    }
    failure_notes: List[str] = []

    # Always checked: this is a real Agent invoking a real Tool, every time.
    checks.append(response.tool_was_used is True)
    if not response.tool_was_used:
        failure_notes.append("tool_was_used was False")

    should_find = case.get("should_find_evidence")
    if should_find is not None:
        ok = response.found_evidence is should_find
        checks.append(ok)
        if not ok:
            failure_notes.append(f"expected found_evidence={should_find}, got {response.found_evidence}")
        if should_find is False:
            ok_sources = response.sources == []
            checks.append(ok_sources)
            if not ok_sources:
                failure_notes.append(f"expected empty sources, got {details['source_pages']}")

    expected_pages = case.get("expected_source_pages")
    if expected_pages:
        hit = any(p in details["source_pages"] for p in expected_pages)
        checks.append(hit)
        details["expected_source_pages"] = expected_pages
        if not hit:
            failure_notes.append(f"none of expected pages {expected_pages} in {details['source_pages']}")

    min_unique = case.get("min_unique_source_pages")
    if min_unique:
        unique_pages = len(set(details["source_pages"]))
        ok = unique_pages >= min_unique
        checks.append(ok)
        details["unique_source_pages"] = unique_pages
        if not ok:
            failure_notes.append(f"expected >= {min_unique} unique source pages, got {unique_pages}")

    expected_keywords = case.get("expected_keywords")
    if expected_keywords:
        min_matches = case.get("min_keyword_matches", 1)
        matches = _count_keyword_matches(response.answer, expected_keywords)
        ok = matches >= min_matches
        checks.append(ok)
        details["keyword_matches"] = f"{matches}/{len(expected_keywords)}"
        if not ok:
            failure_notes.append(
                f"only {matches}/{len(expected_keywords)} expected keywords found "
                f"(needed >= {min_matches}) among {expected_keywords}"
            )

    passed = all(checks)
    return EvalResult(
        case_id, "qa", categories, passed, sum(checks), len(checks),
        reason="; ".join(failure_notes), details=details,
    )


def _run_summary_case(
    case: Dict[str, Any], summary_response, summary_error: Optional[str]
) -> EvalResult:
    case_id = case["id"]
    categories = case.get("categories", [])

    if summary_error is not None:
        return EvalResult(case_id, "summary", categories, False, 0, 1, reason=summary_error)

    checks: List[bool] = []
    details: Dict[str, Any] = {}
    failure_notes: List[str] = []

    non_empty = bool(summary_response.summary and summary_response.summary.strip())
    checks.append(non_empty)
    if not non_empty:
        failure_notes.append("summary text is empty")

    expected_pages = case.get("expected_total_pages")
    if expected_pages is not None:
        ok = summary_response.total_pages == expected_pages
        checks.append(ok)
        details["total_pages"] = summary_response.total_pages
        if not ok:
            failure_notes.append(f"expected total_pages={expected_pages}, got {summary_response.total_pages}")

    expected_chunks = case.get("expected_total_chunks")
    if expected_chunks is not None:
        ok = summary_response.total_chunks == expected_chunks
        checks.append(ok)
        details["total_chunks"] = summary_response.total_chunks
        if not ok:
            failure_notes.append(f"expected total_chunks={expected_chunks}, got {summary_response.total_chunks}")

    expected_keywords = case.get("expected_keywords")
    if expected_keywords:
        min_matches = case.get("min_keyword_matches", 1)
        matches = _count_keyword_matches(summary_response.summary, expected_keywords)
        ok = matches >= min_matches
        checks.append(ok)
        details["keyword_matches"] = f"{matches}/{len(expected_keywords)}"
        if not ok:
            failure_notes.append(
                f"only {matches}/{len(expected_keywords)} expected keywords found "
                f"(needed >= {min_matches}) among {expected_keywords}"
            )

    passed = all(checks)
    return EvalResult(
        case_id, "summary", categories, passed, sum(checks), len(checks),
        reason="; ".join(failure_notes), details=details,
    )


def _run_failure_safety_case(case: Dict[str, Any], document_ids: Dict[str, Optional[str]]) -> EvalResult:
    case_id = case["id"]
    categories = case.get("categories", [])
    document_id = document_ids.get(case["document_alias"])

    raised = False
    reason = ""
    try:
        with patch(
            "agent.tools.rag_retrieval_tool.retrieve",
            side_effect=VectorStoreError("Simulated Chroma failure (eval)"),
        ):
            response = answer_question(case["question"], document_id=document_id)
        reason = (
            f"expected QAServiceError to be raised, but got a normal response "
            f"(found_evidence={response.found_evidence}) instead"
        )
    except QAServiceError:
        raised = True
    except Exception as e:
        reason = f"expected QAServiceError, but got {type(e).__name__}: {e}"

    return EvalResult(case_id, "failure_safety", categories, raised, int(raised), 1, reason=reason)


def _print_console_report(results: List[EvalResult], metrics: Dict[str, Any]) -> None:
    print("\nCourseMind Evaluation Report\n")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        cats = ",".join(r.categories)
        print(f"[{status}] {r.id} — {cats}")
        if not r.passed:
            print(f"         reason: {r.reason}")

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    pass_rate = (passed / total * 100) if total else 0.0

    print(f"\nPassed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {total}")
    print(f"Pass rate: {pass_rate:.1f}%\n")

    print(f"Source hit rate (Hit@K): {metrics['source_hit']['hits']}/{metrics['source_hit']['total']}")
    print(f"Grounding refusal success: {metrics['grounding_refusal']['hits']}/{metrics['grounding_refusal']['total']}")
    print(f"Document isolation: {metrics['document_isolation']['hits']}/{metrics['document_isolation']['total']}")
    print(
        f"Summary coverage (early/general topics): "
        f"{metrics['summary_coverage']['early_hits']}/{metrics['summary_coverage']['early_total']}"
    )
    print(f"Summary late-page coverage: {'PASS' if metrics['summary_coverage']['late_pass'] else 'FAIL'}")


def _build_metrics(results: List[EvalResult], cases_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    def hits_total(ids: List[str]) -> Dict[str, int]:
        relevant = [r for r in results if r.id in ids]
        return {"hits": sum(1 for r in relevant if r.passed), "total": len(relevant)}

    source_hit_ids = [c["id"] for c in cases_by_id.values() if c.get("expected_source_pages")]
    grounding_ids = [c["id"] for c in cases_by_id.values() if "no_evidence" in c.get("categories", [])]
    isolation_ids = [
        c["id"]
        for c in cases_by_id.values()
        if {"document_isolation_negative", "document_isolation_positive"} & set(c.get("categories", []))
    ]
    summary_early_id = next(
        (c["id"] for c in cases_by_id.values() if "summary_coverage" in c.get("categories", []) and "late_page_coverage" not in c.get("categories", [])),
        None,
    )
    summary_late_id = next(
        (c["id"] for c in cases_by_id.values() if "late_page_coverage" in c.get("categories", [])), None
    )
    early_result = next((r for r in results if r.id == summary_early_id), None)
    late_result = next((r for r in results if r.id == summary_late_id), None)

    return {
        "source_hit": hits_total(source_hit_ids),
        "grounding_refusal": hits_total(grounding_ids),
        "document_isolation": hits_total(isolation_ids),
        "summary_coverage": {
            "early_hits": (early_result.checks_passed if early_result else 0),
            "early_total": (early_result.checks_total if early_result else 0),
            "late_pass": bool(late_result and late_result.passed),
        },
    }


def _write_reports(results: List[EvalResult], metrics: Dict[str, Any], pdf_path: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    total = len(results)
    passed = sum(1 for r in results if r.passed)

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        "primary_pdf": Path(pdf_path).name,
        "total_cases": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total * 100, 1) if total else 0.0,
        "metrics": metrics,
        "cases": [
            {
                "id": r.id,
                "type": r.type,
                "categories": r.categories,
                "passed": r.passed,
                "score": round(r.score, 2),
                "checks_passed": r.checks_passed,
                "checks_total": r.checks_total,
                "reason": r.reason,
                "details": r.details,
            }
            for r in results
        ],
    }
    with open(RESULTS_DIR / "latest_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    lines = [
        "# CourseMind Evaluation Report",
        "",
        f"- Timestamp: {payload['timestamp']}",
        f"- Model: {payload['model']} / {payload['embedding_model']}",
        f"- Reference PDF: {payload['primary_pdf']}",
        "",
        f"**Pass rate: {payload['pass_rate']}% ({passed}/{total})**",
        "",
        "| Case | Categories | Result | Reason |",
        "|---|---|---|---|",
    ]
    for r in results:
        reason = r.reason.replace("|", "\\|") if r.reason else ""
        lines.append(f"| {r.id} | {', '.join(r.categories)} | {'PASS' if r.passed else 'FAIL'} | {reason} |")

    lines += [
        "",
        "## Metrics",
        f"- Source hit rate (Hit@K): {metrics['source_hit']['hits']}/{metrics['source_hit']['total']}",
        f"- Grounding refusal success: {metrics['grounding_refusal']['hits']}/{metrics['grounding_refusal']['total']}",
        f"- Document isolation: {metrics['document_isolation']['hits']}/{metrics['document_isolation']['total']}",
        f"- Summary coverage (early/general): {metrics['summary_coverage']['early_hits']}/{metrics['summary_coverage']['early_total']}",
        f"- Summary late-page coverage: {'PASS' if metrics['summary_coverage']['late_pass'] else 'FAIL'}",
    ]
    with open(RESULTS_DIR / "latest_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    parser = argparse.ArgumentParser(description="Run the CourseMind evaluation suite.")
    parser.add_argument(
        "--pdf", required=True, help="Path to the reference PDF (the Hebrew JavaScript course PDF)."
    )
    args = parser.parse_args()

    try:
        cases = _load_test_cases()
        print(f"Loaded {len(cases)} evaluation cases from {TEST_CASES_PATH.name}")

        print("Indexing primary document...")
        primary_id = _index_primary(args.pdf)
        print(f"Primary document indexed: {primary_id}")

        print("Setting up secondary (synthetic) evaluation document...")
        secondary_id = _build_secondary_document()
        document_ids = {"primary": primary_id, "secondary": secondary_id}
    except EvalSetupError as e:
        print(f"\nFATAL: evaluation setup failed — {e}")
        sys.exit(1)

    cases_by_id = {c["id"]: c for c in cases}
    responses: Dict[str, Any] = {}
    results: List[EvalResult] = []

    summary_response = None
    summary_error: Optional[str] = None
    summary_generated = False

    for case in cases:
        case_type = case["type"]
        try:
            if case_type == "qa":
                results.append(_run_qa_case(case, document_ids, responses))
            elif case_type == "summary":
                if not summary_generated:
                    print("Generating full-document summary (once, shared by all summary cases)...")
                    try:
                        summary_response = summarize_document(document_ids["primary"])
                    except SummaryServiceError as e:
                        summary_error = f"SummaryServiceError: {e}"
                    except Exception as e:
                        summary_error = f"Unexpected error: {e}"
                    summary_generated = True
                results.append(_run_summary_case(case, summary_response, summary_error))
            elif case_type == "failure_safety":
                results.append(_run_failure_safety_case(case, document_ids))
            else:
                results.append(
                    EvalResult(case["id"], case_type, case.get("categories", []), False, 0, 1,
                               reason=f"unknown case type '{case_type}'")
                )
        except Exception as e:
            # An individual case must never crash the whole suite.
            results.append(
                EvalResult(case["id"], case_type, case.get("categories", []), False, 0, 1,
                           reason=f"unexpected failure running this case: {e}")
            )

    metrics = _build_metrics(results, cases_by_id)
    _print_console_report(results, metrics)
    _write_reports(results, metrics, args.pdf)
    print(f"\nMachine-readable report: {RESULTS_DIR / 'latest_results.json'}")
    print(f"Markdown report: {RESULTS_DIR / 'latest_report.md'}")

    if secondary_id:
        _cleanup_secondary_document()


if __name__ == "__main__":
    main()

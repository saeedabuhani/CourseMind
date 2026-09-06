"""
CourseMind — Streamlit entry point (Phase 9 MVP UI).

This file only orchestrates the existing backend: it renders widgets,
reads their values, and calls the public ingestion/services interfaces
built in earlier phases. It contains no PDF parsing, chunking,
embedding, ChromaDB, agent, or summarization logic of its own — all of
that stays in ingestion/, agent/, and services/, exactly as before.
"""

import re
import uuid
from pathlib import Path

import streamlit as st

from ingestion import EmbeddingError, PDFLoadError, VectorStoreError
from ingestion.pipeline import ingest_document
from services import (
    QAServiceError,
    SummaryServiceError,
    answer_question,
    summarize_document,
)

UPLOAD_DIR = Path(__file__).resolve().parent / "data" / "uploads"

_HEBREW_RE = re.compile(r"[֐-׿]")


def _looks_hebrew(text: str) -> bool:
    return bool(_HEBREW_RE.search(text))


def _text_align(text: str) -> str:
    return "right" if _looks_hebrew(text) else "left"


def _safe_filename(name: str) -> str:
    """
    Reduce an uploaded filename to a single safe path segment: strip any
    directory components the browser might send, then strip anything
    that isn't a "normal" filename character. The original name is kept
    wherever possible because Source Tracking displays it verbatim to
    the student.
    """
    base = Path(name).name  # drop any leading path (path-traversal guard)
    base = base.replace("\\", "_").replace("/", "_")
    base = re.sub(r'[<>:"|?*\x00-\x1f]', "_", base)
    base = base.strip(" .")
    return base or f"upload_{uuid.uuid4().hex[:8]}.pdf"


def _init_session_state() -> None:
    defaults = {
        "active_document_id": None,
        "active_filename": None,
        "active_page_count": None,
        "active_chunk_count": None,
        "processed": False,
        "qa_history": [],
        "summary_response": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _reset_document_state() -> None:
    """Called when a newly processed document differs from the active one."""
    st.session_state.qa_history = []
    st.session_state.summary_response = None


def _render_qa_response(resp) -> None:
    with st.container(border=True):
        st.markdown(f"**Q:** {resp.question}")
        st.markdown("**Answer**")
        st.markdown(resp.answer, text_alignment=_text_align(resp.answer))
        if resp.sources:
            st.markdown("**Sources**")
            for i, s in enumerate(resp.sources, start=1):
                st.write(f"{i}. {s.source_filename} — page {s.page_number}")
            with st.expander("Source details", icon=":material/info:"):
                for s in resp.sources:
                    st.caption(
                        f"chunk_index={s.chunk_index} · distance={s.distance:.4f}"
                    )


st.set_page_config(
    page_title="CourseMind",
    page_icon=":material/school:",
    layout="centered",
)

_init_session_state()

# ----------------------------- Sidebar -----------------------------
# The "active document" info is rendered into a placeholder (filled near
# the end of the script, see _render_sidebar_active_document() call
# below) rather than written directly here. Streamlit renders elements
# top-to-bottom in one pass per rerun, and the upload/process handler
# further down this script can change the active document during THIS
# SAME rerun — writing it here directly would show last run's value.
with st.sidebar:
    st.markdown("### CourseMind")
    st.caption("AI-powered study assistant")
    st.markdown(
        "**Features**\n"
        "- PDF course material\n"
        "- Grounded Q&A\n"
        "- Source Tracking\n"
        "- Full-document summaries"
    )
    sidebar_active_doc_slot = st.empty()


def _render_sidebar_active_document() -> None:
    if not st.session_state.processed:
        return
    with sidebar_active_doc_slot.container():
        st.markdown("**Active document**")
        st.write(st.session_state.active_filename)
        st.caption(
            f"{st.session_state.active_page_count} pages · "
            f"{st.session_state.active_chunk_count} chunks"
        )

# ----------------------------- Header -----------------------------
st.title("CourseMind", icon=":material/school:")
st.caption("AI Study Assistant for Your Course Material")
st.write(
    "For university and college students across academic fields — "
    "especially while preparing for exams."
)

# ----------------------------- Upload -----------------------------
st.subheader("1. Upload your course material", icon=":material/upload_file:")
st.caption("Only PDF files are supported in this version.")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    process_clicked = st.button(
        "Process document", icon=":material/play_arrow:", type="primary"
    )

    if process_clicked:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = _safe_filename(uploaded_file.name)
        save_path = UPLOAD_DIR / safe_name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Processing your document..."):
            try:
                result = ingest_document(str(save_path))
            except FileNotFoundError:
                st.error(
                    "CourseMind could not find the uploaded file. Please try again."
                )
            except PDFLoadError as e:
                # PDFLoadError's message includes the full server-side file
                # path (useful for local debugging) — replace it with just
                # the filename before showing it to the user.
                safe_message = str(e).replace(str(save_path), safe_name)
                st.error(f"This file could not be processed: {safe_message}")
            except EmbeddingError as e:
                st.error(f"CourseMind could not reach the AI service: {e}")
            except VectorStoreError:
                st.error(
                    "CourseMind could not save the processed document. "
                    "Please try again."
                )
            except Exception:
                st.error(
                    "CourseMind could not process this document because of an "
                    "unexpected technical problem. Please try again."
                )
            else:
                if not result["chunks"]:
                    st.warning(
                        "No readable text was found in this PDF. It may be a "
                        "scanned or image-only document, which isn't supported "
                        "yet — please try a different file."
                    )
                else:
                    new_document_id = result["chunks"][0].document_id
                    if (
                        st.session_state.active_document_id is not None
                        and st.session_state.active_document_id != new_document_id
                    ):
                        _reset_document_state()

                    st.session_state.active_document_id = new_document_id
                    st.session_state.active_filename = safe_name
                    st.session_state.active_page_count = len(result["pages"])
                    st.session_state.active_chunk_count = len(result["chunks"])
                    st.session_state.processed = True

                    st.success(
                        f"Document processed successfully.\n\n"
                        f"{len(result['pages'])} pages · "
                        f"{len(result['chunks'])} chunks"
                    )

if st.session_state.processed:
    st.info(
        f"Active document: **{st.session_state.active_filename}** — "
        f"{st.session_state.active_page_count} pages, "
        f"{st.session_state.active_chunk_count} chunks",
        icon=":material/description:",
    )

# ----------------------------- Q&A -----------------------------
st.subheader("Ask your course material", icon=":material/chat:")

if not st.session_state.processed:
    st.info("Upload and process a PDF first.")
else:
    with st.form("qa_form", border=False):
        question = st.text_input(
            "Your question",
            placeholder="e.g. מה זה משתנה בשפת JavaScript?",
        )
        ask_clicked = st.form_submit_button("Ask", icon=":material/send:")

    if ask_clicked:
        if not question or not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                try:
                    response = answer_question(
                        question, document_id=st.session_state.active_document_id
                    )
                except QAServiceError:
                    st.error(
                        "CourseMind could not answer the question because of a "
                        "technical problem. Please try again."
                    )
                else:
                    st.session_state.qa_history.insert(0, response)

    for resp in st.session_state.qa_history:
        _render_qa_response(resp)

# ----------------------------- Summary -----------------------------
st.subheader("Full study summary", icon=":material/summarize:")

if not st.session_state.processed:
    st.info("Upload and process a PDF first.")
else:
    st.caption(
        "Generates one structured summary covering the entire document — "
        "this can take a little longer than Q&A."
    )
    if st.button("Generate full summary", icon=":material/auto_awesome:"):
        with st.spinner("Generating a full study summary..."):
            try:
                st.session_state.summary_response = summarize_document(
                    st.session_state.active_document_id
                )
            except SummaryServiceError:
                st.error(
                    "CourseMind could not generate a summary because of a "
                    "technical problem. Please try again."
                )

    summary = st.session_state.summary_response
    if summary is not None:
        st.caption(
            f"Generated from {summary.total_pages} pages / "
            f"{summary.total_chunks} chunks"
        )
        st.markdown(summary.summary, text_alignment=_text_align(summary.summary))

# Filled last so it always reflects this rerun's final state, even when
# the upload/process handler above just changed the active document.
_render_sidebar_active_document()

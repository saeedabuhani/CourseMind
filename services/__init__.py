"""
CourseMind services package.

Backend services that sit between the CrewAI agent layer (agent/) and a
future UI (Streamlit, Phase 8+). No UI code lives here.
"""

from .qa_service import QAResponse, QAServiceError, SourceReference, answer_question
from .summary_service import SummaryResponse, SummaryServiceError, summarize_document

__all__ = [
    "answer_question",
    "QAResponse",
    "SourceReference",
    "QAServiceError",
    "summarize_document",
    "SummaryResponse",
    "SummaryServiceError",
]

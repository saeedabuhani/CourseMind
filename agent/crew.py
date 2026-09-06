"""
CourseMind CrewAI agent definition (Phase 6).

Wires together:
- one Agent (role/goal/backstory, incl. the Grounding Rule) — a single
  study-assistant agent, no multi-agent orchestration
- one Tool: CourseMaterialSearchTool (agent/tools/rag_retrieval_tool.py),
  itself a thin CrewAI wrapper around Phase 5's retrieve() — no embedding,
  Chroma, or threshold logic is duplicated here
- one reusable Task ("{question}" is filled at kickoff time — never a
  hardcoded question)
- one Crew (single agent, single task, sequential process)

Public interface: run_agent(question, document_id=None) -> AgentAnswer.
Nothing outside this module needs to import crewai directly.
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple

from dotenv import load_dotenv
from crewai import LLM, Agent, Crew, Process, Task

from agent.tools.rag_retrieval_tool import CourseMaterialSearchTool, RetrievalResult

load_dotenv()  # populate os.environ from a local .env file, if one exists

# Overridable via OPENAI_MODEL; matches .env.example's documented default.
DEFAULT_MODEL = "gpt-4o-mini"


class AgentError(RuntimeError):
    """
    Raised for CourseMind agent-layer errors: empty question, missing
    OPENAI_API_KEY, or a CrewAI execution failure.

    Messages here describe the failure only — never an API key's value.
    Retrieval-layer problems (RetrievalError/EmbeddingError/
    VectorStoreError) are caught inside the Tool itself (see
    CourseMaterialSearchTool._run) and surfaced to the Agent as text, not
    raised through here — see the Grounding Rule in TASK_DESCRIPTION.
    """


ROLE = "Personal AI Study Assistant"

GOAL = (
    "Help university and college students understand their own course "
    "material accurately by retrieving relevant evidence from the "
    "documents they uploaded."
)

BACKSTORY = (
    "You are a patient, clear, and careful private tutor. You prioritize "
    "the student's actual course material. Before answering questions "
    "about the course material, you retrieve relevant evidence from the "
    "indexed documents. If supporting information is not found, you "
    "clearly say so instead of inventing unsupported course content."
)

# Reusable across any question — {question} is filled by Crew.kickoff()'s
# `inputs`, never hardcoded to one specific question.
TASK_DESCRIPTION = (
    "A student has asked the following question about their uploaded "
    "course material:\n\n"
    '"{question}"\n\n'
    "Use the course material search tool to retrieve relevant evidence "
    "before answering — always search first for any question about the "
    "course content.\n\n"
    "Grounding Rule:\n"
    "- If the tool returns relevant evidence, base your answer only on "
    "that evidence.\n"
    "- If the tool reports that no relevant evidence was found, "
    "explicitly tell the student the information was not found in their "
    "uploaded course material. Do not fabricate course-specific "
    "information and do not silently fall back on general knowledge."
)

TASK_EXPECTED_OUTPUT = (
    "A clear, concise answer grounded in the retrieved course material, "
    "or an explicit statement that the information was not found in the "
    "uploaded material."
)


def _get_llm() -> LLM:
    if not os.getenv("OPENAI_API_KEY"):
        raise AgentError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    return LLM(model=model, temperature=0.2)


def build_crew(
    document_id: Optional[str] = None, verbose: bool = True
) -> Tuple[Crew, CourseMaterialSearchTool]:
    """
    Build one Agent + one Task + one Crew for a single run.

    `document_id`, when given, is baked into the tool instance itself
    (not exposed as an LLM-controlled tool argument), so retrieval for
    this run can never search outside the student's currently selected
    document — Phase 5's document isolation stays intact under agent
    control.

    Returns:
        (crew, search_tool) — the tool instance is returned so the
        caller can inspect `search_tool.last_result` (Source Tracking
        metadata) and `search_tool.current_usage_count` (proof the tool
        was actually invoked) after `crew.kickoff()`.

    Raises:
        AgentError: OPENAI_API_KEY is not set.
    """
    search_tool = CourseMaterialSearchTool(document_id=document_id)
    llm = _get_llm()

    study_assistant = Agent(
        role=ROLE,
        goal=GOAL,
        backstory=BACKSTORY,
        tools=[search_tool],
        llm=llm,
        verbose=verbose,
    )

    answer_question_task = Task(
        description=TASK_DESCRIPTION,
        expected_output=TASK_EXPECTED_OUTPUT,
        agent=study_assistant,
    )

    crew = Crew(
        agents=[study_assistant],
        tasks=[answer_question_task],
        process=Process.sequential,
        verbose=False,
        # Avoids CrewAI's interactive "view execution traces?" first-run
        # prompt (which otherwise blocks for up to 20s waiting for input
        # that will never come in a non-interactive/automated run).
        tracing=False,
    )
    return crew, search_tool


@dataclass
class AgentAnswer:
    """
    Public result of run_agent(). Keeps the Agent's free-text answer
    separate from the structured evidence that produced it, so later
    phases can build a properly-cited response without re-parsing text.
    """

    question: str
    document_id: Optional[str]
    answer_text: str
    retrieval: Optional[RetrievalResult]  # None only if the tool was never called
    tool_was_used: bool


def run_agent(
    question: str, document_id: Optional[str] = None, verbose: bool = True
) -> AgentAnswer:
    """
    Run the CourseMind Agent on one question, optionally scoped to one
    uploaded document.

    Args:
        question: the student's natural-language question. Must be a
            non-empty, non-whitespace-only string.
        document_id: if given, retrieval stays scoped to that document
            (Phase 5's document isolation).
        verbose: print the Agent's reasoning/tool-call trace to stdout
            (useful for demos and debugging; set False for quiet runs,
            e.g. inside a future Streamlit app).

    Returns:
        AgentAnswer: the Agent's text answer, the structured
        RetrievalResult the Tool produced (for Source Tracking — None
        only if the Agent never called the tool at all), and whether the
        tool was used.

    Raises:
        AgentError: question is empty/whitespace, OPENAI_API_KEY is
            missing, or the underlying CrewAI run itself failed (e.g. the
            Agent's own chat completion call errored).
    """
    if not question or not question.strip():
        raise AgentError("question must be a non-empty string")

    crew, search_tool = build_crew(document_id=document_id, verbose=verbose)

    try:
        crew_output = crew.kickoff(inputs={"question": question})
    except Exception as e:
        raise AgentError(f"Agent execution failed: {e}") from e

    return AgentAnswer(
        question=question,
        document_id=document_id,
        answer_text=str(crew_output),
        retrieval=search_tool.last_result,
        tool_was_used=search_tool.current_usage_count > 0,
    )

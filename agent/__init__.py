"""CourseMind agent package: the CrewAI agent and the tools it can call."""

from .crew import AgentAnswer, AgentError, run_agent

__all__ = ["run_agent", "AgentAnswer", "AgentError"]

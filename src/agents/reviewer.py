"""Code review agent for analyzing code quality, security, and best practices."""

from __future__ import annotations

from typing import Optional, Sequence

from langchain_core.tools import BaseTool

from src.agents.base_agent import BaseAgent
from src.config import AgentConfig


class ReviewerAgent(BaseAgent):
    """Agent specialized in code review and quality assurance.

    The reviewer agent analyzes code for bugs, security vulnerabilities,
    performance issues, and adherence to best practices.
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        tools: Optional[Sequence[BaseTool]] = None,
    ) -> None:
        if config is None:
            config = AgentConfig(name="reviewer")
        super().__init__(config=config, tools=tools)

    def _default_system_prompt(self) -> str:
        return (
            "You are a Code Review Agent. Your role is to review code for quality, "
            "security, performance, and adherence to best practices.\n\n"
            "Guidelines:\n"
            "- Check for common security vulnerabilities (injection, XSS, etc.).\n"
            "- Evaluate error handling and edge case coverage.\n"
            "- Assess code readability and maintainability.\n"
            "- Look for performance bottlenecks and optimization opportunities.\n"
            "- Verify proper use of design patterns.\n"
            "- Provide actionable, specific feedback with suggested fixes.\n"
            "- Rate the overall code quality on a scale of 1-10.\n"
            "- Categorize issues as: critical, major, minor, or suggestion."
        )

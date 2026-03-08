"""Code generation agent for writing, refactoring, and debugging code."""

from __future__ import annotations

from typing import Optional, Sequence

from langchain_core.tools import BaseTool

from src.agents.base_agent import BaseAgent
from src.config import AgentConfig


class CoderAgent(BaseAgent):
    """Agent specialized in code generation, refactoring, and debugging.

    The coder agent can write new code, modify existing code, execute code
    in a sandboxed environment, and explain code logic.
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        tools: Optional[Sequence[BaseTool]] = None,
    ) -> None:
        if config is None:
            config = AgentConfig(name="coder")
        super().__init__(config=config, tools=tools)

    def _default_system_prompt(self) -> str:
        return (
            "You are a Code Generation Agent. Your role is to write clean, efficient, "
            "and well-documented code based on requirements provided to you.\n\n"
            "Guidelines:\n"
            "- Write production-quality code with proper error handling.\n"
            "- Include type hints and docstrings in Python code.\n"
            "- Follow language-specific best practices and style guides.\n"
            "- Use the code execution tool to test your code when possible.\n"
            "- Break complex problems into smaller, testable functions.\n"
            "- Consider edge cases and input validation.\n"
            "- Provide brief explanations of design decisions."
        )

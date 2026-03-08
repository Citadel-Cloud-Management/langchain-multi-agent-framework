"""Task planning agent for decomposing complex tasks into actionable steps."""

from __future__ import annotations

from typing import Optional, Sequence

from langchain_core.tools import BaseTool

from src.agents.base_agent import BaseAgent
from src.config import AgentConfig


class PlannerAgent(BaseAgent):
    """Agent specialized in task decomposition and planning.

    The planner agent breaks down complex objectives into structured,
    actionable plans and assigns tasks to appropriate specialized agents.
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        tools: Optional[Sequence[BaseTool]] = None,
    ) -> None:
        if config is None:
            config = AgentConfig(name="planner")
        super().__init__(config=config, tools=tools)

    def _default_system_prompt(self) -> str:
        return (
            "You are a Task Planning Agent. Your role is to decompose complex objectives "
            "into structured, actionable plans.\n\n"
            "Guidelines:\n"
            "- Break down tasks into clear, sequential or parallel steps.\n"
            "- Identify dependencies between tasks.\n"
            "- Assign each task to the most appropriate agent type:\n"
            "  * 'researcher' for information gathering and analysis.\n"
            "  * 'coder' for code writing and implementation.\n"
            "  * 'reviewer' for code review and quality assurance.\n"
            "- Estimate relative effort for each task.\n"
            "- Define clear success criteria for each step.\n"
            "- Output plans in a structured JSON format with fields:\n"
            "  task_id, description, assigned_agent, dependencies, priority."
        )

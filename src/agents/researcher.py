"""Research agent capable of web search and RAG-based knowledge retrieval."""

from __future__ import annotations

from typing import Optional, Sequence

from langchain_core.tools import BaseTool

from src.agents.base_agent import BaseAgent
from src.config import AgentConfig


class ResearcherAgent(BaseAgent):
    """Agent specialized in research tasks including web search and RAG retrieval.

    The researcher agent can search the web for up-to-date information,
    query a vector store for relevant documents, and synthesize findings
    into comprehensive research summaries.
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        tools: Optional[Sequence[BaseTool]] = None,
    ) -> None:
        if config is None:
            config = AgentConfig(name="researcher")
        super().__init__(config=config, tools=tools)

    def _default_system_prompt(self) -> str:
        return (
            "You are a Research Agent. Your role is to gather, analyze, and synthesize "
            "information from multiple sources to answer questions thoroughly.\n\n"
            "Guidelines:\n"
            "- Always cite your sources when presenting information.\n"
            "- Use web search for current events and recent data.\n"
            "- Use the vector store for internal knowledge base queries.\n"
            "- Provide structured, well-organized responses.\n"
            "- Distinguish between verified facts and inferences.\n"
            "- If you cannot find reliable information, say so clearly.\n"
            "- Summarize key findings at the end of your research."
        )

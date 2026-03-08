"""Base agent class that all specialized agents inherit from."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from src.config import AgentConfig, LLMConfig, LLMProvider


def create_llm(config: LLMConfig) -> BaseChatModel:
    """Factory function to create the appropriate LLM instance.

    Args:
        config: LLM configuration specifying provider, model, and parameters.

    Returns:
        A configured chat model instance.
    """
    if config.provider == LLMProvider.OPENAI:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=config.api_key,
        )
    elif config.provider == LLMProvider.ANTHROPIC:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=config.api_key,
        )
    elif config.provider == LLMProvider.AZURE_OPENAI:
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=config.api_key,
            azure_endpoint=config.api_base or "",
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {config.provider}")


class BaseAgent(ABC):
    """Abstract base class for all agents in the framework.

    Each agent wraps an LLM with a specific system prompt and set of tools,
    enabling it to perform a specialized role within the multi-agent workflow.
    """

    def __init__(
        self,
        config: AgentConfig,
        tools: Optional[Sequence[BaseTool]] = None,
    ) -> None:
        self.config = config
        self.name = config.name
        self.llm = create_llm(config.llm)
        self.tools: list[BaseTool] = list(tools) if tools else []
        self.system_prompt = config.system_prompt or self._default_system_prompt()

        if self.tools:
            self.llm_with_tools = self.llm.bind_tools(self.tools)
        else:
            self.llm_with_tools = self.llm

    @abstractmethod
    def _default_system_prompt(self) -> str:
        """Return the default system prompt for this agent type."""
        ...

    def invoke(self, messages: list[BaseMessage], **kwargs: Any) -> AIMessage:
        """Invoke the agent with a list of messages.

        Args:
            messages: Conversation messages to process.
            **kwargs: Additional keyword arguments passed to the LLM.

        Returns:
            The agent's response as an AIMessage.
        """
        full_messages = [SystemMessage(content=self.system_prompt)] + messages
        response = self.llm_with_tools.invoke(full_messages, **kwargs)
        return response  # type: ignore[return-value]

    async def ainvoke(self, messages: list[BaseMessage], **kwargs: Any) -> AIMessage:
        """Async version of invoke.

        Args:
            messages: Conversation messages to process.
            **kwargs: Additional keyword arguments passed to the LLM.

        Returns:
            The agent's response as an AIMessage.
        """
        full_messages = [SystemMessage(content=self.system_prompt)] + messages
        response = await self.llm_with_tools.ainvoke(full_messages, **kwargs)
        return response  # type: ignore[return-value]

    def get_tool_names(self) -> list[str]:
        """Return the names of all tools available to this agent."""
        return [tool.name for tool in self.tools]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, tools={self.get_tool_names()})"

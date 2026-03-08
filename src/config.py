"""Configuration management for the multi-agent framework."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"


class VectorStoreBackend(str, Enum):
    """Supported vector store backends."""

    CHROMADB = "chromadb"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"


@dataclass
class LLMConfig:
    """Configuration for the LLM provider."""

    provider: LLMProvider = LLMProvider.OPENAI
    model: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 4096
    api_key: Optional[str] = None
    api_base: Optional[str] = None

    def __post_init__(self) -> None:
        if self.api_key is None:
            env_map = {
                LLMProvider.OPENAI: "OPENAI_API_KEY",
                LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
                LLMProvider.AZURE_OPENAI: "AZURE_OPENAI_API_KEY",
            }
            self.api_key = os.environ.get(env_map.get(self.provider, ""), "")


@dataclass
class VectorStoreConfig:
    """Configuration for the vector store."""

    backend: VectorStoreBackend = VectorStoreBackend.CHROMADB
    collection_name: str = "agent_knowledge"
    persist_directory: str = "./data/vectorstore"
    embedding_model: str = "text-embedding-3-small"


@dataclass
class AgentConfig:
    """Configuration for an individual agent."""

    name: str = "default_agent"
    llm: LLMConfig = field(default_factory=LLMConfig)
    max_iterations: int = 10
    verbose: bool = True
    tools: list[str] = field(default_factory=list)
    system_prompt: str = ""


@dataclass
class OrchestratorConfig:
    """Top-level orchestrator configuration."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    agents: dict[str, AgentConfig] = field(default_factory=dict)
    max_concurrent_agents: int = 3
    default_timeout_seconds: int = 300
    enable_shared_memory: bool = True
    recursion_limit: int = 50

    @classmethod
    def from_env(cls) -> "OrchestratorConfig":
        """Create configuration from environment variables."""
        llm_config = LLMConfig(
            provider=LLMProvider(os.environ.get("LLM_PROVIDER", "openai")),
            model=os.environ.get("LLM_MODEL", "gpt-4o"),
            temperature=float(os.environ.get("LLM_TEMPERATURE", "0.0")),
            max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "4096")),
        )
        return cls(llm=llm_config)

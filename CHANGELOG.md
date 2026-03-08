# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-07

### Added

- Initial release of the multi-agent orchestration framework.
- LangGraph-based StateGraph orchestrator with supervisor routing pattern.
- Specialized agents: Planner, Researcher, Coder, Reviewer.
- Tool implementations: Web Search (Tavily/SerpAPI/DuckDuckGo), Code Executor (sandboxed), Vector Store (ChromaDB/Pinecone/Weaviate).
- Thread-safe shared memory system for inter-agent communication.
- Support for OpenAI, Anthropic, and Azure OpenAI LLM providers.
- Configuration management via dataclasses and environment variables.
- Example workflows: research team, dev team, customer support.

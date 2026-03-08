"""Example: Software development team with planner, coder, and reviewer agents.

This example simulates a development workflow where:
1. The planner breaks down a feature request into tasks.
2. The coder implements the solution.
3. The reviewer checks code quality and suggests improvements.
"""

from src.config import LLMConfig, LLMProvider, OrchestratorConfig
from src.orchestrator import MultiAgentOrchestrator
from src.tools.code_executor import CodeExecutorTool


def main() -> None:
    code_tool = CodeExecutorTool(restricted_mode=True)

    config = OrchestratorConfig(
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4o",
            temperature=0.0,
        ),
        recursion_limit=30,
    )

    orchestrator = MultiAgentOrchestrator(config)
    orchestrator.agents["coder"].tools = [code_tool]
    orchestrator.agents["coder"].llm_with_tools = (
        orchestrator.agents["coder"].llm.bind_tools([code_tool])
    )

    task = (
        "Build a Python module that implements a thread-safe LRU cache with TTL support. "
        "Requirements:\n"
        "- Generic type support with TypeVar\n"
        "- Configurable max size and TTL per entry\n"
        "- get(), put(), delete(), and clear() methods\n"
        "- Statistics tracking (hits, misses, evictions)\n"
        "- Include comprehensive unit tests"
    )

    print(f"Starting development task: {task}\n")
    result = orchestrator.run(task)
    print("=" * 60)
    print("DEVELOPMENT RESULT")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()

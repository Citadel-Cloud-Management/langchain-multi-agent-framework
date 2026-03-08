"""Example: Multi-agent research team for comprehensive topic analysis.

This example demonstrates how to set up a research team that uses
the planner and researcher agents to investigate a topic and produce
a structured research report.
"""

from src.config import LLMConfig, LLMProvider, OrchestratorConfig
from src.orchestrator import MultiAgentOrchestrator


def main() -> None:
    config = OrchestratorConfig(
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4o",
            temperature=0.1,
        ),
        max_concurrent_agents=2,
        recursion_limit=20,
    )

    orchestrator = MultiAgentOrchestrator(config)

    task = (
        "Research the current state of quantum computing in 2026. "
        "Cover the major players, recent breakthroughs, practical applications, "
        "and expected timeline for quantum advantage in different industries. "
        "Produce a structured report with sections and citations."
    )

    print(f"Starting research task: {task}\n")
    result = orchestrator.run(task)
    print("=" * 60)
    print("RESEARCH REPORT")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()

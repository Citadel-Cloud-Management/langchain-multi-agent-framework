"""Example: Customer support agents with knowledge base integration.

This example sets up a support workflow where a researcher agent
queries a vector store knowledge base and a planner agent coordinates
the response strategy for complex support tickets.
"""

from langchain_core.documents import Document

from src.config import (
    LLMConfig,
    LLMProvider,
    OrchestratorConfig,
    VectorStoreConfig,
)
from src.orchestrator import MultiAgentOrchestrator
from src.tools.vector_store import VectorStoreTool
from src.tools.web_search import WebSearchTool


def seed_knowledge_base(vs_tool: VectorStoreTool) -> None:
    """Populate the vector store with sample support documentation."""
    docs = [
        Document(
            page_content=(
                "Password Reset: Users can reset passwords via the Settings > Security page. "
                "An email with a reset link is sent within 2 minutes. Links expire after 24 hours. "
                "If the user does not receive the email, check spam folders and verify the email "
                "address on file."
            ),
            metadata={"source": "support_docs/password_reset.md", "category": "authentication"},
        ),
        Document(
            page_content=(
                "Billing FAQ: Subscriptions are billed monthly on the anniversary of sign-up. "
                "Refunds are available within 14 days of charge. Enterprise plans have custom "
                "billing cycles. Contact billing@example.com for invoice disputes."
            ),
            metadata={"source": "support_docs/billing_faq.md", "category": "billing"},
        ),
        Document(
            page_content=(
                "API Rate Limits: Free tier allows 100 requests/minute, Pro tier allows "
                "1000 requests/minute, Enterprise has custom limits. Rate limit headers are "
                "returned with every response: X-RateLimit-Remaining, X-RateLimit-Reset."
            ),
            metadata={"source": "support_docs/api_rate_limits.md", "category": "api"},
        ),
    ]
    vs_tool.add_documents(docs)
    print(f"Loaded {len(docs)} documents into knowledge base.")


def main() -> None:
    vs_config = VectorStoreConfig(
        collection_name="support_kb",
        persist_directory="./data/support_vectorstore",
    )
    vs_tool = VectorStoreTool(config=vs_config)
    web_tool = WebSearchTool()

    seed_knowledge_base(vs_tool)

    config = OrchestratorConfig(
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4o",
            temperature=0.1,
        ),
        vector_store=vs_config,
        recursion_limit=15,
    )

    orchestrator = MultiAgentOrchestrator(config)
    orchestrator.agents["researcher"].tools = [vs_tool, web_tool]
    orchestrator.agents["researcher"].llm_with_tools = (
        orchestrator.agents["researcher"].llm.bind_tools([vs_tool, web_tool])
    )

    ticket = (
        "Customer reports: 'I upgraded to Pro plan yesterday but I am still "
        "getting rate limited at 100 requests/minute. I also haven't received "
        "the billing confirmation email. Please help resolve both issues.'"
    )

    print(f"Processing support ticket:\n{ticket}\n")
    result = orchestrator.run(ticket)
    print("=" * 60)
    print("SUPPORT RESPONSE")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()

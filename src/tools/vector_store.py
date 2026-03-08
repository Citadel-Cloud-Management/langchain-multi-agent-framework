"""Vector store tool for RAG-based knowledge retrieval."""

from __future__ import annotations

from typing import Any, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.config import VectorStoreBackend, VectorStoreConfig


class VectorStoreQueryInput(BaseModel):
    """Input schema for vector store queries."""

    query: str = Field(description="The search query for semantic retrieval.")
    top_k: int = Field(default=5, description="Number of results to retrieve.")


class VectorStoreTool(BaseTool):
    """Tool for querying a vector store for semantically similar documents.

    Supports ChromaDB, Pinecone, and Weaviate backends. Documents can be
    ingested ahead of time and queried during agent execution.
    """

    name: str = "vector_store_search"
    description: str = (
        "Search the internal knowledge base for relevant documents. "
        "Use this when you need information from project documentation, "
        "past conversations, or uploaded reference materials."
    )
    args_schema: Type[BaseModel] = VectorStoreQueryInput
    config: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    _vectorstore: Any = None

    def model_post_init(self, __context: Any) -> None:
        """Initialize the vector store backend after model creation."""
        self._vectorstore = self._create_vectorstore()

    def _create_vectorstore(self) -> Any:
        """Create the appropriate vector store instance based on config."""
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(model=self.config.embedding_model)

        if self.config.backend == VectorStoreBackend.CHROMADB:
            from langchain_chroma import Chroma

            return Chroma(
                collection_name=self.config.collection_name,
                embedding_function=embeddings,
                persist_directory=self.config.persist_directory,
            )
        elif self.config.backend == VectorStoreBackend.PINECONE:
            from langchain_pinecone import PineconeVectorStore

            return PineconeVectorStore(
                index_name=self.config.collection_name,
                embedding=embeddings,
            )
        elif self.config.backend == VectorStoreBackend.WEAVIATE:
            from langchain_weaviate import WeaviateVectorStore

            return WeaviateVectorStore(
                index_name=self.config.collection_name,
                embedding=embeddings,
                text_key="text",
            )
        else:
            raise ValueError(f"Unsupported backend: {self.config.backend}")

    def _run(
        self,
        query: str,
        top_k: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Query the vector store for relevant documents.

        Args:
            query: Semantic search query.
            top_k: Number of results to return.
            run_manager: Optional callback manager.

        Returns:
            Formatted string of matching documents with metadata.
        """
        if self._vectorstore is None:
            return "Error: Vector store not initialized."

        docs = self._vectorstore.similarity_search(query, k=top_k)
        if not docs:
            return "No relevant documents found."

        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            results.append(
                f"[{i}] Source: {source}\n{doc.page_content[:500]}"
            )
        return "\n\n---\n\n".join(results)

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents to the vector store.

        Args:
            documents: List of LangChain Document objects to ingest.

        Returns:
            List of document IDs.
        """
        if self._vectorstore is None:
            raise RuntimeError("Vector store not initialized.")
        return self._vectorstore.add_documents(documents)

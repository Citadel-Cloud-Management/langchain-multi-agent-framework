"""Web search tool wrapper supporting multiple search providers."""

from __future__ import annotations

import os
from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class WebSearchInput(BaseModel):
    """Input schema for the web search tool."""

    query: str = Field(description="The search query string.")
    num_results: int = Field(default=5, description="Number of results to return.")


class WebSearchTool(BaseTool):
    """Tool for performing web searches using Tavily, SerpAPI, or DuckDuckGo.

    The tool attempts to use Tavily first, falls back to SerpAPI,
    and finally to DuckDuckGo if no API keys are configured.
    """

    name: str = "web_search"
    description: str = (
        "Search the web for current information. Use this when you need "
        "up-to-date facts, recent events, or information not in your training data."
    )
    args_schema: Type[BaseModel] = WebSearchInput

    def _run(
        self,
        query: str,
        num_results: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute a web search and return formatted results.

        Args:
            query: The search query.
            num_results: Maximum number of results to return.
            run_manager: Optional callback manager.

        Returns:
            Formatted search results as a string.
        """
        tavily_key = os.environ.get("TAVILY_API_KEY")
        serpapi_key = os.environ.get("SERPAPI_API_KEY")

        if tavily_key:
            return self._search_tavily(query, num_results, tavily_key)
        elif serpapi_key:
            return self._search_serpapi(query, num_results, serpapi_key)
        else:
            return self._search_duckduckgo(query, num_results)

    def _search_tavily(self, query: str, num_results: int, api_key: str) -> str:
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, max_results=num_results)
        results = []
        for item in response.get("results", []):
            results.append(
                f"Title: {item['title']}\nURL: {item['url']}\n"
                f"Content: {item['content']}\n"
            )
        return "\n---\n".join(results) if results else "No results found."

    def _search_serpapi(self, query: str, num_results: int, api_key: str) -> str:
        from serpapi import GoogleSearch

        params = {
            "q": query,
            "num": num_results,
            "api_key": api_key,
            "engine": "google",
        }
        search = GoogleSearch(params)
        data = search.get_dict()
        results = []
        for item in data.get("organic_results", [])[:num_results]:
            results.append(
                f"Title: {item.get('title', '')}\nURL: {item.get('link', '')}\n"
                f"Snippet: {item.get('snippet', '')}\n"
            )
        return "\n---\n".join(results) if results else "No results found."

    def _search_duckduckgo(self, query: str, num_results: int) -> str:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results_list = list(ddgs.text(query, max_results=num_results))

        results = []
        for item in results_list:
            results.append(
                f"Title: {item.get('title', '')}\nURL: {item.get('href', '')}\n"
                f"Body: {item.get('body', '')}\n"
            )
        return "\n---\n".join(results) if results else "No results found."

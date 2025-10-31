"""
SearXNG search interface.

This module provides an interface to SearXNG search engine for fetching
article content and other search results asynchronously.
"""

from __future__ import annotations

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import aiohttp
from dotenv import load_dotenv

from sentiment_analysis.utils import setup_logging

# Configure logging
logger = setup_logging(__name__)


async def fetch_search_results(
    session: aiohttp.ClientSession, base_url: str, query: str, category: str | None
) -> list[dict[str, Any]]:
    """
    Fetches search results for a single query asynchronously.

    Args:
        session: The aiohttp session to use for the request.
        base_url: The base URL for the SearXNG instance.
        query: The search query.
        category: The category of the search query.

    Returns:
        A list of search result dictionaries.

    Raises:
        Exception: If the request to SearXNG fails.
    """
    query_params = {
        "q": query,
        "safesearch": "0",
        "format": "json",
        "language": "en",
    }

    if category:
        query_params["categories"] = category

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    url = f"{base_url}/search"

    async with session.get(url, params=query_params, headers=headers) as response:

        if response.status != 200:
            response_text = await response.text()
            raise Exception(
                f"Failed to fetch search results for query '{query}': {response.status} {response.reason}"
            )

        data = await response.json()
        results = data.get("results", [])

        # Add the query to each result
        for result in results:
            result["query"] = query

        return results


def process_search_results(
    all_results: list[dict[str, Any]],
    category: str | None = None,
    max_results: int = 10,
) -> list[dict[str, Any]]:
    """
    Processes and filters search results.

    Args:
        all_results: List of all search result dictionaries.
        category: Optional category to filter results by.
        max_results: Maximum number of results to return.

    Returns:
        Filtered and processed list of search results.
    """
    # Sort the combined results by score in descending order
    sorted_results = sorted(all_results, key=lambda x: x.get("score", 0), reverse=True)

    # Remove duplicates while preserving order
    seen_urls = set()
    unique_results = []
    for result in sorted_results:
        if (
            "content" not in result
            or "title" not in result
            or "url" not in result
            or "query" not in result
        ):
            continue
        if result["url"] not in seen_urls:
            unique_results.append(result)
            if "metadata" in result:
                result["metadata"] = result["metadata"]
            if result.get("publishedDate"):
                result["timestamp"] = result["publishedDate"]
            seen_urls.add(result["url"])

    # Filter results to include only those with the correct category if it is set
    if category:
        filtered_results = [
            result for result in unique_results if result.get("category") == category
        ]
    else:
        filtered_results = unique_results

    return filtered_results[:max_results]


async def searxng_search_async(
    queries: list[str],
    base_url: str,
    category: str | None = None,
    max_results: int = 10,
) -> dict[str, Any]:
    """
    Runs SearXNG search asynchronously with the given parameters.

    Args:
        queries: List of search queries.
        base_url: The base URL for the SearXNG instance.
        category: Optional category to search in.
        max_results: Maximum number of results to return.

    Returns:
        Search results in dictionary format.

    Raises:
        Exception: If the request to SearXNG fails.
    """
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_search_results(session, base_url, query, category)
            for query in queries
        ]
        results = await asyncio.gather(*tasks)

    all_results = [item for sublist in results for item in sublist]

    # Process and filter results
    final_results = process_search_results(all_results, category, max_results)

    return {
        "results": [
            {
                "title": result["title"],
                "content": result.get("content"),
                "url": result["url"],
                "timestamp": result.get("publishedDate", None),
                "metadata": result.get("metadata", None),
                "query": result["query"],
            }
            for result in final_results
        ],
        "category": category,
    }


def searxng_search(
    queries: list[str],
    base_url: str,
    category: str | None = None,
    max_results: int = 10,
) -> dict[str, Any]:
    """
    Runs SearXNG search synchronously with the given parameters.

    This method creates an event loop in a separate thread to run the asynchronous operations.

    Args:
        queries: List of search queries.
        base_url: The base URL for the SearXNG instance.
        category: Optional category to search in.
        max_results: Maximum number of results to return.

    Returns:
        Search results in dictionary format.

    Raises:
        Exception: If the request to SearXNG fails.
    """
    load_dotenv()
    final_base_url = base_url or os.getenv("SEARXNG_BASE_URL", "http://localhost:8080")

    with ThreadPoolExecutor() as executor:
        return executor.submit(
            asyncio.run,
            searxng_search_async(queries, final_base_url, category, max_results),
        ).result()


# Define the public API for this module
__all__ = [
    "fetch_search_results",
    "process_search_results",
    "searxng_search",
    "searxng_search_async",
]


if __name__ == "__main__":
    # Example usage
    results = searxng_search(
        queries=["weather in paris", "what is paris known for"], max_results=2
    )

    logger.debug(f"Search results: {json.dumps(results, indent=2)}")

#!/usr/bin/env python3
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
from typing import Any, cast

import aiohttp
from dotenv import load_dotenv

from sentiment_analysis.logging_utils import setup_logging

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
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
        "Accept": "text/html,application/json,text/plain,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "DNT": "1",
        # "Referer": "https://fairsuch.net/",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-User": "?1",
        "Sec-Fetch-Dest": "document",
    }

    url = f"{base_url}/search"

    async with session.get(url, params=query_params, headers=headers) as response:

        if response.status != 200:
            raise Exception(
                f"Failed to fetch search results for query '{query}': {response.status} {response.reason}"
            )

        data = await response.json()
        results = cast("list", data.get("results", []))

        # Add the query to each result
        for result in results:
            if isinstance(result, dict):
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
    base_url: str | None = None,
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
    final_base_url = cast("str", final_base_url)

    with ThreadPoolExecutor() as executor:
        return executor.submit(
            asyncio.run,
            searxng_search_async(queries, final_base_url, category, max_results),
        ).result()


def smart_searxng_search(
    queries: list[str],
    category: str | None = None,
    max_results: int = 10,
    max_cycles: int = 2,
    base_url: str | None = None,
    fallback_instances: list[str] | None = None,
) -> dict[str, Any]:
    """Search using SearXNG with intelligent instance cycling for rate limit handling.

    Tries local instance first, then cycles through fallback instances when
    rate limits are encountered. Provides robust fallback behavior for reliable
    content fetching.

    Args:
        queries: List of search queries to execute
        category: Optional category to search in (news, images, videos, etc.)
        max_results: Maximum number of results per query
        max_cycles: Number of complete cycles through all instances before giving up
        fallback_instances: List of fallback instances (uses defaults if None)

    Returns:
        Dictionary containing search results and metadata about which instance
        succeeded and cycling behavior:
        {
            "results": [...],           # Search results (same format as searxng_search)
            "category": "news",         # Search category
            "instance_used": "https://...",  # Which instance succeeded
            "cycles_used": 1,           # Number of cycles needed
            "total_attempts": 3,         # Total attempts made
            "fallback_used": True        # Whether fallback was needed
        }
    """
    load_dotenv()
    logger.info(f"Starting smart SearXNG search: {len(queries)} queries, max_cycles={max_cycles}")

    # Define instance priority list
    local_instance = base_url or os.getenv("SEARXNG_BASE_URL", "http://localhost:8080")
    default_fallbacks = [
        "https://searx.perennialte.ch/",
        "https://s.mble.dk/",
        "https://searx.sev.monster/",
        "https://searxng.canine.tools/",
        "https://etsi.me/",
    ]

    instances = [local_instance] + (fallback_instances or default_fallbacks)
    total_attempts = 0

    # Cycle through instances with rate limit handling
    for cycle in range(max_cycles):
        logger.info(f"Starting cycle {cycle + 1}/{max_cycles}")

        for instance_url in instances:
            total_attempts += 1
            logger.debug(f"Attempt {total_attempts}: Trying {instance_url}")

            try:
                # Use the existing searxng_search function
                result = searxng_search(
                    queries=queries,
                    base_url=instance_url,
                    category=category,
                    max_results=max_results,
                )

                # Check if we got valid results
                if result and result.get("results"):
                    logger.info(f"Success with {instance_url} (cycle {cycle + 1}, attempt {total_attempts})")
                    return {
                        **result,
                        "instance_used": instance_url,
                        "cycles_used": cycle + 1,
                        "total_attempts": total_attempts,
                        "fallback_used": instance_url != local_instance,
                    }
                logger.warning(f"No results returned from {instance_url}")

            except Exception as e:
                error_str = str(e).lower()

                # Check for rate limiting
                if "429" in error_str or "too many requests" in error_str:
                    logger.warning(f"Rate limited by {instance_url}, trying next instance")
                    continue

                # Check for other HTTP errors
                if "timeout" in error_str:
                    logger.warning(f"Timeout from {instance_url}, trying next instance")
                    continue

                if "connection" in error_str or "refused" in error_str:
                    logger.warning(f"Connection error from {instance_url}, trying next instance")
                    continue

                if "403" in error_str or "forbidden" in error_str:
                    logger.warning(f"Forbidden by {instance_url}, trying next instance")
                    continue

                logger.warning(f"Error with {instance_url}: {str(e)[:100]}...")
                continue

    # If we get here, all instances failed after all cycles
    logger.error(f"All instances failed after {max_cycles} cycles ({total_attempts} total attempts)")

    return {
        "results": [],
        "category": category,
        "instance_used": None,
        "cycles_used": max_cycles,
        "total_attempts": total_attempts,
        "fallback_used": False,
    }


# Define the public API for this module
__all__ = [
    "fetch_search_results",
    "process_search_results",
    "searxng_search",
    "searxng_search_async",
    "smart_searxng_search",
]


if __name__ == "__main__":
    # Example usage
    start_time = asyncio.get_event_loop().time()
    results = smart_searxng_search(queries=["Bitcoin Bears See More Peril After $300 Billion Crypto Selloff (BTC) - Bloomberg.com"], max_results=2)
    elapsed_time = asyncio.get_event_loop().time() - start_time

    logger.info(f"Search results ({elapsed_time:.2f}s): {json.dumps(results, indent=2)}")

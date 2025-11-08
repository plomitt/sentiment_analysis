"""
RSS news fetcher for Bitcoin news articles.

This module fetches Bitcoin news articles from Google RSS feeds and optionally
fetches article content using SearXNG search engine.
"""

from __future__ import annotations

import argparse
import calendar
import os
import sys
import time
from typing import Any, Dict, List

import feedparser

from sentiment_analysis.searxng_search import searxng_search, smart_searxng_search
from sentiment_analysis.utils import (
    make_timestamped_filename,
    save_json_data,
    setup_logging,
)

# Set up logging
logger = setup_logging(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for news fetching.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Fetch Bitcoin news and save to JSON")
    parser.add_argument(
        "--query", default="bitcoin", help="Search query (default: bitcoin)"
    )
    parser.add_argument(
        "--count", type=int, default=10, help="Number of articles to save (default: 10)"
    )
    parser.add_argument(
        "--searxng-url",
        help="SearXNG instance URL (default: from SEARXNG_BASE_URL env var or http://localhost:8080)",
    )
    parser.add_argument(
        "--no-content",
        action="store_true",
        help="Skip fetching article content from SearXNG",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.0,
        help="Delay between SearXNG requests in seconds (default: 0.0)",
    )
    args = parser.parse_args()
    return args


def fetch_article_body_content(title: str, searxng_url: str | None = None) -> str | None:
    try:
        logger.info(f"Fetching content for: {title[:50]}...")
        search_results = smart_searxng_search(
            queries=[title], max_results=1
        )

        if (
            search_results.get("results")
            and len(search_results["results"]) > 0
        ):
            first_result = search_results["results"][0]
            content = first_result.get("content")
            if content:
                logger.info(f"Successfully fetched content ({len(content)} chars)")
                return content
            else:
                logger.warning(f"No content found in search result for: {title[:50]}")
                return None
        else:
            logger.warning(f"No search results found for: {title[:50]}")
            return None

    except Exception as e:
        logger.error(f"Failed to fetch content for '{title[:50]}': {e!s}")
        return None


def fetch_news_rss(
    query: str = "bitcoin",
    count: int = 10,
    searxng_url: str | None = None,
    no_content: bool = False,
    request_delay: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Fetch news from RSS feeds with explicit parameters and return articles list.

    This function contains the core logic for fetching news articles from RSS feeds,
    optionally fetching article content using SearXNG search, and returning results
    as a list of article dictionaries.

    Args:
        query: Search query for RSS feed (default: "bitcoin").
        count: Number of articles to fetch (default: 10).
        searxng_url: SearXNG instance URL (optional - uses env var or default if None).
        no_content: Skip fetching article content if True (default: False).
        request_delay: Delay between SearXNG requests in seconds (default: 0.0).

    Returns:
        List of article dictionaries with title, url, timestamp, source, and optionally body.
    """
    rss_url = f"https://news.google.com/rss/search?q={query}"
    feed = feedparser.parse(rss_url)

    if feed.entries:
        articles = []
        for i, entry in enumerate(feed.entries[:count]):
            logger.info(f"Processing article {i+1}/{count}: {entry.title[:50]}...")

            article = {
                "title": entry.title,
                "body": "",
                "source": entry.source.title,
                "url": entry.link,
                "timestamp": entry.published,
                "unix_timestamp": calendar.timegm(entry.published_parsed),
            }

            # Fetch article content using SearXNG if not disabled
            if not no_content:
                try:
                    article_body = fetch_article_body_content(title=entry.title, searxng_url=searxng_url)
                    if article_body is not None:
                        article["body"] = article_body

                    # Add delay between requests to be respectful
                    if (i < len(feed.entries[:count]) - 1):  # Don't delay after the last article
                        time.sleep(request_delay)

                except Exception as e:
                    logger.error(f"Failed to fetch content for '{entry.title[:50]}': {e!s}")
            else:
                logger.info("Skipping content fetch due to no_content parameter")

            articles.append(article)

        articles_with_content = sum(1 for article in articles if article.get("body"))
        content_msg = (
            "(content fetching disabled)"
            if no_content
            else f"({articles_with_content} with content)"
        )
        logger.info(f"Fetched {len(articles)} articles {content_msg} from RSS feed")

        return articles

    logger.warning("No articles fetched from RSS feed")
    return []


def save_articles_to_json(
    articles: list[dict[str, Any]], args: argparse.Namespace
) -> None:
    """
    Save articles to JSON file.

    Args:
        articles: List of article dictionaries.
        args: Parsed command line arguments.
    """
    # Save articles to JSON file using existing logic
    news_dir = "src/sentiment_analysis/news"
    filename = make_timestamped_filename(output_name="news")
    filepath = os.path.join(news_dir, filename)

    save_json_data(articles, filepath)

    articles_with_content = sum(1 for article in articles if article.get("body"))
    content_msg = (
        "(content fetching disabled)"
        if args.no_content
        else f"({articles_with_content}/{len(articles)} with content)"
    )
    logger.info(f"Saved {len(articles)} articles to {filepath} {content_msg}")


def main() -> None:
    """Main function to fetch news articles and save them to JSON."""
    args = parse_args()

    articles = fetch_news_rss(
        query=args.query,
        count=args.count,
        searxng_url=args.searxng_url,
        no_content=args.no_content,
        request_delay=args.request_delay,
    )

    if not articles:
        sys.exit(1)

    save_articles_to_json(articles, args)


# Define the public API for this module
__all__ = [
    "fetch_news_rss",
    "main",
    "save_articles_to_json",
]


if __name__ == "__main__":
    main()

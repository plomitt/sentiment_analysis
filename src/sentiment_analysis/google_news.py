#!/usr/bin/env python3
"""
RSS news fetcher for Bitcoin news articles.

This module fetches Bitcoin news articles from Google RSS feeds and optionally
fetches article content using SearXNG search engine.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

import feedparser

from sentiment_analysis.config_utils import CONFIG
from sentiment_analysis.logging_utils import setup_logging
from sentiment_analysis.pipeline import run_pipeline
from sentiment_analysis.searxng_search import searxng_search, smart_searxng_search
from sentiment_analysis.utils import (
    clean_up_body_text,
    convert_google_rss_to_iso,
    convert_google_rss_to_unix,
    make_timestamped_filename,
    save_json_data,
)

# Set up logging
logger = setup_logging(__name__)

NO_CONTENT=bool(CONFIG["no_content"])


def fetch_article_body_content(title: str) -> str | None:
    """
    Fetch article body content using SearXNG search.

    Args:
        title: Article title.
        searxng_url: SearXNG instance url.
        use_smart_search: Use smart search if True (default: False).

    Returns:
        Article body content as a string, or None if not found.
    """
    try:
        logger.info(f"Fetching content for: {title[:50]}...")
        use_smart_search=bool(CONFIG["use_smart_search"])
        if use_smart_search:
            search_results = smart_searxng_search(queries=[title], max_results=1)
        else:
            search_results = searxng_search(queries=[title], max_results=1)

        if (
            search_results.get("results")
            and len(search_results["results"]) > 0
        ):
            first_result = search_results["results"][0]
            content = first_result.get("content")
            if isinstance(content, str):
                logger.info(f"Successfully fetched content ({len(content)} chars)")
                return content
            logger.warning(f"No content found in search result for: {title[:50]}")
            return None
        logger.warning(f"No search results found for: {title[:50]}")
        return None

    except Exception as e:
        logger.error(f"Failed to fetch content for '{title[:50]}': {e!s}")
        return None


def fetch_news_rss() -> list[dict[str, Any]]:
    """
    Fetch news from RSS feeds with explicit parameters and return articles list.

    This function contains the core logic for fetching news articles from RSS feeds,
    optionally fetching article content using SearXNG search, and returning results
    as a list of article dictionaries.

    Returns:
        list[dict[str, Any]]: List of article dictionaries with title, url, timestamp, source, and optionally body.
    """
    query=str(CONFIG["query"])
    count=int(CONFIG["article_count"])
    request_delay=int(CONFIG["request_delay"])

    rss_url = f"https://news.google.com/rss/search?q={query}"
    feed = feedparser.parse(rss_url)

    if feed.entries:
        articles = []
        for i, entry in enumerate(feed.entries[:count]):
            logger.info(f"Processing article {i+1}/{count}: {entry.title[:50]}...")

            article = {
                "title": entry.title,
                "body": "",
                "source": f"Google News ({entry.source.title})",
                "url": entry.link,
                "timestamp": convert_google_rss_to_iso(entry.published_parsed),
                "unix_timestamp": convert_google_rss_to_unix(entry.published_parsed),
            }

            # Fetch article content using SearXNG if not disabled
            if not NO_CONTENT:
                try:
                    article_body = fetch_article_body_content(title=entry.title)
                    if article_body is not None:
                        article["body"] = clean_up_body_text(article_body)

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
            if NO_CONTENT
            else f"({articles_with_content} with content)"
        )
        logger.info(f"Fetched {len(articles)} articles {content_msg} from RSS feed")

        return articles

    logger.warning("No articles fetched from RSS feed")
    return []


def save_articles_to_json(articles: list[dict[str, Any]]) -> None:
    """
    Save articles to JSON file.

    Args:
        articles: List of article dictionaries.
    """

    news_dir = "src/sentiment_analysis/news"
    filename = make_timestamped_filename(output_name="news")
    if filename is None:
        raise ValueError("Failed to generate filename")
    filepath = os.path.join(news_dir, filename)

    save_json_data(articles, filepath)

    articles_with_content = sum(1 for article in articles if article.get("body"))
    content_msg = (
        "(content fetching disabled)"
        if NO_CONTENT
        else f"({articles_with_content}/{len(articles)} with content)"
    )
    logger.info(f"Saved {len(articles)} articles to {filepath} {content_msg}")


def process_google_news() -> None:
    fetched_articles = fetch_news_rss()
    run_pipeline(fetched_articles)


def main() -> None:
    """Main function to fetch news articles and save them to JSON."""

    articles = fetch_news_rss()

    if not articles:
        sys.exit(1)

    save_articles_to_json(articles)


# Define the public API for this module
__all__ = [
    "fetch_article_body_content",
    "fetch_news_rss",
    "process_google_news",
    "save_articles_to_json"
]


if __name__ == "__main__":
    main()

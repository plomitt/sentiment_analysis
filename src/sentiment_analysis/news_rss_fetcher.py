from datetime import datetime
import sys
import feedparser
import argparse
import calendar
import logging
import json
import time
import os

from sentiment_analysis.searxng_search import searxng_search
from sentiment_analysis.utils import make_timestamped_filename

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Fetch Bitcoin news and save to JSON')
    parser.add_argument('--query', default='bitcoin', help='Search query (default: bitcoin)')
    parser.add_argument('--count', type=int, default=10, help='Number of articles to save (default: 10)')
    parser.add_argument('--searxng-url', help='SearXNG instance URL (default: from SEARXNG_BASE_URL env var or http://localhost:8080)')
    parser.add_argument('--no-content', action='store_true', help='Skip fetching article content from SearXNG')
    parser.add_argument('--request-delay', type=float, default=0.0, help='Delay between SearXNG requests in seconds (default: 0.0)')
    args = parser.parse_args()
    return args

def fetch_news_rss(query="bitcoin", count=10, searxng_url=None, no_content=False, request_delay=0.0):
    """
    Fetch news from RSS feeds with explicit parameters and return articles list.
    
    This function contains the core logic for fetching news articles from RSS feeds,
    optionally fetching article content using SearXNG search, and returning results
    as a list of article dictionaries.
    
    Args:
        query: Search query for RSS feed (default: "bitcoin")
        count: Number of articles to fetch (default: 10)
        searxng_url: SearXNG instance URL (optional - uses env var or default if None)
        no_content: Skip fetching article content if True (default: False)
        request_delay: Delay between SearXNG requests in seconds (default: 0.0)
    
    Returns:
        list: List of article dictionaries with title, url, timestamp, source, and optionally body
    """
    rss_url = f"https://news.google.com/rss/search?q={query}"
    feed = feedparser.parse(rss_url)

    if feed.entries:
        articles = []
        for i, entry in enumerate(feed.entries[:count]):
            logger.info(f"Processing article {i+1}/{count}: {entry.title[:50]}...")

            article = {
                "title": entry.title,
                "url": entry.link,
                "timestamp": entry.published,
                "unix_timestamp": calendar.timegm(entry.published_parsed),
                "source": entry.source.title
            }

            # Fetch article content using SearXNG if not disabled
            if not no_content:
                try:
                    logger.info(f"Fetching content for: {entry.title[:50]}...")
                    search_results = searxng_search(
                        queries=[entry.title],
                        base_url=searxng_url,
                        max_results=1
                    )

                    if search_results.get("results") and len(search_results["results"]) > 0:
                        first_result = search_results["results"][0]
                        if first_result.get("content"):
                            article["body"] = first_result["content"]
                            logger.info(f"Successfully fetched content ({len(first_result['content'])} chars)")
                        else:
                            article["body"] = ""
                            logger.warning(f"No content found in search result for: {entry.title[:50]}")
                    else:
                        article["body"] = ""
                        logger.warning(f"No search results found for: {entry.title[:50]}")

                    # Add delay between requests to be respectful
                    if i < len(feed.entries[:count]) - 1:  # Don't delay after the last article
                        time.sleep(request_delay)

                except Exception as e:
                    article["body"] = ""
                    logger.error(f"Failed to fetch content for '{entry.title[:50]}': {str(e)}")
            else:
                article["body"] = ""
                logger.info("Skipping content fetch due to no_content parameter")

            articles.append(article)

        if no_content:
            logger.info(f"Fetched {len(articles)} articles (content fetching disabled)")
        else:
            logger.info(f"Fetched {len(articles)} articles ({sum(1 for article in articles if article.get('body'))} with content)")

        print(f"Fetched {len(articles)} articles")
        if not no_content:
            articles_with_content = sum(1 for article in articles if article.get("body"))
            print(f"Articles with content: {articles_with_content}/{len(articles)}")
            
        return articles
    else:
        print("No entries found")
        return []

def save_articles_to_json(articles, args):
    # Save articles to JSON file using existing logic
    news_dir = "src/sentiment_analysis/news"
    filename = make_timestamped_filename(output_name="news")
    filepath = os.path.join(news_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=4, ensure_ascii=False)

    articles_with_content = sum(1 for article in articles if article.get("body"))
    content_msg = f"(content fetching disabled)" if args.no_content else f"({articles_with_content}/{len(articles)} with content)"
    logger.info(f"Saved {len(articles)} articles to {filepath} {content_msg}")

def main():
    args = parse_args()
    
    articles = fetch_news_rss(
        query=args.query,
        count=args.count,
        searxng_url=args.searxng_url,
        no_content=args.no_content,
        request_delay=args.request_delay
    )
    
    if articles is None:
        sys.exit(1)
    
    save_articles_to_json(articles, args)

__all__ = ["fetch_news_rss"]

if __name__ == "__main__":
    main()
from datetime import datetime
import feedparser
import argparse
import calendar
import logging
import json
import time
import os

from searxng_search import searxng_search


def main():
    parser = argparse.ArgumentParser(description='Fetch Bitcoin news and save to JSON')
    parser.add_argument('--query', default='bitcoin', help='Search query (default: bitcoin)')
    parser.add_argument('--count', type=int, default=10, help='Number of articles to save (default: 10)')
    parser.add_argument('--searxng-url', help='SearXNG instance URL (default: from SEARXNG_BASE_URL env var or http://localhost:8080)')
    parser.add_argument('--no-content', action='store_true', help='Skip fetching article content from SearXNG')
    parser.add_argument('--request-delay', type=float, default=0.0, help='Delay between SearXNG requests in seconds (default: 0.0)')
    args = parser.parse_args()

    # Configure SearXNG base URL
    searxng_base_url = args.searxng_url

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    query = args.query
    rss_url = f"https://news.google.com/rss/search?q={query}"
    feed = feedparser.parse(rss_url)

    if feed.entries:
        # Create news directory if it doesn't exist
        news_dir = "src/sentiment_analysis/news"
        os.makedirs(news_dir, exist_ok=True)

        now = datetime.now()
        sortable_timestamp = f"{99999999999999 - int(now.timestamp())}"
        readable_timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"news_{sortable_timestamp}_{readable_timestamp}.json"
        filepath = os.path.join(news_dir, filename)

        articles = []
        for i, entry in enumerate(feed.entries[:args.count]):
            logger.info(f"Processing article {i+1}/{args.count}: {entry.title[:50]}...")

            article = {
                "title": entry.title,
                "url": entry.link,
                "timestamp": entry.published,
                "unix_timestamp": calendar.timegm(entry.published_parsed),
                "source": entry.source.title
            }

            # Fetch article content using SearXNG if not disabled
            if not args.no_content:
                try:
                    logger.info(f"Fetching content for: {entry.title[:50]}...")
                    search_results = searxng_search(
                        queries=[entry.title],
                        base_url=searxng_base_url,
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
                    if i < len(feed.entries[:args.count]) - 1:  # Don't delay after the last article
                        time.sleep(args.request_delay)

                except Exception as e:
                    article["body"] = ""
                    logger.error(f"Failed to fetch content for '{entry.title[:50]}': {str(e)}")
            else:
                article["body"] = ""
                logger.info("Skipping content fetch due to --no-content flag")

            articles.append(article)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(articles, f, indent=4, ensure_ascii=False)

        # Count articles with content
        articles_with_content = sum(1 for article in articles if article.get("body"))

        if args.no_content:
            logger.info(f"Saved {len(articles)} articles to {filepath} (content fetching disabled)")
        else:
            logger.info(f"Saved {len(articles)} articles to {filepath} ({articles_with_content} with content)")

        print(f"Saved {len(articles)} articles to {filepath}")
        if not args.no_content:
            print(f"Articles with content: {articles_with_content}/{len(articles)}")
    else:
        print("No entries found")

if __name__ == "__main__":
    main()
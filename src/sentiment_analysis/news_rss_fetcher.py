import feedparser
import json
import time
import argparse
import os
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description='Fetch Bitcoin news and save to JSON')
    parser.add_argument('--query', default='bitcoin', help='Search query (default: bitcoin)')
    parser.add_argument('--count', type=int, default=10, help='Number of articles to save (default: 10)')
    args = parser.parse_args()

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
        for entry in feed.entries[:args.count]:
            article = {
                "title": entry.title,
                "url": entry.link,
                "timestamp": entry.published,
                "unix_timestamp": int(time.mktime(entry.published_parsed)),
                "source": entry.source.title
            }
            articles.append(article)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(articles, f, indent=4, ensure_ascii=False)

        print(f"Saved {len(articles)} articles to {filepath}")
    else:
        print("No entries found")

if __name__ == "__main__":
    main()
import feedparser
import json
import time
import argparse


def main():
    parser = argparse.ArgumentParser(description='Fetch Bitcoin news and save to JSON')
    parser.add_argument('--count', type=int, default=10, help='Number of articles to save (default: 10)')
    args = parser.parse_args()

    query = "bitcoin"
    rss_url = f"https://news.google.com/rss/search?q={query}"
    feed = feedparser.parse(rss_url)

    if feed.entries:
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

        with open("news.json", "w", encoding="utf-8") as f:
            json.dump(articles, f, indent=4, ensure_ascii=False)

        print(f"Saved {len(articles)} articles to news.json")
    else:
        print("No entries found")

if __name__ == "__main__":
    main()
import psycopg
from typing import Dict, List, Any, Set

from sentiment_analysis.db_utils import get_postgres_connection_string, save_analyzed_article_to_db
from sentiment_analysis.news_rss_fetcher import fetch_news_rss
from sentiment_analysis.sentiment_analyzer import analyze_article, create_client
from sentiment_analysis.utils import setup_logging

logger = setup_logging(__name__)

def is_sentiment_analysis_successful(reasoning):
    return "Analysis failed due to error:" not in reasoning

def get_analysis_results(title, body, client):
    sentiment_result = analyze_article(title, body, client)
    sentiment_data = sentiment_result.model_dump()

    score = sentiment_data.get("score", "")
    reasoning = sentiment_data.get("reasoning", "")

    return score, reasoning

def get_analyzed_article(article, client):
    # Extract article data
    title = article.get("title", "")
    body = article.get("body", "")
    source = article.get("source", "")
    url = article.get("url", "")
    timestamp = article.get("timestamp", "")
    unix_timestamp = article.get("unix_timestamp", "")

    # Analyze sentiment
    score, reasoning = get_analysis_results(title, body, client)
    sentiment_analysis_success = is_sentiment_analysis_successful(reasoning)

    analyzed_article = {
        "title": title,
        "body": body,
        "source": source,
        "url": url,
        "timestamp": timestamp,
        "unix_timestamp": unix_timestamp,
        "sentiment_analysis_success": sentiment_analysis_success,
        "sentiment_score": score,
        "sentiment_reasoning": reasoning
    }

    return analyzed_article


def get_existing_article_urls() -> Set[str]:
    """
    Fetch all existing article URLs from the database.

    Returns:
        Set[str]: Set of URLs already stored in the database.

    Raises:
        OperationalError: If database query fails.
    """
    try:
        conn_string = get_postgres_connection_string(logger)

        with psycopg.connect(conn_string) as conn:
            with conn.cursor() as cur:
                logger.debug("Fetching existing article URLs from database")
                cur.execute("SELECT url FROM articles")
                existing_urls = {url for (url,) in cur.fetchall()}
                logger.debug(f"Found {len(existing_urls)} existing articles in database")
                return existing_urls

    except Exception as e:
        logger.error(f"Failed to fetch existing article URLs: {e}")
        raise


def filter_duplicate_articles(fetched_articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter out articles that already exist in the database.

    Performs efficient duplicate detection using a single database query
    and set-based O(1) lookups, preserving original article order.
    This prevents unnecessary sentiment analysis on articles already
    processed and stored in the database.

    Args:
        fetched_articles: List of article dictionaries to filter. Each article
            should contain a 'url' key for duplicate detection.

    Returns:
        List[Dict[str, Any]]: Filtered list containing only new articles
            that don't exist in the database, preserving original order.
    """
    if not fetched_articles:
        logger.debug("No articles to filter")
        return []

    try:
        # Get existing URLs from database (single query for efficiency)
        existing_urls = get_existing_article_urls()

        # Filter duplicates while preserving order using set-based O(1) lookups
        filtered_articles = []
        duplicates_count = 0

        for article in fetched_articles:
            article_url = article.get("url")
            if article_url and article_url not in existing_urls:
                filtered_articles.append(article)
            else:
                duplicates_count += 1
                logger.debug(f"Skipping duplicate article: {article_url}")

        # Log filtering results
        if duplicates_count > 0:
            logger.info(f"Filtered out {duplicates_count} duplicate articles")
            logger.info(f"Retained {len(filtered_articles)} new articles for processing")
        else:
            logger.info("No duplicates found, all articles are new")

        return filtered_articles

    except Exception as e:
        logger.warning(f"Failed to filter duplicates, processing all articles: {e}")
        logger.info("Falling back to processing all articles without filtering")
        return fetched_articles


def process_articles(fetched_articles: List[Dict[str, Any]]):
    """
    Process a batch of articles through sentiment analysis and save to database.

    This function takes a list of articles, analyzes each one for sentiment,
    and saves the results to the PostgreSQL database with comprehensive error
    handling and statistics tracking.

    Args:
        fetched_articles: List of article dictionaries to process. Each article
            should contain keys: title, body, source, url, timestamp, unix_timestamp.
    """
    if not fetched_articles:
        logger.info("No articles to process")
        return

    logger.info(f"Starting processing of {len(fetched_articles)} articles")

    # Initialize statistics
    stats = {'processed': 0, 'failed': 0, 'total': len(fetched_articles)}

    try:
        # Create AI client
        client = create_client()
        logger.info("AI client created successfully")

        # Get database connection
        conn_string = get_postgres_connection_string(logger)

        with psycopg.connect(conn_string) as conn:
            with conn.cursor() as cur:
                logger.info("Database connection established")

                for i, article in enumerate(fetched_articles, 1):
                    try:
                        logger.debug(f"Processing article {i}/{len(fetched_articles)}")

                        # Analyze article
                        analyzed_article = get_analyzed_article(article, client)
                        logger.debug(f"Article {i} analyzed successfully")

                        # Save to database
                        success = save_analyzed_article_to_db(analyzed_article, cur, logger)

                        if success:
                            stats['processed'] += 1
                            logger.debug(f"Successfully processed and saved article {i}")
                        else:
                            stats['failed'] += 1
                            logger.error(f"Failed to save article {i} to database")

                    except Exception as e:
                        stats['failed'] += 1
                        logger.error(f"Error processing article {i}: {e}")
                        logger.debug(f"Article that failed: {article}")
                        continue

                # Commit transaction if we processed at least one article
                if stats['processed'] > 0:
                    conn.commit()
                    logger.info(f"Committed {stats['processed']} articles to database")
                else:
                    conn.rollback()
                    logger.warning("No articles processed successfully, rolling back transaction")

    except Exception as e:
        logger.error(f"Fatal error in process_articles: {e}")
        raise

    logger.info(f"Processing complete: {stats['processed']}/{stats['total']} successful, {stats['failed']} failed")


def run_pipeline():
    """
    Run the complete sentiment analysis pipeline.

    Fetches Bitcoin news articles, analyzes their sentiment, and saves results
    to the PostgreSQL database.
    """
    try:
        # Fetch Bitcoin news articles
        fetched_articles = fetch_news_rss(query="bitcoin", count=10, no_content=True)
        logger.info(f"Fetched {len(fetched_articles)} articles from RSS feed")

        if not fetched_articles:
            logger.warning("No articles fetched from RSS feed")
            return

        # Filter out duplicates before processing
        filtered_articles = filter_duplicate_articles(fetched_articles)
        if not filtered_articles:
            logger.info("All articles are duplicates, no new articles to process")
            return

        # Analyze sentiment and save to db
        logger.info(f"Processing {len(filtered_articles)} unique articles")
        process_articles(filtered_articles)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    run_pipeline()
import time
import psycopg
from typing import Dict, List, Any, Optional, Set

from sentence_transformers import SentenceTransformer

from sentiment_analysis.db_utils import get_postgres_connection_string, save_article_to_db
from sentiment_analysis.news_rss_fetcher import fetch_news_rss
from sentiment_analysis.sentiment_analyzer import analyze_article, create_client
from sentiment_analysis.utils import setup_logging

logger = setup_logging(__name__)
EMBEDDING_DIMENSIONS = 384


def get_existing_article_urls() -> Set[str]:
    """
    Fetch all existing article URLs from the database.

    Returns:
        Set[str]: Set of URLs already stored in the database.

    Raises:
        OperationalError: If database query fails.
    """
    try:
        conn_string = get_postgres_connection_string()

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


def make_embedding_text(article):
    title = article.get("title", "")
    body = article.get("body", "")
    text = f"{title} {body}"
    truncated_text = f"{text[:1000]}" if len(text) > 1000 else text
    return truncated_text

def get_embedded_articles(analyzed_articles: List[Dict[str, Any]], batch_size: Optional[int] = 32) -> List[Dict[str, Any]]:
    """
    Generate embeddings for a batch of analyzed articles using batch processing.

    Args:
        analyzed_articles: List of analyzed article dictionaries
        batch_size: Number of articles to process in each batch (default: 32)

    Returns:
        List of analyzed articles with embedding vectors added

    Raises:
        ValueError: If no articles provided
        Exception: If embedding generation fails
    """
    if not analyzed_articles:
        logger.warning("No articles provided for embedding generation")
        return []

    start_time = time.perf_counter()
    logger.info(f"Generating embeddings for {len(analyzed_articles)} articles with batch size {batch_size}")

    try:
        # Initialize model once for the entire batch
        model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.debug("SentenceTransformer model initialized successfully")

        embedded_articles = []

        # Process articles in batches to manage memory efficiently
        for i in range(0, len(analyzed_articles), batch_size):
            batch = analyzed_articles[i:i + batch_size]
            batch_start_time = time.perf_counter()

            # Prepare batch texts for embedding
            batch_texts = [make_embedding_text(article) for article in batch]

            # Generate embeddings for the entire batch at once
            batch_embeddings = model.encode(batch_texts, batch_size=len(batch))

            # Create embedded articles by adding embeddings to original articles
            for article, embedding in zip(batch, batch_embeddings):
                embedded_article = dict(article)  # Create a copy to avoid modifying original
                embedded_article["embedding"] = embedding.tolist()
                embedded_articles.append(embedded_article)

            batch_duration = time.perf_counter() - batch_start_time
            logger.debug(f"Processed batch {i//batch_size + 1}: {len(batch)} articles in {batch_duration:.2f}s")

        total_duration = time.perf_counter() - start_time
        avg_time_per_article = total_duration / len(analyzed_articles)

        logger.info(f"Successfully generated embeddings for {len(embedded_articles)} articles in {total_duration:.2f}s (avg: {avg_time_per_article:.3f}s per article).")

        return embedded_articles

    except Exception as e:
        logger.error(f"Failed to generate embeddings for articles: {e}")
        raise


def get_analysis_results(title, body, client):
    sentiment_result = analyze_article(title, body, client)
    sentiment_data = sentiment_result.model_dump()

    sentiment_analysis_success = sentiment_data.get("success", False)
    score = sentiment_data.get("score", "")
    reasoning = sentiment_data.get("reasoning", "")

    return score, reasoning, sentiment_analysis_success

def get_analyzed_article(article, client):
    # Extract article data
    title = article.get("title", "")
    body = article.get("body", "")

    # Analyze sentiment
    score, reasoning, sentiment_analysis_success = get_analysis_results(title, body, client)

    analyzed_article = dict(article)
    analyzed_article["sentiment_analysis_success"] = sentiment_analysis_success
    analyzed_article["sentiment_score"] = score
    analyzed_article["sentiment_reasoning"] = reasoning

    return analyzed_article

def get_analyzed_articles(articles):
    if not articles:
        logger.warning("No articles provided for sentiment analysis")
        return []

    logger.info(f"Analyzing {len(articles)} articles")

    try:
        start_time = time.perf_counter()

        # Initialize client once for the entire batch
        client = create_client()
        logger.debug("AI client created successfully")

        analyzed_articles = []

        # Analyze articles
        for i, article in enumerate(articles, 1):
            article_start_time = time.perf_counter()

            analyzed_article = get_analyzed_article(article, client)
            analyzed_articles.append(analyzed_article)

            article_duration = time.perf_counter() - article_start_time
            logger.debug(f"Processed article {i}/{len(articles)} in {article_duration:.2f}s")

        total_duration = time.perf_counter() - start_time
        avg_time_per_article = total_duration / len(analyzed_articles)
        logger.info(f"Successfully analyzed {len(analyzed_articles)} articles in {total_duration:.2f}s (avg: {avg_time_per_article:.3f}s per article).")

        return analyzed_articles

    except Exception as e:
        logger.error(f"Failed to analyze articles: {e}")
        raise


def save_articles_to_db(articles: List[Dict[str, Any]]):
    """
    Process a batch of articles through sentiment analysis and save to database.

    This function takes a list of articles, analyzes each one for sentiment,
    and saves the results to the PostgreSQL database with comprehensive error
    handling and statistics tracking.

    Args:
        articles: List of article dictionaries to process. Each article
        should contain keys: title, body, source, url, timestamp, unix_timestamp.
    """
    if not articles:
        logger.info("No articles to save")
        return

    total_articles = len(articles)
    processed_articles = 0
    logger.info(f"Saving {total_articles} articles to db")

    try:
        # Open database connection
        conn_string = get_postgres_connection_string()

        with psycopg.connect(conn_string) as conn:
            with conn.cursor() as cur:
                logger.info("Database connection established")

                # Save to database
                for i, article in enumerate(articles, 1):
                    try:
                        logger.debug(f"Saving article {i}/{total_articles}")
                        success = save_article_to_db(article, cur)

                        if success:
                            processed_articles += 1
                            logger.debug(f"Successfully processed and saved article {i}")
                        else:
                            logger.error(f"Failed to save article {i} to database")

                    except Exception as e:
                        logger.error(f"Error processing article {i}: {e}")
                        logger.debug(f"Article that failed: {article}")
                        continue

                # Commit transaction if we processed at least one article
                if processed_articles > 0:
                    conn.commit()
                    logger.info(f"Committed {processed_articles} articles to database")
                else:
                    conn.rollback()
                    logger.warning("No articles processed successfully, rolling back transaction")

    except Exception as e:
        logger.error(f"Fatal error in save_articles_to_db: {e}")
        raise

    logger.info(f"Saving complete: {processed_articles}/{total_articles} successful, {total_articles - processed_articles} failed")
    return processed_articles


def run_pipeline():
    """
    Run the complete sentiment analysis pipeline.

    Fetches Bitcoin news articles, analyzes their sentiment, and saves results
    to the PostgreSQL database.
    """
    try:
        logger.info("Starting sentiment analysis pipeline")
        pipeline_start_time = time.perf_counter()

        # Fetch Bitcoin news articles
        fetched_articles = fetch_news_rss(query="bitcoin", count=10, no_content=True)
        if not fetched_articles:
            logger.warning("No articles fetched from RSS feed")
            return

        # Filter out duplicates
        filtered_articles = filter_duplicate_articles(fetched_articles)
        if not filtered_articles:
            logger.info("All articles are duplicates, no new articles to process")
            return
        
        # Make vector embeddings
        embedded_articles = get_embedded_articles(filtered_articles)
        if not embedded_articles:
            logger.error("Failed to generate embeddings for articles")
            return
        
        # Analyze sentiment
        analyzed_articles = get_analyzed_articles(embedded_articles)
        if not analyzed_articles:
            logger.error("Failed to analyze articles")
            return

        # Save to db
        num_saved_articles = save_articles_to_db(analyzed_articles)
        if not num_saved_articles:
            logger.error("No articles saved to db")
            return

        pipeline_duration = time.perf_counter() - pipeline_start_time
        logger.info(f"Pipeline completed successfully in {pipeline_duration:.2f}s: fetched {len(fetched_articles)}, filtered {len(filtered_articles)}, embedded {len(embedded_articles)}, analyzed {len(analyzed_articles)}, saved {num_saved_articles} articles.")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    run_pipeline()
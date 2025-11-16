"""
Sentiment analysis pipeline orchestration.

This module provides the main pipeline functionality for fetching news articles,
analyzing sentiment, and saving results to the PostgreSQL database.
"""

from __future__ import annotations

import time
from typing import Any

import psycopg
from pgvector.psycopg import register_vector

from sentiment_analysis.config_utils import CONFIG
from sentiment_analysis.db_utils import (
    get_postgres_connection_string,
    save_article_to_db,
)
from sentiment_analysis.embedding_model import EMBEDDING_MODEL
from sentiment_analysis.logging_utils import setup_logging
from sentiment_analysis.sentiment_analyzer import analyze_article
from sentiment_analysis.utils import make_embedding_text

logger = setup_logging(__name__)


def get_existing_article_urls() -> set[str]:
    """
    Fetch all existing article URLs from the database.

    Returns:
        Set[str]: Set of URLs already stored in the database.

    Raises:
        OperationalError: If database query fails.
    """
    try:
        conn_string = get_postgres_connection_string()

        with psycopg.connect(conn_string) as conn, conn.cursor() as cur:
            logger.debug("Fetching existing article URLs from database")
            cur.execute("SELECT url FROM articles")
            existing_urls = {url for (url,) in cur.fetchall()}
            logger.debug(f"Found {len(existing_urls)} existing articles in database")
            return existing_urls

    except Exception as e:
        logger.error(f"Failed to fetch existing article URLs: {e}")
        raise

def filter_duplicate_articles(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Filter out articles that already exist in the database.

    Performs efficient duplicate detection using a single database query
    and set-based O(1) lookups, preserving original article order.
    This prevents unnecessary sentiment analysis on articles already
    processed and stored in the database.

    Args:
        articles: List of article dictionaries to filter. Each article
            should contain a 'url' key for duplicate detection.

    Returns:
        List[Dict[str, Any]]: Filtered list containing only new articles
            that don't exist in the database, preserving original order.
    """
    if not articles:
        logger.debug("No articles to filter")
        return []

    try:
        # Get existing URLs from database (single query for efficiency)
        existing_urls = get_existing_article_urls()

        # Filter duplicates while preserving order using set-based O(1) lookups
        filtered_articles = []
        duplicates_count = 0

        for article in articles:
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
        return articles


def get_embedded_articles(articles: list[dict[str, Any]], batch_size: int = 32) -> list[dict[str, Any]]:
    """
    Generate embeddings for a batch of analyzed articles using batch processing.

    Args:
        articles: List of analyzed article dictionaries
        batch_size: Number of articles to process in each batch (default: 32)

    Returns:
        List of analyzed articles with embedding vectors added

    Raises:
        ValueError: If no articles provided
        Exception: If embedding generation fails
    """
    if not articles:
        logger.warning("No articles provided for embedding generation")
        return []

    start_time = time.perf_counter()
    logger.info(f"Generating embeddings for {len(articles)} articles with batch size {batch_size}")

    try:
        embedded_articles = []

        # Process articles in batches to manage memory efficiently
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]
            batch_start_time = time.perf_counter()

            # Prepare batch texts for embedding
            batch_texts = [make_embedding_text(article) for article in batch]

            # Generate embeddings for the entire batch at once
            batch_embeddings = EMBEDDING_MODEL.encode(batch_texts, batch_size=len(batch))

            # Create embedded articles by adding embeddings to original articles
            for article, embedding in zip(batch, batch_embeddings):
                embedded_article = dict(article)  # Create a copy to avoid modifying original
                embedded_article["embedding"] = embedding.tolist()
                embedded_articles.append(embedded_article)

            batch_duration = time.perf_counter() - batch_start_time
            logger.debug(f"Processed batch {i//batch_size + 1}: {len(batch)} articles in {batch_duration:.2f}s")

        total_duration = time.perf_counter() - start_time
        avg_time_per_article = total_duration / len(articles)

        logger.info(
            f"Successfully generated embeddings for {len(embedded_articles)} articles "
            f"in {total_duration:.2f}s (avg: {avg_time_per_article:.3f}s per article)."
        )

        return embedded_articles

    except Exception as e:
        logger.error(f"Failed to generate embeddings for articles: {e}")
        raise


def fetch_similar_articles(conn: psycopg.Connection, embedding: list, limit: int = 5) -> list[dict[str, Any]]:
    """
    Fetch similar articles from database using vector similarity.

    Args:
        conn: PostgreSQL database connection.
        embedding: Vector embedding to find similar articles for.
        limit: Maximum number of similar articles to return (default: 5).

    Returns:
        list[dict[str, Any]]: List of similar articles with title, body, and sentiment_score.
    """
    with conn.cursor() as cur:
        # Using cosine distance
        cur.execute(
            """
            SELECT title, body, sentiment_score
            FROM articles
            ORDER BY embedding <=> %s::vector(384)
            LIMIT %s
            """,
            (embedding, limit)
        )
        rows = cur.fetchall()
        similar_articles = [{"title": r[0], "body": r[1], "sentiment_score": r[2]} for r in rows]

    return similar_articles

def get_enriched_articles(articles: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    """
    Enrich articles with similar articles from the database.

    Args:
        articles: List of articles with embedding vectors.
        limit: Maximum number of similar articles to find per article (default: 5).

    Returns:
        list[dict[str, Any]]: List of articles enriched with nearest_similar_articles key.
    """
    if not articles:
        logger.warning("No articles provided for enrichment")
        return []

    logger.info(f"Enriching {len(articles)} articles")

    try:
        start_time = time.perf_counter()

        conn_string = get_postgres_connection_string()
        with psycopg.connect(conn_string, autocommit=True) as conn:
            register_vector(conn)
            logger.debug("Vector extension registered successfully")
            enriched_articles = []

            for i, article in enumerate(articles, 1):
                article_start_time = time.perf_counter()

                nearest_similar_articles = fetch_similar_articles(conn, article["embedding"], limit)
                enriched_article = dict(article)
                enriched_article["nearest_similar_articles"] = nearest_similar_articles
                enriched_articles.append(enriched_article)

                article_duration = time.perf_counter() - article_start_time
                logger.debug(f"Enriched article {i}/{len(articles)} in {article_duration:.2f}s")

            total_duration = time.perf_counter() - start_time
            avg_time_per_article = total_duration / len(enriched_articles)
            logger.info(
                f"Successfully enriched {len(enriched_articles)} articles "
                f"in {total_duration:.2f}s (avg: {avg_time_per_article:.3f}s per article)."
            )
            return enriched_articles

    except Exception as e:
        logger.warning(f"Failed to enrich articles: {e}")
        logger.info("Falling back to processing all articles without enrichment")
        return articles


def get_analysis_results(title: str, body: str, nearest_similar_articles: list[dict[str, Any]]) -> tuple[str, str, bool]:
    """
    Analyze sentiment for an article and return results.

    Args:
        title: Article title.
        body: Article body text.
        nearest_similar_articles: List of similar articles for context.
        client: Instructor client for AI analysis.
        use_reasoning: Whether to use reasoning in analysis (default: None).
        temperature: Temperature for AI model (default: None).

    Returns:
        tuple[str, str, bool]: Tuple containing (score, reasoning, success_flag).
    """
    sentiment_result = analyze_article(
        title,
        body,
        nearest_similar_articles=nearest_similar_articles,
    )
    sentiment_data = sentiment_result.model_dump()

    sentiment_analysis_success = sentiment_data.get("success", False)
    score = sentiment_data.get("score", "")
    reasoning = sentiment_data.get("reasoning", "")

    return score, reasoning, sentiment_analysis_success

def get_analyzed_article(article: dict[str, Any]) -> dict[str, Any]:
    """
    Analyze sentiment for a single article.

    Args:
        article: Article dictionary containing title, body, and nearest_similar_articles.
        client: Instructor client for AI analysis.
        use_reasoning: Whether to use reasoning in analysis (default: None).
        temperature: Temperature for AI model (default: None).

    Returns:
        dict[str, Any]: Article enriched with sentiment analysis results.
    """
    # Extract article data
    title = article.get("title", "")
    body = article.get("body", "")
    nearest_similar_articles = article.get("nearest_similar_articles", [])

    # Analyze sentiment
    score, reasoning, sentiment_analysis_success = get_analysis_results(title, body, nearest_similar_articles)

    analyzed_article = dict(article)
    analyzed_article["sentiment_analysis_success"] = sentiment_analysis_success
    analyzed_article["sentiment_score"] = score
    analyzed_article["sentiment_reasoning"] = reasoning

    return analyzed_article

def get_analyzed_articles(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not articles:
        logger.warning("No articles provided for sentiment analysis")
        return []

    logger.info(f"Analyzing {len(articles)} articles")

    try:
        start_time = time.perf_counter()
        analyzed_articles = []

        # Analyze articles
        for i, article in enumerate(articles, 1):
            article_start_time = time.perf_counter()

            analyzed_article = get_analyzed_article(article)
            analyzed_articles.append(analyzed_article)

            article_duration = time.perf_counter() - article_start_time
            logger.debug(f"Analyzed article {i}/{len(articles)} in {article_duration:.2f}s")

        total_duration = time.perf_counter() - start_time
        avg_time_per_article = total_duration / len(analyzed_articles)
        logger.info(f"Successfully analyzed {len(analyzed_articles)} articles in {total_duration:.2f}s (avg: {avg_time_per_article:.3f}s per article).")

        return analyzed_articles

    except Exception as e:
        logger.error(f"Failed to analyze articles: {e}")
        raise


def save_articles_to_db(articles: list[dict[str, Any]]) -> int:
    """
    Process a batch of articles through sentiment analysis and save to database.

    This function takes a list of articles, analyzes each one for sentiment,
    and saves the results to the PostgreSQL database with comprehensive error
    handling and statistics tracking.

    Args:
        articles: List of article dictionaries to process. Each article
            should contain keys: title, body, source, url, timestamp, unix_timestamp.

    Returns:
        int: Number of articles successfully saved to the database.
    """
    if not articles:
        logger.info("No articles to save")
        return 0

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


def run_pipeline(news_articles: list[dict[str, Any]]) -> None:
    """
    Run the complete sentiment analysis pipeline.

    Receives news articles, analyzes their sentiment, and saves results
    to the PostgreSQL database.

    Args:
        news_articles: Optional list of article dictionaries to use instead of fetching from RSS (default: None).
    """
    try:
        logger.info("Starting pipeline")
        pipeline_start_time = time.perf_counter()

        # Check if articles are provided
        if not news_articles:
            logger.warning("No articles provided")
            return

        # Filter out duplicates
        filtered_articles = filter_duplicate_articles(news_articles)
        if not filtered_articles:
            logger.info("All articles are duplicates, no new articles to process")
            return

        # Make vector embeddings
        embedded_articles = get_embedded_articles(filtered_articles)
        if not embedded_articles:
            logger.error("Failed to generate embeddings for articles")
            return

        # Enrich with nearest similar articles from db
        use_similarity_scoring = bool(CONFIG["use_similarity_scoring"])
        enriched_articles = embedded_articles
        if use_similarity_scoring:
            enriched_articles = get_enriched_articles(embedded_articles)
            if not enriched_articles:
                logger.warning("Failed to enrich articles with similar articles, proceeding without enrichment")
        else:
            logger.info("Skipping enrichment")

        # Analyze sentiment
        analyzed_articles = get_analyzed_articles(enriched_articles)
        if not analyzed_articles:
            logger.error("Failed to analyze articles")
            return

        # Save to db
        num_saved_articles = save_articles_to_db(analyzed_articles)
        if not num_saved_articles:
            logger.error("No articles saved to db")
            return

        pipeline_duration = time.perf_counter() - pipeline_start_time
        logger.info(f"Pipeline completed successfully in {pipeline_duration:.2f}s: provided {len(news_articles)}, filtered {len(filtered_articles)}, embedded {len(embedded_articles)}, analyzed {len(analyzed_articles)}, saved {num_saved_articles} articles.")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


# Define the public API for this module
__all__ = [
    "filter_duplicate_articles",
    "get_analyzed_articles",
    "get_embedded_articles",
    "get_enriched_articles",
    "run_pipeline",
]

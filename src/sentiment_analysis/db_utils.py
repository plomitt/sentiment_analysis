
import os
import logging
from typing import Any, Dict, Optional
from decimal import Decimal, InvalidOperation

import psycopg
from psycopg import OperationalError
from dotenv import load_dotenv

from sentiment_analysis.utils import validate_env_config, setup_logging


def get_postgres_connection_string(logger: Optional[logging.Logger] = None) -> str:
    """
    Build PostgreSQL connection string from environment variables.

    Returns:
        str: Complete PostgreSQL connection string.

    Raises:
        ValueError: If required environment variables are missing.
    """
    load_dotenv()

    config_status = validate_env_config(["POSTGRES_PASSWORD"], logger)
    if not config_status:
        raise ValueError("Missing required PostgreSQL environment variables (POSTGRES_PASSWORD)")

    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD")
    database = os.getenv("POSTGRES_DB", "postgres")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")

    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


def save_analyzed_article_to_db(
    analyzed_article: Dict[str, Any],
    cur: psycopg.Cursor,
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Save a single analyzed article to the PostgreSQL database with upsert logic.

    This function performs an UPSERT operation based on the unique URL constraint,
    updating existing articles with new sentiment analysis data while maintaining
    data integrity and providing comprehensive error handling.

    Args:
        analyzed_article: Dictionary containing article data with keys:
            - title (str): Article title (required)
            - body (str | None): Article body content (optional)
            - source (str): Article source (required)
            - url (str): Article URL (required, unique)
            - timestamp (str): Article timestamp string (required)
            - unix_timestamp (int): Unix timestamp (required)
            - sentiment_analysis_success (bool): Analysis success flag (required)
            - sentiment_score (float): Sentiment score 1-10 (required)
            - sentiment_reasoning (str): Analysis reasoning (required)
        cur: PostgreSQL cursor object for database operations.
        logger: Optional logger instance. If not provided, a new one will be created.

    Returns:
        bool: True if the article was successfully saved/updated, False otherwise.

    Raises:
        ValueError: If required fields are missing or invalid.
        OperationalError: If database operation fails.

    Example:
        >>> with psycopg.connect(conn_string) as conn:
        ...     with conn.cursor() as cur:
        ...         success = save_analyzed_article_to_db(article, cur)
        >>> print(f"Save successful: {success}")
    """
    if logger is None:
        logger = setup_logging(__name__)

    # Validate required fields
    required_fields = [
        'title', 'source', 'url', 'timestamp', 'unix_timestamp',
        'sentiment_analysis_success', 'sentiment_score', 'sentiment_reasoning'
    ]

    missing_fields = [field for field in required_fields if field not in analyzed_article]
    if missing_fields:
        error_msg = f"Missing required fields: {missing_fields}"
        logger.error(f"Article validation failed: {error_msg}")
        raise ValueError(error_msg)

    # Validate and prepare data
    try:
        article_data = {
            'title': str(analyzed_article['title']),
            'body': analyzed_article.get('body'),  # Can be None
            'source': str(analyzed_article['source']),
            'url': str(analyzed_article['url']),
            'timestamp': str(analyzed_article['timestamp']),
            'unix_timestamp': int(analyzed_article['unix_timestamp']),
            'sentiment_analysis_success': bool(analyzed_article['sentiment_analysis_success']),
            'sentiment_score': Decimal(str(analyzed_article['sentiment_score'])),
            'sentiment_reasoning': str(analyzed_article['sentiment_reasoning'])
        }

        # Validate sentiment score range (1-10)
        score = article_data['sentiment_score']
        if not (Decimal('1.0') <= score <= Decimal('10.0')):
            raise ValueError(f"Sentiment score {score} out of valid range (1.0-10.0)")

    except (ValueError, TypeError, InvalidOperation) as e:
        error_msg = f"Data validation failed: {e}"
        logger.error(f"Article data validation error: {error_msg}")
        raise ValueError(error_msg)

    # Log article being processed
    logger.debug(f"Processing article: {article_data['title'][:50]}...")
    logger.debug(f"URL: {article_data['url']}")
    logger.debug(f"Sentiment score: {article_data['sentiment_score']}")

    # Prepare SQL for upsert operation
    upsert_sql = """
    INSERT INTO articles (
        title, body, source, url, timestamp, unix_timestamp,
        sentiment_analysis_success, sentiment_score, sentiment_reasoning
    ) VALUES (
        %(title)s, %(body)s, %(source)s, %(url)s, %(timestamp)s,
        %(unix_timestamp)s, %(sentiment_analysis_success)s,
        %(sentiment_score)s, %(sentiment_reasoning)s
    ) ON CONFLICT (url) DO UPDATE SET
        title = EXCLUDED.title,
        body = EXCLUDED.body,
        source = EXCLUDED.source,
        timestamp = EXCLUDED.timestamp,
        unix_timestamp = EXCLUDED.unix_timestamp,
        sentiment_analysis_success = EXCLUDED.sentiment_analysis_success,
        sentiment_score = EXCLUDED.sentiment_score,
        sentiment_reasoning = EXCLUDED.sentiment_reasoning,
        updated_at = CURRENT_TIMESTAMP
    """

    try:
        # Execute upsert operation
        cur.execute(upsert_sql, article_data)

        # Log success
        logger.info(f"Successfully saved/updated article: {article_data['title'][:50]}...")
        logger.debug(f"Database operation completed for URL: {article_data['url']}")

        return True

    except OperationalError as e:
        error_msg = f"Database operation failed: {e}"
        logger.error(f"Failed to save article to database: {error_msg}")
        logger.debug(f"Article data that failed: {article_data}")
        raise OperationalError(error_msg)

    except Exception as e:
        error_msg = f"Unexpected error during database operation: {e}"
        logger.error(f"Unexpected error saving article: {error_msg}")
        logger.debug(f"Error type: {type(e).__name__}")
        logger.debug(f"Article data: {article_data}")
        raise

import os
import logging
from typing import Any, Dict, Optional
from decimal import Decimal, InvalidOperation

import psycopg
from psycopg import OperationalError
from dotenv import load_dotenv

from sentiment_analysis.utils import validate_env_config, setup_logging

logger = setup_logging(__name__)


def get_postgres_connection_string() -> str:
    """
    Build PostgreSQL connection string from environment variables.

    Returns:
        str: Complete PostgreSQL connection string.

    Raises:
        ValueError: If required environment variables are missing.
    """
    load_dotenv()

    config_status = validate_env_config(["POSTGRES_PASSWORD"])
    if not config_status:
        raise ValueError("Missing required PostgreSQL environment variables (POSTGRES_PASSWORD)")

    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD")
    database = os.getenv("POSTGRES_DB", "postgres")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")

    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


def save_article_to_db(article: Dict[str, Any], cur: psycopg.Cursor) -> bool:
    """
    Save a single analyzed article to the PostgreSQL database with upsert logic.

    This function performs an UPSERT operation based on the unique URL constraint,
    updating existing articles with new sentiment analysis data while maintaining
    data integrity and providing comprehensive error handling.

    Args:
        article: Dictionary containing article data with keys
        cur: PostgreSQL cursor object for database operations.

    Returns:
        bool: True if the article was successfully saved/updated, False otherwise.

    Raises:
        ValueError: If required fields are missing or invalid.
        OperationalError: If database operation fails.
    """
    # Validate required fields
    required_fields = [
        'title', 'source', 'url', 'timestamp', 'unix_timestamp',
        'sentiment_analysis_success', 'sentiment_score', 'sentiment_reasoning'
    ]

    missing_fields = [field for field in required_fields if field not in article]
    if missing_fields:
        error_msg = f"Missing required fields: {missing_fields}"
        logger.error(f"Article validation failed: {error_msg}")
        raise ValueError(error_msg)

    # Validate and prepare data
    try:
        article_data = {
            'title': str(article['title']),
            'body': article.get('body'),  # Can be None
            'source': str(article['source']),
            'url': str(article['url']),
            'timestamp': str(article['timestamp']),
            'unix_timestamp': int(article['unix_timestamp']),
            'sentiment_analysis_success': bool(article['sentiment_analysis_success']),
            'sentiment_score': Decimal(str(article['sentiment_score'])),
            'sentiment_reasoning': str(article['sentiment_reasoning']),
            'embedding': list(article['embedding'])
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
        sentiment_analysis_success, sentiment_score, sentiment_reasoning, embedding
    ) VALUES (
        %(title)s, %(body)s, %(source)s, %(url)s, %(timestamp)s,
        %(unix_timestamp)s, %(sentiment_analysis_success)s,
        %(sentiment_score)s, %(sentiment_reasoning)s, %(embedding)s
    ) ON CONFLICT (url) DO UPDATE SET
        title = EXCLUDED.title,
        body = EXCLUDED.body,
        source = EXCLUDED.source,
        timestamp = EXCLUDED.timestamp,
        unix_timestamp = EXCLUDED.unix_timestamp,
        sentiment_analysis_success = EXCLUDED.sentiment_analysis_success,
        sentiment_score = EXCLUDED.sentiment_score,
        sentiment_reasoning = EXCLUDED.sentiment_reasoning,
        embedding = EXCLUDED.embedding,
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

def setup_database(clean_install: bool = False) -> bool:
    """
    Set up PostgreSQL database with required extensions and tables for sentiment analysis.
    
    This function executes the SQL script to create a clean PostgreSQL instance with:
    - pgcrypto and vector extensions
    - articles table with proper structure
    - unique index on URL
    
    Args:
        clean_install: If True, drops existing table before creating new one.
    
    Returns:
        bool: True if setup was successful, False otherwise.
    
    Raises:
        OperationalError: If database operation fails.
        ValueError: If required environment variables are missing.
    """
    logger.info("Starting database setup...")
    
    # Get database connection string
    try:
        conn_string = get_postgres_connection_string()
        logger.debug("Successfully obtained database connection string")
    except ValueError as e:
        logger.error(f"Failed to get database connection: {e}")
        raise
    
    # SQL setup script
    setup_sql = """
    -- 1. Enable extensions
    CREATE EXTENSION IF NOT EXISTS pgcrypto;
    CREATE EXTENSION IF NOT EXISTS vector;
    
    -- 2. Drop existing table if clean_install is True
    """
    
    if clean_install:
        setup_sql += """
        DROP TABLE IF EXISTS public.articles CASCADE;
        """
        logger.info("Clean install enabled: dropping existing articles table")
    
    setup_sql += """
    -- 3. Create the articles table
    CREATE TABLE public.articles (
        id uuid NOT NULL DEFAULT gen_random_uuid(),
        title text NOT NULL,
        body text,
        source text NOT NULL,
        url text NOT NULL,
        timestamp text NOT NULL,
        unix_timestamp int4 NOT NULL,
        sentiment_analysis_success bool NOT NULL,
        sentiment_score numeric NOT NULL,
        sentiment_reasoning text NOT NULL,
        created_at timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
        embedding vector(384)
    );
    
    -- 4. Create unique index on url
    CREATE UNIQUE INDEX articles_url ON public.articles USING btree (url);
    """
    
    # Execute setup script
    try:
        with psycopg.connect(conn_string) as conn:
            with conn.cursor() as cur:
                logger.info("Executing database setup script...")
                
                # Execute the setup SQL
                cur.execute(setup_sql)
                
                # Verify table creation
                cur.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_name = 'articles'
                """)
                
                result = cur.fetchone()
                if not result:
                    raise OperationalError("Articles table was not created successfully")
                
                # Verify extensions
                cur.execute("""
                    SELECT extname 
                    FROM pg_extension 
                    WHERE extname IN ('pgcrypto', 'vector')
                """)
                
                extensions = [row[0] for row in cur.fetchall()]
                missing_extensions = []
                for ext in ['pgcrypto', 'vector']:
                    if ext not in extensions:
                        missing_extensions.append(ext)
                
                if missing_extensions:
                    raise OperationalError(f"Failed to create extensions: {missing_extensions}")
                
                logger.info("Database setup completed successfully")
                logger.info(f"Created articles table with extensions: {', '.join(extensions)}")
                
                return True
                
    except OperationalError as e:
        error_msg = f"Database setup failed: {e}"
        logger.error(error_msg)
        raise OperationalError(error_msg)
    except Exception as e:
        logger.error(f"Unexpected error during database setup: {e}")
        logger.debug(f"Error type: {type(e).__name__}")
        raise

if __name__ == "__main__":
    user_input = input("This will set up a clean database instance (dropping existing tables if they exist). Type 'confirm' to proceed: ")
    if user_input.lower() == 'confirm':
        setup_database(clean_install=True)
    else:
        print("Database setup cancelled.")
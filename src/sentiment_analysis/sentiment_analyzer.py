"""
Bitcoin news sentiment analyzer.

This module uses Instructor to analyze Bitcoin news articles and generate
sentiment scores with trading-focused reasoning.
"""

from __future__ import annotations

import os
from typing import Any

from instructor import Instructor, Mode
from pydantic import BaseModel, Field, field_validator

from sentiment_analysis.client_manager import build_client
from sentiment_analysis.prompt_manager import get_sentiment_analysis_prompt_with_context
from sentiment_analysis.utils import (
    find_latest_file,
    load_json_data,
    make_timestamped_filename,
    save_json_data,
    setup_logging,
)

# Configure logging
logger = setup_logging(__name__)


class SentimentAnalysis(BaseModel):
    """Structured output for sentiment analysis results."""

    score: float = Field(
        ...,
        ge=1.0,
        le=10.0,
        description="Sentiment score from 1 (strong sell) to 10 (strong buy)",
    )
    reasoning: str = Field(
        ...,
        description="Concise reasoning for the score focusing on trading implications",
    )

    @field_validator("score")
    @classmethod
    def validate_score_range(cls, v: float) -> float:
        """Ensure score is within valid range."""
        if not (1.0 <= v <= 10.0):
            raise ValueError("Score must be between 1.0 and 10.0")
        return v


class ArticleWithSentiment(BaseModel):
    """Article data with sentiment analysis."""

    title: str = Field(..., description="Article title")
    body: str | None = Field(None, description="Article body content")
    source: str | None = Field(None, description="Article source")
    timestamp: str = Field(..., description="Article timestamp")
    url: str = Field(..., description="Article URL")
    unix_timestamp: int | None = Field(
        None, description="Unix timestamp for sorting and analysis"
    )
    sentiment: SentimentAnalysis | None = Field(
        None, description="Sentiment analysis results"
    )


def get_file_dirs() -> tuple[str, str]:
    """
    Get file directories for news and sentiments.

    Returns:
        Tuple of (news_dir, sentiments_dir) paths.
    """
    # Get script directory to handle file paths correctly
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define directory paths
    news_dir = os.path.join(script_dir, "news")
    sentiments_dir = os.path.join(script_dir, "sentiments")

    # Ensure directories exist
    os.makedirs(news_dir, exist_ok=True)
    os.makedirs(sentiments_dir, exist_ok=True)

    return news_dir, sentiments_dir


def load_articles_from_json(file_path: str) -> list[dict[str, Any]]:
    """
    Load articles from a JSON file.

    Args:
        file_path: Path to the JSON file containing articles.

    Returns:
        List of article dictionaries.
    """
    try:
        articles = load_json_data(file_path, logger)
        logger.info(f"Loaded {len(articles)} articles from {file_path}")
        return articles
    except Exception as e:
        logger.error("Error: No articles found in input file")
        logger.error(f"Error loading articles from {file_path}: {e!s}")
        return []


def create_client() -> Instructor:
    """
    Create and return an Instructor client for sentiment analysis.

    Returns:
        Configured client instance.
    """
    config = {"mode": Mode.JSON}
    client = build_client(config=config)
    logger.info("Instructor client created for sentiment analysis")
    return client


def analyze_article(
    title: str, body: str | None, client: Instructor
) -> SentimentAnalysis:
    """
    Analyze a single article for sentiment.

    Args:
        title: Article title.
        body: Article body content (optional - can be None or empty).
        client: Instructor client instance.

    Returns:
        SentimentAnalysis object with score and reasoning.
    """
    # Handle None or empty body gracefully
    body_content = body if body else ""

    try:
        # Get the formatted prompt
        prompt = get_sentiment_analysis_prompt_with_context(title, body_content)

        # Use Instructor to get structured output
        sentiment = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert Bitcoin trading analyst.",
                },
                {"role": "user", "content": prompt},
            ],
            response_model=SentimentAnalysis,
            temperature=0.1,  # Lower temperature for more consistent scoring
        )

        logger.info(f"Analysis complete - Score: {sentiment.score}")
        return sentiment

    except Exception as e:
        logger.error(f"Error analyzing article '{title[:50]}...': {e!s}")
        # Return a neutral sentiment as fallback
        return SentimentAnalysis(
            score=5.0, reasoning=f"Analysis failed due to error: {e!s}"
        )


def analyze_articles_batch(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Analyze multiple articles in batch.

    Args:
        articles: List of article dictionaries with title, body, timestamp, url.

    Returns:
        List of ArticleWithSentiment objects as dictionaries.
    """
    client = create_client()

    results = []
    total_articles = len(articles)

    logger.info(f"Starting batch analysis of {total_articles} articles")

    for i, article in enumerate(articles, 1):
        try:
            # Extract article data
            title = article.get("title", "")
            body = article.get("body", "")
            source = article.get("source", "")
            url = article.get("url", "")
            timestamp = article.get("timestamp", "")
            unix_timestamp = article.get("unix_timestamp", "")

            # Analyze sentiment
            sentiment = analyze_article(title, body, client)

            # Create result object
            article_with_sentiment = ArticleWithSentiment(
                title=title,
                body=body,
                source=source,
                url=url,
                timestamp=timestamp,
                unix_timestamp=unix_timestamp,
                sentiment=sentiment,
            )

            results.append(article_with_sentiment)

            # Log progress
            logger.info(f"Progress: {i}/{total_articles} articles processed")

        except Exception as e:
            logger.error(f"Error processing article {i}: {e!s}")
            continue

    results_dict = [article.model_dump() for article in results]
    logger.info(
        f"Batch analysis complete. {len(results)} articles processed successfully"
    )
    logger.info(f"Analysis complete! Analyzed {len(results)} articles")
    return results_dict


def print_analysis_summary(articles_with_sentiment: list[dict[str, Any]]) -> None:
    """
    Print a summary of the analysis results.

    Args:
        articles_with_sentiment: List of articles with sentiment analysis.
    """
    if not articles_with_sentiment:
        return

    scores = [
        article["sentiment"]["score"]
        for article in articles_with_sentiment
        if article.get("sentiment")
    ]

    if not scores:
        logger.warning("No valid sentiment scores found")
        return

    avg_score = sum(scores) / len(scores)
    min_score = min(scores)
    max_score = max(scores)

    # Count articles in each category
    strong_sell = sum(1 for score in scores if score <= 3.0)
    weak_sell = sum(1 for score in scores if 3.1 <= score <= 5.0)
    neutral = sum(1 for score in scores if 5.1 <= score <= 6.0)
    weak_buy = sum(1 for score in scores if 6.1 <= score <= 8.0)
    strong_buy = sum(1 for score in scores if score > 8.0)

    logger.info("=" * 50)
    logger.info("BITCOIN NEWS SENTIMENT ANALYSIS SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total Articles Analyzed: {len(scores)}")
    logger.info(f"Average Sentiment Score: {avg_score:.2f}")
    logger.info(f"Score Range: {min_score:.1f} - {max_score:.1f}")
    logger.info(f"Strong Sell (1.0-3.0): {strong_sell} articles")
    logger.info(f"Weak Sell (3.1-5.0): {weak_sell} articles")
    logger.info(f"Neutral (5.1-6.0): {neutral} articles")
    logger.info(f"Weak Buy (6.1-8.0): {weak_buy} articles")
    logger.info(f"Strong Buy (8.1-10.0): {strong_buy} articles")
    logger.info("=" * 50)


def save_results_to_json(
    output_file: str, articles_with_sentiment: list[dict[str, Any]]
) -> None:
    """
    Save analysis results to JSON file.

    Args:
        output_file: Path to output file.
        articles_with_sentiment: List of articles with sentiment analysis.
    """
    save_json_data(articles_with_sentiment, output_file, logger)
    logger.info(f"Analysis complete! Results saved to {output_file}")


def main() -> None:
    """Main function to run the sentiment analysis process."""
    news_dir, sentiments_dir = get_file_dirs()

    # Find latest news file
    input_file = find_latest_file(news_dir, "news", "json", logger)

    if input_file is None:
        logger.error("No news files found to analyze")
        return

    # Load articles and analyze sentiment
    articles = load_articles_from_json(input_file)
    if not articles:
        logger.error("No articles loaded for analysis")
        return

    articles_with_sentiment = analyze_articles_batch(articles)

    # Save the results
    output_filename = make_timestamped_filename(
        input_file=input_file,
        input_name="news",
        output_name="sentiments",
        output_filetype="json",
        logger=logger,
    )
    output_file = os.path.join(sentiments_dir, output_filename)
    save_results_to_json(output_file, articles_with_sentiment)

    print_analysis_summary(articles_with_sentiment)


# Define the public API for this module
__all__ = [
    "ArticleWithSentiment",
    "SentimentAnalysis",
    "analyze_article",
    "analyze_articles_batch",
    "main",
    "print_analysis_summary",
    "save_results_to_json",
]


if __name__ == "__main__":
    main()

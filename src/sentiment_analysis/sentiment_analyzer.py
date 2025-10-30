"""
Bitcoin News Sentiment Analyzer

This module uses Instructor to analyze Bitcoin news articles and generate
sentiment scores with trading-focused reasoning.
"""

from __future__ import annotations

from datetime import datetime
import os
import sys
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from instructor import Instructor, Mode

from sentiment_analysis.client_manager import build_client
from sentiment_analysis.prompt_manager import get_sentiment_analysis_prompt_with_context
from sentiment_analysis.utils import (
    make_timestamped_filename,
    setup_logging, load_json_data, save_json_data, find_latest_file
)

# Configure logging
logger = setup_logging(__name__)

class SentimentAnalysis(BaseModel):
    """Structured output for sentiment analysis results."""

    score: float = Field(
        ...,
        ge=1.0,
        le=10.0,
        description="Sentiment score from 1 (strong sell) to 10 (strong buy)"
    )
    reasoning: str = Field(
        ...,
        description="Concise reasoning for the score focusing on trading implications"
    )

    @field_validator('score')
    @classmethod
    def validate_score_range(cls, v):
        """Ensure score is within valid range."""
        if not (1.0 <= v <= 10.0):
            raise ValueError('Score must be between 1.0 and 10.0')
        return v

class ArticleWithSentiment(BaseModel):
    """Article data with sentiment analysis."""

    title: str = Field(..., description="Article title")
    body: Optional[str] = Field(None, description="Article body content")
    timestamp: str = Field(..., description="Article timestamp")
    url: str = Field(..., description="Article URL")
    unix_timestamp: Optional[int] = Field(None, description="Unix timestamp for sorting and analysis")
    sentiment: Optional[SentimentAnalysis] = Field(None, description="Sentiment analysis results")


def get_file_dirs():
    # Get script directory to handle file paths correctly
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define directory paths
    news_dir = os.path.join(script_dir, "news")
    sentiments_dir = os.path.join(script_dir, "sentiments")

    # Ensure directories exist
    os.makedirs(news_dir, exist_ok=True)
    os.makedirs(sentiments_dir, exist_ok=True)

    return news_dir, sentiments_dir

def load_articles_from_json(file_path: str) -> List[Dict[str, Any]]:
    """
    Load articles from a JSON file.

    Args:
        file_path: Path to the JSON file containing articles

    Returns:
        List of article dictionaries
    """
    try:
        articles = load_json_data(file_path, logger)
        logger.info(f"Loaded {len(articles)} articles from {file_path}")
        return articles
    except Exception as e:
        print("❌ Error: No articles found in input file")
        logger.error(f"Error loading articles from {file_path}: {str(e)}")
        return []

def create_client() -> Instructor:
    """
    Create and return an Instructor client for sentiment analysis.

    Returns:
        Instructor: Configured client instance
    """
    config = {
        "mode": Mode.JSON
    }
    client = build_client(config=config)
    logger.info("Instructor client created for sentiment analysis")
    return client

def analyze_article(title: str, body: Optional[str], client: Instructor) -> SentimentAnalysis:
    """
    Analyze a single article for sentiment.

    Args:
        title: Article title
        body: Article body content (optional - can be None or empty)
        client: Instructor client instance. If None, creates default client.

    Returns:
        SentimentAnalysis object with score and reasoning
    """
    # Handle None or empty body gracefully
    body_content = body if body else ""

    try:
        # Get the formatted prompt
        prompt = get_sentiment_analysis_prompt_with_context(title, body_content)

        # Use Instructor to get structured output
        sentiment = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert Bitcoin trading analyst."},
                {"role": "user", "content": prompt}
            ],
            response_model=SentimentAnalysis,
            temperature=0.1  # Lower temperature for more consistent scoring
        )

        logger.info(f"Analysis complete - Score: {sentiment.score}")
        return sentiment

    except Exception as e:
        logger.error(f"Error analyzing article '{title[:50]}...': {str(e)}")
        # Return a neutral sentiment as fallback
        return SentimentAnalysis(
            score=5.0,
            reasoning=f"Analysis failed due to error: {str(e)}"
        )

def analyze_articles_batch(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Analyze multiple articles in batch.

    Args:
        articles: List of article dictionaries with title, body, timestamp, url
        client: Instructor client instance. If None, creates default client.

    Returns:
        List of ArticleWithSentiment objects
    """
    client = create_client()

    results = []
    total_articles = len(articles)

    logger.info(f"Starting batch analysis of {total_articles} articles")

    for i, article in enumerate(articles, 1):
        try:
            # Extract article data
            title = article.get('title', '')
            body = article.get('body', '')  # Returns empty string if body field doesn't exist
            timestamp = article.get('timestamp', '')
            url = article.get('url', '')
            unix_timestamp = article.get('unix_timestamp')

            # Analyze sentiment
            sentiment = analyze_article(title, body, client)

            # Create result object
            article_with_sentiment = ArticleWithSentiment(
                title=title,
                body=body if body else None,  # Store None if body was empty/missing
                timestamp=timestamp,
                url=url,
                unix_timestamp=unix_timestamp,
                sentiment=sentiment
            )

            results.append(article_with_sentiment)

            # Log progress
            logger.info(f"Progress: {i}/{total_articles} articles processed")

        except Exception as e:
            logger.error(f"Error processing article {i}: {str(e)}")
            continue

    results_dict = [article.model_dump() for article in results]
    logger.info(f"Batch analysis complete. {len(results)} articles processed successfully")
    print(f"\n✅ Analysis complete! Analyzed {len(results)} articles")
    return results_dict

def print_analysis_summary(articles_with_sentiment: List[ArticleWithSentiment]):
    """
    Print a summary of the analysis results.

    Args:
        articles_with_sentiment: List of articles with sentiment analysis
    """
    if not articles_with_sentiment:
        return

    scores = [article['sentiment']['score'] for article in articles_with_sentiment if article['sentiment']]

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

    print("\n" + "="*50)
    print("BITCOIN NEWS SENTIMENT ANALYSIS SUMMARY")
    print("="*50)
    print(f"Total Articles Analyzed: {len(scores)}")
    print(f"Average Sentiment Score: {avg_score:.2f}")
    print(f"Score Range: {min_score:.1f} - {max_score:.1f}")
    print(f"Strong Sell (1.0-3.0): {strong_sell} articles")
    print(f"Weak Sell (3.1-5.0): {weak_sell} articles")
    print(f"Neutral (5.1-6.0): {neutral} articles")
    print(f"Weak Buy (6.1-8.0): {weak_buy} articles")
    print(f"Strong Buy (8.1-10.0): {strong_buy} articles")
    print("="*50)

def save_results_to_json(output_file, articles_with_sentiment):
    # Save results to file using existing logic
    save_json_data(articles_with_sentiment, output_file, logger)

    print(f"\n✅ Analysis complete! Results saved to {output_file}")
    
def main():
    news_dir, sentiments_dir = get_file_dirs()

    # Find latest news file
    input_file = find_latest_file(news_dir, "news", "json", logger)

    # Load articles and analyze sentiment
    articles = load_articles_from_json(input_file)
    articles_with_sentiment = analyze_articles_batch(articles)
    
    # Save the results
    output_filename = make_timestamped_filename(input_file, "news", "sentiments", "json", logger)
    output_file = os.path.join(sentiments_dir, output_filename)
    save_results_to_json(output_file, articles_with_sentiment)
    
    print_analysis_summary(articles_with_sentiment)

__all__ = ["analyze_sentiments"]

if __name__ == "__main__":
    main()
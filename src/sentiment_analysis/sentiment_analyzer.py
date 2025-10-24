"""
Bitcoin News Sentiment Analyzer

This module uses Instructor to analyze Bitcoin news articles and generate
sentiment scores with trading-focused reasoning.
"""

from __future__ import annotations

from datetime import datetime
import json
import logging
import os
import glob
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from instructor import Instructor, Mode

from sentiment_analysis.client_manager import build_client
from sentiment_analysis.prompt_manager import get_sentiment_analysis_prompt_with_context

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def create_client() -> Instructor:
    """
    Create and return an Instructor client for sentiment analysis.

    Returns:
        Instructor: Configured client instance
    """
    config = {
        "mode": Mode.JSON
    }
    client = build_client(config)
    logger.info("Instructor client created for sentiment analysis")
    return client


def analyze_article(title: str, body: Optional[str], client: Optional[Instructor] = None) -> SentimentAnalysis:
    """
    Analyze a single article for sentiment.

    Args:
        title: Article title
        body: Article body content (optional - can be None or empty)
        client: Instructor client instance. If None, creates default client.

    Returns:
        SentimentAnalysis object with score and reasoning
    """
    if client is None:
        client = create_client()

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


def analyze_articles_batch(articles: List[Dict[str, Any]], client: Optional[Instructor] = None) -> List[ArticleWithSentiment]:
    """
    Analyze multiple articles in batch.

    Args:
        articles: List of article dictionaries with title, body, timestamp, url
        client: Instructor client instance. If None, creates default client.

    Returns:
        List of ArticleWithSentiment objects
    """
    if client is None:
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

    logger.info(f"Batch analysis complete. {len(results)} articles processed successfully")
    return results


def find_latest_news_file(news_dir: str) -> Optional[str]:
    """
    Find the latest news file from the news directory.

    Since news files are named with sortable prefixes for reverse chronological order,
    the first file alphabetically is the newest.

    Args:
        news_dir: Path to the news directory

    Returns:
        Path to the latest news file, or None if no files found
    """
    try:
        # Look for all news JSON files
        pattern = os.path.join(news_dir, "news_*.json")
        news_files = glob.glob(pattern)

        if not news_files:
            logger.error(f"No news files found in {news_dir}")
            return None

        # Sort alphabetically - with the new naming scheme, this puts newest first
        news_files.sort()
        latest_file = news_files[0]

        logger.info(f"Found latest news file: {latest_file}")
        return latest_file

    except Exception as e:
        logger.error(f"Error finding latest news file: {str(e)}")
        return None


def extract_timestamp_from_filename(filepath: str) -> Optional[str]:
    """
    Extract timestamp from a news filename for use in output filename.

    Expected format: news_[sortable]_[readable].json
    Example: news_99998238678017_2025-10-24_18-06-22.json

    Args:
        filepath: Full path to the input news file

    Returns:
        Timestamp string (sortable_readable) or None if extraction fails
    """
    try:
        filename = os.path.basename(filepath)

        # Expected pattern: news_[sortable]_[readable].json
        # Example: news_99998238678017_2025-10-24_18-06-22.json

        if not (filename.startswith("news_") and filename.endswith(".json")):
            logger.warning(f"Filename doesn't match expected pattern: {filename}")
            return None

        # Remove "news_" prefix and ".json" suffix
        # From: "news_99998238678017_2025-10-24_18-06-22.json"
        # To:   "99998238678017_2025-10-24_18-06-22"
        timestamp_part = filename[5:-5]

        # Validate that the timestamp part contains the expected format
        # Should have at least one underscore (separating sortable and readable parts)
        # The readable part should also contain underscores for date formatting
        if "_" not in timestamp_part or timestamp_part.count("_") < 2:
            logger.warning(f"Timestamp part doesn't contain expected format: {timestamp_part}")
            return None

        logger.info(f"Extracted timestamp from filename: {timestamp_part}")
        return timestamp_part

    except Exception as e:
        logger.error(f"Error extracting timestamp from filename {filepath}: {str(e)}")
        return None


def load_articles_from_json(file_path: str) -> List[Dict[str, Any]]:
    """
    Load articles from a JSON file.

    Args:
        file_path: Path to the JSON file containing articles

    Returns:
        List of article dictionaries
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        logger.info(f"Loaded {len(articles)} articles from {file_path}")
        return articles
    except Exception as e:
        logger.error(f"Error loading articles from {file_path}: {str(e)}")
        return []


def save_results_to_json(articles_with_sentiment: List[ArticleWithSentiment], output_path: str):
    """
    Save analysis results to JSON file.

    Args:
        articles_with_sentiment: List of ArticleWithSentiment objects
        output_path: Path where to save the results
    """
    try:
        # Convert to dictionaries for JSON serialization
        results_dict = [article.model_dump() for article in articles_with_sentiment]

        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {output_path}")

    except Exception as e:
        logger.error(f"Error saving results to {output_path}: {str(e)}")


def print_analysis_summary(articles_with_sentiment: List[ArticleWithSentiment]):
    """
    Print a summary of the analysis results.

    Args:
        articles_with_sentiment: List of articles with sentiment analysis
    """
    if not articles_with_sentiment:
        return

    scores = [article.sentiment.score for article in articles_with_sentiment if article.sentiment]

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


def analyze_news_file(input_file: str, output_file: str, client: Optional[Instructor] = None) -> bool:
    """
    Complete pipeline: load articles, analyze sentiment, save results.

    Args:
        input_file: Path to input JSON file with articles
        output_file: Path to output JSON file for results
        client: Instructor client instance. If None, creates default client.

    Returns:
        True if successful, False otherwise
    """
    try:
        # Load articles
        articles = load_articles_from_json(input_file)
        if not articles:
            logger.error("No articles loaded, aborting analysis")
            return False

        # Analyze sentiment
        articles_with_sentiment = analyze_articles_batch(articles, client)

        # Save results
        save_results_to_json(articles_with_sentiment, output_file)

        # Print summary
        print_analysis_summary(articles_with_sentiment)

        return True

    except Exception as e:
        logger.error(f"Error in analysis pipeline: {str(e)}")
        return False


def main():
    """Main function to run the sentiment analysis."""
    # Get script directory to handle file paths correctly
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define directory paths
    news_dir = os.path.join(script_dir, "news")
    sentiments_dir = os.path.join(script_dir, "sentiments")

    # Ensure directories exist
    os.makedirs(news_dir, exist_ok=True)
    os.makedirs(sentiments_dir, exist_ok=True)

    # Find the latest news file
    input_file = find_latest_news_file(news_dir)
    if not input_file:
        print("❌ Error: No news files found in src/sentiment_analysis/news/")
        print("Please run the RSS fetcher first to generate news files.")
        return

    # Try to extract timestamp from input filename for consistent tracking
    extracted_timestamp = extract_timestamp_from_filename(input_file)

    if extracted_timestamp:
        # Use the same timestamp as the input file
        output_filename = f"sentiments_{extracted_timestamp}.json"
        print(f"📋 Using input timestamp: {extracted_timestamp}")
    else:
        # Fallback: generate new timestamp if extraction fails
        print("⚠️  Could not extract timestamp from input filename, generating new timestamp")
        now = datetime.now()
        # Create sortable prefix: subtract from max timestamp to invert ordering
        sortable_timestamp = f"{99999999999999 - int(now.timestamp())}"
        readable_timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        output_filename = f"sentiments_{sortable_timestamp}_{readable_timestamp}.json"

    output_file = os.path.join(sentiments_dir, output_filename)

    # Run analysis
    print(f"📁 Input file: {input_file}")
    print(f"📁 Output file: {output_file}")
    print()

    success = analyze_news_file(input_file, output_file)

    if success:
        print(f"\n✅ Analysis complete! Results saved to {output_file}")
    else:
        print(f"\n❌ Analysis failed. Check logs for details.")


if __name__ == "__main__":
    main()
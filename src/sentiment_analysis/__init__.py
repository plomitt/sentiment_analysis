"""
Sentiment Analysis Package.

A comprehensive Bitcoin news sentiment analysis tool with AI-powered insights.
"""

__version__ = "1.0.0"
__author__ = "plomitt"
__email__ = "46419727+plomitt@users.noreply.github.com"

# Public API exports
from sentiment_analysis.client_manager import build_client
from sentiment_analysis.news_rss_fetcher import fetch_news_rss
from sentiment_analysis.prompt_manager import get_sentiment_analysis_prompt_with_context
from sentiment_analysis.sentiment_analyzer import (
    ArticleWithSentiment,
    SentimentAnalysis,
    analyze_article,
    analyze_articles_batch,
)
from sentiment_analysis.sentiment_grapher import generate_sentiment_charts
from sentiment_analysis.utils import setup_logging

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    # Core analysis
    "SentimentAnalysis",
    "ArticleWithSentiment",
    "analyze_article",
    "analyze_articles_batch",
    # Data fetching
    "fetch_news_rss",
    # Visualization
    "generate_sentiment_charts",
    # Utilities
    "build_client",
    "get_sentiment_analysis_prompt_with_context",
    "setup_logging",
]

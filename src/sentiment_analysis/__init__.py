"""
Sentiment Analysis Package.

A comprehensive Bitcoin news sentiment analysis tool with AI-powered insights.
"""

__version__ = "3.0.0"
__author__ = "plomitt"
__email__ = "46419727+plomitt@users.noreply.github.com"

# Core pipeline and workflow
# Client management
from sentiment_analysis.client_manager import build_client
from sentiment_analysis.config_utils import get_config

# Data fetching and processing
from sentiment_analysis.google_news import fetch_news_rss, process_google_news
from sentiment_analysis.telegram_news import process_realtime_telegram_news
from sentiment_analysis.alpaca_news import process_realtime_alpaca_news
from sentiment_analysis.pipeline import run_pipeline

# Prompts
from sentiment_analysis.prompt_manager import get_sentiment_analysis_prompt_with_context

# Search functionality
from sentiment_analysis.searxng_search import (
    searxng_search,
    smart_searxng_search,
)

# Sentiment analysis
from sentiment_analysis.sentiment_analyzer import (
    ArticleWithSentiment,
    SentimentAnalysis,
    SentimentAnalysisWithReasoning,
    analyze_article
)

# Multi-source processing
from sentiment_analysis.parallel_processor import (
    ParallelProcessor,
    run_parallel_processor
)

# Visualization
from sentiment_analysis.sentiment_grapher import generate_sentiment_charts

# Utilities and file I/O
from sentiment_analysis.utils import (
    find_latest_file,
    load_json_data,
    make_timestamped_filename,
    save_json_data,
    setup_logging,
)

__all__ = [
    # Package metadata
    "__version__",
    "__author__",
    "__email__",

    # Core workflow
    "run_pipeline",
    "get_config",

    # Data operations
    "fetch_news_rss",
    "load_json_data",
    "save_json_data",
    "find_latest_file",
    "make_timestamped_filename",

    # Analysis
    "SentimentAnalysis",
    "SentimentAnalysisWithReasoning",
    "ArticleWithSentiment",
    "analyze_article",

    # Multi-source processing
    "process_google_news",
    "process_realtime_telegram_news",
    "process_realtime_alpaca_news",
    "ParallelProcessor",
    "run_parallel_processor",

    # Search
    "searxng_search",
    "smart_searxng_search",

    # Clients
    "build_client",

    # Utilities
    "setup_logging",

    # Visualization
    "generate_sentiment_charts",

    # Prompts
    "get_sentiment_analysis_prompt_with_context",
]

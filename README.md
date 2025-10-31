# Bitcoin News Sentiment Analysis Pipeline

A production-ready sentiment analysis pipeline for Bitcoin news articles using Instructor and large language models. This system fetches real-time news, performs trading-focused sentiment analysis, generates visualizations, and provides statistical consistency testing - all with automatic file management and timestamp tracking.

Built with **state-of-the-art code quality standards** using Black, isort, ruff, and mypy for enterprise-grade reliability and maintainability.

The pipeline provides sentiment scores on a 1-10 scale where 1 suggests "sell" and 10 suggests "buy", specifically designed for Bitcoin trading decisions.

## Features

### Core Functionality
- **Fully Automated Pipeline**: Complete workflow from RSS fetch to statistical analysis
- **Real-time News Fetching**: Automatically fetches Bitcoin news from Google RSS feeds
- **Trading-Focused Analysis**: Sentiment scores specifically designed for Bitcoin trading decisions
- **Auto-detection System**: Automatically finds and uses the latest files throughout the pipeline
- **Timestamped Organization**: Consistent sortable file naming for easy tracking and management
- **Statistical Consistency Testing**: Comprehensive analysis of sentiment reliability and variability
- **Chart Generation**: High-quality time-series visualizations with rolling averages
- **Structured Output**: Uses Pydantic models for consistent, validated results
- **Intelligent Prompting**: Few-shot learning with chain-of-thought reasoning
- **Batch Processing**: Efficiently analyze multiple articles
- **Detailed Reasoning**: Each score includes concise trading-focused explanations

### Code Quality & Development
- **Industry-Standard Code Quality**: Built with Black, isort, ruff, and mypy for enterprise-grade reliability
- **Comprehensive Type Safety**: Full type annotations with strict mypy checking
- **Modular Architecture**: Clean separation of concerns with well-defined public API
- **Pre-commit Hooks**: Automated code quality enforcement
- **Comprehensive Documentation**: Full API reference and usage examples
- **Production Ready**: Robust error handling, logging, and configuration management

## Project Structure

The codebase follows a clean, modular architecture with comprehensive API design:

```
sentiment-analysis/
├── pyproject.toml               # Project configuration and dependencies
├── ruff.toml                    # Linting and formatting rules
├── .mypy.ini                    # Type checking configuration
├── .pre-commit-config.yaml      # Pre-commit hooks configuration
├── README.md                    # This file
└── src/
    └── sentiment_analysis/
        ├── __init__.py           # Public API exports and package metadata
        ├── client_manager.py     # AI client configuration (OpenRouter/LMStudio)
        ├── news_rss_fetcher.py   # RSS news fetcher for real-time Bitcoin news
        ├── prompt_manager.py     # Specialized prompts for Bitcoin sentiment analysis
        ├── searxng_search.py     # Search engine integration for article content
        ├── sentiment_analyzer.py # LLM sentiment analysis engine with Pydantic models
        ├── sentiment_grapher.py  # Chart generation and data visualization
        ├── utils.py              # Common utility functions and helpers
        ├── news/                 # Auto-generated news JSON files
        │   └── news_[sortable]_[timestamp].json
        ├── sentiments/           # Auto-generated sentiment analysis results
        │   └── sentiments_[sortable]_[timestamp].json
        ├── charts/               # Auto-generated sentiment charts
        │   └── chart_[sortable]_[timestamp].png
        └── consistency/          # Auto-generated consistency reports
            └── consistency_[sortable]_[timestamp]/
                ├── consistency_[sortable]_[timestamp].json
                ├── consistency_summary_[sortable]_[timestamp].csv
                ├── consistency_report_[sortable]_[timestamp].html
                └── *.png charts
```

### Module Architecture

- **`__init__.py`**: Clean public API with 9 exported functions
- **Core Modules**: Separated concerns for fetching, analysis, and visualization
- **Configuration**: Flexible client management for multiple AI providers
- **Utilities**: Shared functionality for file operations, logging, and data handling
- **Type Safety**: Full type annotations throughout the codebase

## Pipeline Overview

The system follows a fully automated 4-stage pipeline:

### 1. News Fetching (`news_rss_fetcher.py`)
- Fetches Bitcoin news from Google RSS feeds
- Saves structured JSON with timestamped filenames
- Supports configurable article count and search queries

### 2. Sentiment Analysis (`sentiment_analyzer.py`)
- Auto-detects the latest news file
- Analyzes each article using LLM with trading-focused prompts
- Outputs sentiment scores (1-10) with detailed reasoning
- Saves results with matching timestamps for easy tracking

### 3. Chart Generation (`sentiment_grapher.py`)
- Auto-detects the latest sentiment analysis file
- Creates time-series visualizations with rolling averages
- Supports multiple chart types and time window filtering
- Saves high-quality PNG charts with matching timestamps

### 4. Consistency Testing (`consistency_tester.py`)
- Auto-detects the latest news file for testing
- Runs multiple analysis iterations to test reliability
- Provides comprehensive statistical analysis (CV, std dev, consistency rates)
- Generates detailed reports in JSON, CSV, and HTML formats

### Automated File Management
- **Sortable Timestamps**: Files use `99999999999999 - unix_timestamp` for reverse chronological sorting
- **Auto-detection**: Each pipeline stage automatically finds the latest input file
- **Consistent Naming**: All files use the same timestamp format for easy tracking
- **Organized Output**: Separate directories for each pipeline stage

## Installation

### Production Installation

Install dependencies using Poetry:

```bash
poetry install
```

### Development Setup

For development with code quality tools and pre-commit hooks:

```bash
# Install with development dependencies
poetry install --extras dev

# Set up pre-commit hooks for automated code quality
poetry run pre-commit install

# Optional: Run code quality tools manually
poetry run black src/      # Code formatting
poetry run isort src/      # Import sorting
poetry run ruff check src/  # Linting
poetry run mypy src/       # Type checking
```

### Code Quality Tools

The project includes comprehensive code quality tooling:

- **Black**: Code formatting (line length: 88)
- **isort**: Import organization and sorting
- **ruff**: Fast Python linter with 40+ rule categories
- **mypy**: Strict type checking with Pydantic support
- **pre-commit**: Automated code quality enforcement

### Requirements

- Python 3.12 or higher
- Poetry for dependency management
- Environment variables for AI provider configuration (see Configuration section)

## Quick Start

The complete pipeline can be run with these four simple commands:

### 1. Fetch Latest Bitcoin News

```bash
poetry run python src/sentiment_analysis/news_rss_fetcher.py --count 20
```

This fetches the latest 20 Bitcoin news articles and saves them to `src/sentiment_analysis/news/` with a timestamped filename.

#### Advanced News Fetching with Content

The news fetcher can optionally fetch full article content using SearXNG search:

```bash
# Fetch with article content (requires SearXNG instance)
poetry run python src/sentiment_analysis/news_rss_fetcher.py --count 10 --searxng-url "https://your-searxng-instance.com"

# Fetch without content (faster, default behavior)
poetry run python src/sentiment_analysis/news_rss_fetcher.py --count 10 --no-content

# Custom delay between requests (be respectful to SearXNG instance)
poetry run python src/sentiment_analysis/news_rss_fetcher.py --count 10 --request-delay 2.0
```

**Content Fetching Features:**
- **SearXNG Integration**: Uses article titles as search queries to find full content
- **Rate Limiting**: Configurable delays between requests to respect SearXNG instances
- **Error Handling**: Graceful fallback when content fetching fails
- **Backward Compatibility**: Works with or without content fetching

**Configuration:**
- Set `SEARXNG_BASE_URL` environment variable or use `--searxng-url` flag
- Default: `http://localhost:8080` (local SearXNG instance)
- Use `--no-content` to disable content fetching for faster execution

### 2. Analyze Sentiment

```bash
poetry run python src/sentiment_analysis/sentiment_analyzer.py
```

Automatically detects the latest news file and analyzes each article's sentiment using the LLM. Results are saved to `src/sentiment_analysis/sentiments/` with matching timestamps.

### 3. Generate Charts

```bash
poetry run python src/sentiment_analysis/sentiment_grapher.py
```

Creates time-series visualizations from the latest sentiment data. Charts are saved to `src/sentiment_analysis/charts/` with matching timestamps.

### 4. Run Consistency Testing

```bash
poetry run python src/sentiment_analysis/consistency_tester.py --iterations 5
```

Tests the reliability of the sentiment analyzer by running multiple iterations. Generates comprehensive statistical reports in `src/sentiment_analysis/consistency/`.

### Advanced Options

```bash
# Custom search query and article count
poetry run python src/sentiment_analysis/news_rss_fetcher.py --query "cryptocurrency" --count 50

# Specific time window for charts
poetry run python src/sentiment_analysis/sentiment_grapher.py --window 60

# More extensive consistency testing
poetry run python src/sentiment_analysis/consistency_tester.py --iterations 10 --articles 15
```

## Configuration

### Environment Variables

Set up your environment variables in `.env` (create this file in your project root):

```env
# OpenRouter Configuration (Recommended)
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_MODEL_ID=anthropic/claude-3-haiku

# LMStudio Configuration (Optional Local Alternative)
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_API_KEY=123
LMSTUDIO_MODEL_ID=your_model_here

# Choose provider: "openrouter" or "lmstudio"
USE_LMSTUDIO=false

# SearXNG Configuration (for article content fetching)
SEARXNG_BASE_URL=http://localhost:8080

# Optional: Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(levelname)s - %(message)s
```

### Configuration Options

#### AI Provider Selection

**OpenRouter (Recommended)**
- **Base URL**: `https://openrouter.ai/api/v1`
- **Models**: Access to 100+ models including Claude, GPT-4, Llama
- **Advantages**: Reliable, scalable, multiple model options
- **Use Case**: Production environments, model comparison

**LMStudio (Local Alternative)**
- **Base URL**: `http://localhost:1234/v1` (default)
- **Models**: Local GGUF models
- **Advantages**: Privacy, offline usage, no API costs
- **Use Case**: Development, privacy-sensitive applications

#### Model Selection

**Recommended Models for Sentiment Analysis:**
```env
# High Quality (slower, more expensive)
OPENROUTER_MODEL_ID=qwen/qwen3-next-80b-a3b-thinking
OPENROUTER_MODEL_ID=qwen/qwen3-next-80b-a3b-instruct

# Balanced (good speed/quality ratio)
OPENROUTER_MODEL_ID=openai/gpt-4o-mini
OPENROUTER_MODEL_ID=openai/gpt-4o

# Fast (lower cost, good for batch processing)
OPENROUTER_MODEL_ID=meta-llama/llama-3.1-8b-instruct
OPENROUTER_MODEL_ID=ibm-granite/granite-4.0-h-micro
```

#### SearXNG Integration

**For Enhanced Article Content:**
```env
# Public instances (rate limited)
SEARXNG_BASE_URL=https://searx.be
SEARXNG_BASE_URL=https://search.brave.com

# Self-hosted (recommended for production)
SEARXNG_BASE_URL=http://your-searxng-instance:8080
```

### Advanced Configuration

#### Custom Client Configuration

```python
from sentiment_analysis import build_client

# Custom OpenRouter configuration
client = build_client(
    use_lmstudio=False,
    config={
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "your-custom-key",
        "model": "anthropic/claude-3.5-sonnet",
        "timeout": 60,  # seconds
        "max_retries": 3,
        "temperature": 0.1,  # Lower temperature for consistent sentiment analysis
    }
)

# Custom LMStudio configuration
client = build_client(
    use_lmstudio=True,
    config={
        "base_url": "http://localhost:1234/v1",
        "api_key": "123",
        "model": "your-local-model",
        "timeout": 120,  # Local models may need more time
    }
)
```

#### Logging Configuration

```python
from sentiment_analysis import setup_logging
import logging

# Production configuration (minimal output)
logger = setup_logging(
    name="sentiment_analysis",
    level=logging.WARNING,
    format_string="%(levelname)s - %(message)s"
)

# Development configuration (verbose output)
logger = setup_logging(
    name="sentiment_analysis",
    level=logging.DEBUG,
    format_string="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
```

### Environment File Templates

#### Production Template (.env.production)
```env
# Production configuration
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_API_KEY=sk-or-v1-prod-...
OPENROUTER_MODEL_ID=anthropic/claude-3-haiku
USE_LMSTUDIO=false

# Production logging
LOG_LEVEL=WARNING
LOG_FORMAT=%(levelname)s - %(message)s

# Self-hosted SearXNG
SEARXNG_BASE_URL=https://your-searxng-domain.com
```

#### Development Template (.env.development)
```env
# Development configuration (faster, cheaper model)
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_API_KEY=sk-or-v1-dev-...
OPENROUTER_MODEL_ID=openai/gpt-4o-mini
USE_LMSTUDIO=false

# Development logging
LOG_LEVEL=DEBUG
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s

# Public SearXNG instance
SEARXNG_BASE_URL=https://searx.be
```

#### Local Development Template (.env.local)
```env
# Local LMStudio configuration
USE_LMSTUDIO=true
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_API_KEY=123
LMSTUDIO_MODEL_ID=llama-3.1-8b-instruct

# Debug logging
LOG_LEVEL=DEBUG
```

### Configuration Validation

```python
import os
from pathlib import Path

def validate_configuration():
    """Validate that required environment variables are set."""

    # Check .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        raise FileNotFoundError("Create .env file with your API configuration")

    # Validate required variables
    required_vars = ["OPENROUTER_API_KEY"] if not os.getenv("USE_LMSTUDIO") == "true" else ["LMSTUDIO_BASE_URL"]

    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")

    print("✅ Configuration is valid")

# Validate on startup
validate_configuration()
```

### Getting API Keys

1. **OpenRouter**: Sign up at [openrouter.ai](https://openrouter.ai) and get your API key from the dashboard
2. **LMStudio**: Download [LM Studio](https://lmstudio.ai), run it locally, and use the default settings

The system defaults to OpenRouter for production use. Switch to LMStudio for local, private processing.

## API Reference

The library provides a clean, type-safe public API through the main package import:

```python
from sentiment_analysis import (
    # Core analysis functions
    analyze_article,
    analyze_articles_batch,

    # Data fetching
    fetch_news_rss,

    # Visualization
    generate_sentiment_charts,

    # Configuration and utilities
    build_client,
    get_sentiment_analysis_prompt_with_context,
    setup_logging,

    # Data models
    SentimentAnalysis,
    ArticleWithSentiment,
)
```

### Core Analysis Functions

#### `analyze_article(title, body, client)`

Analyzes a single article for sentiment.

```python
from sentiment_analysis import analyze_article, build_client

# Create AI client
client = build_client()

# Analyze single article
sentiment = analyze_article(
    title="Bitcoin ETF Approval Expected This Week",
    body="The SEC is expected to approve spot Bitcoin ETFs...",
    client=client
)

print(f"Score: {sentiment.score}")
print(f"Reasoning: {sentiment.reasoning}")
```

**Parameters:**
- `title` (str): Article title
- `body` (str | None): Article body content (optional)
- `client` (Instructor): AI client instance

**Returns:** `SentimentAnalysis` object with score (1-10) and reasoning.

#### `analyze_articles_batch(articles)`

Analyzes multiple articles efficiently.

```python
from sentiment_analysis import analyze_articles_batch

articles = [
    {
        "title": "Bitcoin reaches new all-time high",
        "body": "Bitcoin has surged past previous records...",
        "url": "https://example.com/article1",
        "timestamp": "2024-01-01T12:00:00Z",
        "unix_timestamp": 1704110400
    },
    # ... more articles
]

results = analyze_articles_batch(articles)
print(f"Analyzed {len(results)} articles")
```

**Parameters:**
- `articles` (list[dict]): List of article dictionaries with title, body, url, timestamp, unix_timestamp

**Returns:** `list[dict]` - List of analysis results with sentiment data.

### Data Fetching Functions

#### `fetch_news_rss(query, count, searxng_url, no_content, request_delay)`

Fetches Bitcoin news from RSS feeds with optional content fetching.

```python
from sentiment_analysis import fetch_news_rss

# Basic news fetching
articles = fetch_news_rss(
    query="bitcoin",
    count=20
)

# Advanced fetching with content
articles = fetch_news_rss(
    query="bitcoin",
    count=10,
    searxng_url="https://your-searxng-instance.com",
    no_content=False,
    request_delay=2.0
)
```

**Parameters:**
- `query` (str, default="bitcoin"): Search query for RSS feed
- `count` (int, default=10): Number of articles to fetch
- `searxng_url` (str | None): SearXNG instance URL for content fetching
- `no_content` (bool, default=False): Skip content fetching
- `request_delay` (float, default=0.0): Delay between requests in seconds

**Returns:** `list[dict]` - List of article dictionaries with metadata.

### Visualization Functions

#### `generate_sentiment_charts(records, window_minutes, interval_minutes, title, dpi, max_points)`

Creates high-quality sentiment charts from analysis data.

```python
from sentiment_analysis import generate_sentiment_charts

# Assuming you have sentiment analysis results
sentiment_data = [
    {
        "title": "Article 1",
        "sentiment_score": 7.5,
        "timestamp": "2024-01-01T12:00:00Z",
        "unix_timestamp": 1704110400
    },
    # ... more data
]

# Generate charts
charts = generate_sentiment_charts(
    records=sentiment_data,
    window_minutes=5,
    interval_minutes=60,
    title="Bitcoin Sentiment Analysis - Jan 2024",
    dpi=300,
    max_points=400
)

print(f"Generated {len(charts)} charts")
```

**Parameters:**
- `records` (list[dict]): Sentiment analysis records
- `window_minutes` (int, default=5): Rolling average window in minutes
- `interval_minutes` (str | int, default="60"): Time window for data filtering
- `title` (str | None): Custom chart title
- `dpi` (int, default=300): Image resolution
- `max_points` (int, default=400): Maximum data points per chart

**Returns:** `list[str]` - List of base64-encoded PNG chart images.

### Configuration Functions

#### `build_client(use_lmstudio, config)`

Creates an AI client with flexible configuration.

```python
from sentiment_analysis import build_client

# Use default configuration (OpenRouter)
client = build_client()

# Use LMStudio for local processing
client = build_client(use_lmstudio=True)

# Custom configuration
client = build_client(
    use_lmstudio=False,
    config={
        "mode": "json",
        "temperature": 0.1
    }
)
```

**Parameters:**
- `use_lmstudio` (bool | None): Force LMStudio client
- `config` (dict | None): Custom configuration dictionary

**Returns:** `Instructor` - Configured AI client instance.

#### `get_sentiment_analysis_prompt_with_context(title, body)`

Gets the formatted sentiment analysis prompt for custom usage.

```python
from sentiment_analysis import get_sentiment_analysis_prompt_with_context

prompt = get_sentiment_analysis_prompt_with_context(
    title="Bitcoin Price Surges",
    body="Bitcoin has experienced a significant price increase..."
)
print(prompt)
```

**Parameters:**
- `title` (str): Article title
- `body` (str): Article body content

**Returns:** `str` - Formatted prompt string.

#### `setup_logging(name, level, format_string)`

Configures standardized logging for the application.

```python
import logging
from sentiment_analysis import setup_logging

# Default logging setup
logger = setup_logging(__name__)

# Custom logging setup
logger = setup_logging(
    name="my_app",
    level=logging.DEBUG,
    format_string="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
```

**Parameters:**
- `name` (str | None): Logger name
- `level` (int, default=logging.INFO): Logging level
- `format_string` (str | None): Custom log format

**Returns:** `logging.Logger` - Configured logger instance.

### Data Models

#### `SentimentAnalysis`

Pydantic model for sentiment analysis results.

```python
from sentiment_analysis import SentimentAnalysis

# Create sentiment analysis
sentiment = SentimentAnalysis(
    score=8.5,
    reasoning="Strong bullish signal from institutional adoption news"
)

print(f"Score: {sentiment.score}")
print(f"Reasoning: {sentiment.reasoning}")
```

**Fields:**
- `score` (float): Sentiment score from 1.0 to 10.0
- `reasoning` (str): Trading-focused explanation for the score

#### `ArticleWithSentiment`

Pydantic model for articles with sentiment data.

```python
from sentiment_analysis import ArticleWithSentiment

article = ArticleWithSentiment(
    title="Bitcoin News",
    body="Article content...",
    timestamp="2024-01-01T12:00:00Z",
    url="https://example.com",
    unix_timestamp=1704110400,
    sentiment=SentimentAnalysis(score=7.5, reasoning="...")
)
```

**Fields:**
- `title` (str): Article title
- `body` (str | None): Article body content
- `timestamp` (str): Publication timestamp
- `url` (str): Article URL
- `unix_timestamp` (int | None): Unix timestamp for sorting
- `sentiment` (SentimentAnalysis | None): Sentiment analysis results

## Programmatic Usage Examples

### Complete Analysis Workflow

Here's a complete example of using the library programmatically:

```python
from sentiment_analysis import (
    fetch_news_rss,
    analyze_articles_batch,
    generate_sentiment_charts,
    build_client,
    setup_logging
)

# Set up logging
logger = setup_logging(__name__)

# 1. Fetch Bitcoin news
logger.info("Fetching Bitcoin news...")
articles = fetch_news_rss(
    query="bitcoin",
    count=20,
    searxng_url="https://your-searxng-instance.com",
    no_content=False,
    request_delay=1.0
)

logger.info(f"Fetched {len(articles)} articles")

# 2. Analyze sentiment
logger.info("Analyzing sentiment...")
results = analyze_articles_batch(articles)
logger.info(f"Analyzed {len(results)} articles")

# 3. Generate charts
logger.info("Generating charts...")
charts = generate_sentiment_charts(
    records=results,
    window_minutes=5,
    interval_minutes=60,
    title="Bitcoin Sentiment Analysis"
)
logger.info(f"Generated {len(charts)} charts")

# 4. Display results
for result in results[:5]:  # Show first 5 results
    sentiment = result['sentiment']
    print(f"Title: {result['title'][:50]}...")
    print(f"Score: {sentiment['score']}")
    print(f"Reasoning: {sentiment['reasoning']}")
    print("-" * 50)
```

### Custom Configuration Examples

#### Using Different AI Providers

```python
from sentiment_analysis import build_client

# OpenRouter (default for production)
openrouter_client = build_client()

# LMStudio (local processing)
lmstudio_client = build_client(use_lmstudio=True)

# Custom configuration
custom_client = build_client(
    use_lmstudio=False,
    config={
        "mode": "json",
        "temperature": 0.1,
        "model": "anthropic/claude-3-haiku"
    }
)
```

#### Advanced Analysis with Custom Prompts

```python
from sentiment_analysis import (
    analyze_article,
    get_sentiment_analysis_prompt_with_context,
    build_client
)

# Create client
client = build_client()

# Get custom prompt
custom_prompt = get_sentiment_analysis_prompt_with_context(
    title="Bitcoin Regulation News",
    body="Regulatory bodies announced new guidelines..."
)

# Use with custom client configuration
from instructor import Instructor
import openai

# Create custom client
openai_client = openai.OpenAI(api_key="your-key")
custom_client = instructor.from_openai(openai_client)

# Analyze with custom setup
sentiment = analyze_article(
    title="Bitcoin Regulation News",
    body="Regulatory bodies announced new guidelines...",
    client=custom_client
)
```

### Data Processing and Integration

#### Processing Existing Data

```python
import json
from sentiment_analysis import analyze_articles_batch, SentimentAnalysis

# Load existing articles
with open('existing_articles.json', 'r') as f:
    articles = json.load(f)

# Filter articles (example: only recent ones)
from datetime import datetime, timedelta
import time

recent_articles = [
    article for article in articles
    if article.get('unix_timestamp', 0) > time.time() - 86400  # Last 24 hours
]

# Analyze only recent articles
results = analyze_articles_batch(recent_articles)

# Process results
bullish_articles = []
bearish_articles = []
neutral_articles = []

for result in results:
    sentiment = result['sentiment']
    score = sentiment['score']

    if score >= 7.0:
        bullish_articles.append(result)
    elif score <= 4.0:
        bearish_articles.append(result)
    else:
        neutral_articles.append(result)

print(f"Bullish: {len(bullish_articles)}")
print(f"Neutral: {len(neutral_articles)}")
print(f"Bearish: {len(bearish_articles)}")
```

#### Integration with External Systems

```python
import pandas as pd
from sentiment_analysis import fetch_news_rss, analyze_articles_batch

# Fetch and analyze news
articles = fetch_news_rss(query="bitcoin", count=50)
results = analyze_articles_batch(articles)

# Convert to DataFrame for analysis
df = pd.DataFrame([
    {
        'title': result['title'],
        'score': result['sentiment']['score'],
        'reasoning': result['sentiment']['reasoning'],
        'timestamp': result['timestamp'],
        'url': result['url']
    }
    for result in results
])

# Statistical analysis
print(f"Average Score: {df['score'].mean():.2f}")
print(f"Score Distribution:")
print(df['score'].describe())

# Export to CSV
df.to_csv('sentiment_analysis_results.csv', index=False)

# Find most bullish and bearish articles
most_bullish = df.loc[df['score'].idxmax()]
most_bearish = df.loc[df['score'].idxmin()]

print(f"\nMost Bullish: {most_bullish['title']} (Score: {most_bullish['score']})")
print(f"Most Bearish: {most_bearish['title']} (Score: {most_bearish['score']})")
```

### Real-time Monitoring Example

```python
import time
from sentiment_analysis import fetch_news_rss, analyze_articles_batch
from datetime import datetime

def monitor_sentiment(check_interval_minutes=30):
    """Continuously monitor Bitcoin sentiment."""

    while True:
        print(f"\n[{datetime.now()}] Checking Bitcoin sentiment...")

        try:
            # Fetch latest news
            articles = fetch_news_rss(count=10)

            if not articles:
                print("No articles found. Retrying...")
                time.sleep(check_interval_minutes * 60)
                continue

            # Analyze sentiment
            results = analyze_articles_batch(articles)

            # Calculate average sentiment
            scores = [r['sentiment']['score'] for r in results]
            avg_score = sum(scores) / len(scores)

            # Classify overall sentiment
            if avg_score >= 7.0:
                sentiment_trend = "🟢 STRONG BULLISH"
            elif avg_score <= 4.0:
                sentiment_trend = "🔴 STRONG BEARISH"
            else:
                sentiment_trend = "⚪ NEUTRAL"

            print(f"Articles: {len(results)}")
            print(f"Average Score: {avg_score:.2f}")
            print(f"Overall Trend: {sentiment_trend}")

            # Show top articles
            top_articles = sorted(results,
                key=lambda x: x['sentiment']['score'],
                reverse=True)[:3]

            print("\nTop 3 Articles:")
            for i, article in enumerate(top_articles, 1):
                sentiment = article['sentiment']
                print(f"{i}. {article['title'][:60]}... (Score: {sentiment['score']})")

        except Exception as e:
            print(f"Error: {e}")

        print(f"Next check in {check_interval_minutes} minutes...")
        time.sleep(check_interval_minutes * 60)

# Run monitoring (commented out for example)
# monitor_sentiment(check_interval_minutes=15)
```

### Batch Processing with Custom Logic

```python
from sentiment_analysis import fetch_news_rss, analyze_articles_batch
import json
from pathlib import Path

def process_multiple_search_terms():
    """Process multiple search terms and save results separately."""

    search_terms = ["bitcoin", "cryptocurrency", "blockchain", "ethereum", "defi"]

    results_dir = Path("analysis_results")
    results_dir.mkdir(exist_ok=True)

    for term in search_terms:
        print(f"Processing: {term}")

        # Fetch news for this term
        articles = fetch_news_rss(query=term, count=15)

        if not articles:
            print(f"No articles found for {term}")
            continue

        # Analyze sentiment
        results = analyze_articles_batch(articles)

        # Save results
        output_file = results_dir / f"{term}_sentiment_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Calculate statistics
        scores = [r['sentiment']['score'] for r in results]
        avg_score = sum(scores) / len(scores)

        print(f"  Articles: {len(results)}")
        print(f"  Average Score: {avg_score:.2f}")
        print(f"  Saved to: {output_file}")

# Run batch processing
# process_multiple_search_terms()
```

## Advanced Examples and Use Cases

### Real-time Sentiment Monitoring

```python
import time
from datetime import datetime
from sentiment_analysis import fetch_news_rss, analyze_articles_batch, generate_sentiment_charts

def real_time_sentiment_monitor(interval_minutes=30, max_cycles=10):
    """Monitor Bitcoin sentiment in real-time with configurable intervals."""

    cycle = 0
    while cycle < max_cycles:
        cycle += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"\n{'='*50}")
        print(f"Cycle {cycle}/{max_cycles} - {timestamp}")
        print(f"{'='*50}")

        # Fetch latest news
        articles = fetch_news_rss(query="bitcoin", count=20)

        if not articles:
            print("No articles found in this cycle")
            time.sleep(interval_minutes * 60)
            continue

        # Analyze sentiment
        results = analyze_articles_batch(articles)

        # Calculate statistics
        scores = [r['sentiment']['score'] for r in results]
        avg_score = sum(scores) / len(scores)

        # Determine signal
        if avg_score >= 8.0:
            signal = "🟢 STRONG BUY"
        elif avg_score >= 6.1:
            signal = "🟢 WEAK BUY"
        elif avg_score >= 5.1:
            signal = "⚪ NEUTRAL/HOLD"
        elif avg_score >= 3.1:
            signal = "🟡 WEAK SELL"
        else:
            signal = "🔴 STRONG SELL"

        print(f"Articles Analyzed: {len(results)}")
        print(f"Average Score: {avg_score:.2f}")
        print(f"Signal: {signal}")
        print(f"Score Range: {min(scores):.1f} - {max(scores):.1f}")

        # Wait for next cycle
        if cycle < max_cycles:
            print(f"Waiting {interval_minutes} minutes until next cycle...")
            time.sleep(interval_minutes * 60)

# Run real-time monitoring
# real_time_sentiment_monitor(interval_minutes=15, max_cycles=5)
```

### Comparative Analysis

```python
import pandas as pd
from sentiment_analysis import fetch_news_rss, analyze_articles_batch

def comparative_sentiment_analysis(terms, articles_per_term=15):
    """Compare sentiment across different search terms."""

    results_summary = {}

    for term in terms:
        print(f"Analyzing: {term}")

        # Fetch and analyze articles
        articles = fetch_news_rss(query=term, count=articles_per_term)
        if not articles:
            print(f"No articles found for {term}")
            continue

        results = analyze_articles_batch(articles)
        scores = [r['sentiment']['score'] for r in results]

        # Calculate statistics
        results_summary[term] = {
            'count': len(results),
            'avg_score': sum(scores) / len(scores),
            'min_score': min(scores),
            'max_score': max(scores),
            'std_dev': pd.Series(scores).std(),
            'strong_buy': len([s for s in scores if s >= 8.0]),
            'weak_buy': len([s for s in scores if 6.1 <= s < 8.0]),
            'neutral': len([s for s in scores if 5.1 <= s < 6.1]),
            'weak_sell': len([s for s in scores if 3.1 <= s < 5.0]),
            'strong_sell': len([s for s in scores if s < 3.0])
        }

    # Create comparison DataFrame
    df = pd.DataFrame(results_summary).T
    df = df.sort_values('avg_score', ascending=False)

    print("\nComparative Sentiment Analysis:")
    print("=" * 80)
    print(f"{'Term':<15} {'Articles':<10} {'Avg Score':<10} {'Signal':<12} {'Volatility':<10}")
    print("-" * 80)

    for term, data in df.iterrows():
        score = data['avg_score']
        if score >= 8.0:
            signal = "STRONG BUY"
        elif score >= 6.1:
            signal = "WEAK BUY"
        elif score >= 5.1:
            signal = "NEUTRAL"
        elif score >= 3.1:
            signal = "WEAK SELL"
        else:
            signal = "STRONG SELL"

        print(f"{term:<15} {int(data['count']):<10} {score:<10.2f} {signal:<12} {data['std_dev']:<10.2f}")

    return df

# Run comparative analysis
# terms = ["bitcoin", "ethereum", "cryptocurrency", "blockchain", "defi"]
# comparison_df = comparative_sentiment_analysis(terms)
```

### Sentiment Trend Analysis

```python
import matplotlib.pyplot as plt
import pandas as pd
from sentiment_analysis import fetch_news_rss, analyze_articles_batch
from datetime import datetime, timedelta

def sentiment_trend_analysis(days_back=7, articles_per_day=10):
    """Analyze sentiment trends over time."""

    trend_data = []

    for i in range(days_back):
        target_date = datetime.now() - timedelta(days=i)
        date_str = target_date.strftime("%Y-%m-%d")

        print(f"Analyzing sentiment for {date_str}")

        # Fetch articles (in real implementation, you'd fetch from specific dates)
        articles = fetch_news_rss(query="bitcoin", count=articles_per_day)

        if articles:
            results = analyze_articles_batch(articles)
            scores = [r['sentiment']['score'] for r in results]

            trend_data.append({
                'date': date_str,
                'avg_score': sum(scores) / len(scores),
                'article_count': len(scores),
                'min_score': min(scores),
                'max_score': max(scores)
            })

    # Convert to DataFrame and sort by date
    df = pd.DataFrame(trend_data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # Create trend visualization
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['avg_score'], marker='o', linewidth=2, markersize=6)
    plt.axhline(y=5.0, color='gray', linestyle='--', alpha=0.7, label='Neutral (5.0)')
    plt.axhline(y=7.0, color='green', linestyle='--', alpha=0.7, label='Buy Threshold (7.0)')
    plt.axhline(y=3.0, color='red', linestyle='--', alpha=0.7, label='Sell Threshold (3.0)')

    plt.title('Bitcoin Sentiment Trend Analysis', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Sentiment Score', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save chart
    chart_file = f"sentiment_trend_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Trend chart saved as: {chart_file}")
    return df

# Run trend analysis
# trend_df = sentiment_trend_analysis(days_back=14)
```

### Custom Sentiment Scoring

```python
from sentiment_analysis import analyze_article, build_client, get_sentiment_analysis_prompt_with_context

def custom_sentiment_analysis(title, body, custom_factors=None):
    """Perform sentiment analysis with custom criteria."""

    client = build_client()

    # Create custom prompt with additional context
    custom_context = """
    Analyze this Bitcoin news from a DeFi trader's perspective with emphasis on:
    1. Smart contract implications
    2. DeFi protocol integration potential
    3. Yield farming opportunities
    4. Cross-chain compatibility
    5. Governance token implications

    Provide a score from 1.0-10.0 focusing on DeFi ecosystem impact.
    """

    # Get custom prompt
    custom_prompt = get_sentiment_analysis_prompt_with_context(custom_context)

    # Perform analysis with custom prompt
    result = analyze_article(title, body, client)

    return result

# Example usage
# custom_result = custom_sentiment_analysis(
#     title="Bitcoin Layer 2 Solution Integrates with DeFi Protocols",
#     body="A new Bitcoin layer 2 scaling solution announces integration..."
# )
# print(f"Custom DeFi-focused score: {custom_result.score}")
```

### Integration with Trading Systems

```python
import json
import time
from sentiment_analysis import fetch_news_rss, analyze_articles_batch

class TradingSignalGenerator:
    """Generate trading signals based on sentiment analysis."""

    def __init__(self, buy_threshold=7.0, sell_threshold=3.0, confidence_threshold=0.7):
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.confidence_threshold = confidence_threshold
        self.signal_history = []

    def generate_signal(self, articles):
        """Generate trading signal from article analysis."""

        results = analyze_articles_batch(articles)
        scores = [r['sentiment']['score'] for r in results]

        avg_score = sum(scores) / len(scores)
        confidence = min(len(scores) / 10, 1.0)  # Confidence based on sample size

        # Generate signal
        if avg_score >= self.buy_threshold and confidence >= self.confidence_threshold:
            signal = "BUY"
            strength = min((avg_score - self.buy_threshold) / 3.0, 1.0)
        elif avg_score <= self.sell_threshold and confidence >= self.confidence_threshold:
            signal = "SELL"
            strength = min((self.sell_threshold - avg_score) / 3.0, 1.0)
        else:
            signal = "HOLD"
            strength = 0.5

        signal_data = {
            'timestamp': datetime.now().isoformat(),
            'signal': signal,
            'strength': strength,
            'avg_score': avg_score,
            'confidence': confidence,
            'article_count': len(results),
            'score_distribution': {
                'strong_buy': len([s for s in scores if s >= 8.0]),
                'weak_buy': len([s for s in scores if 6.1 <= s < 8.0]),
                'neutral': len([s for s in scores if 5.1 <= s < 6.1]),
                'weak_sell': len([s for s in scores if 3.1 <= s < 5.0]),
                'strong_sell': len([s for s in scores if s < 3.0])
            }
        }

        self.signal_history.append(signal_data)
        return signal_data

    def save_signals(self, filename):
        """Save signal history to file."""
        with open(filename, 'w') as f:
            json.dump(self.signal_history, f, indent=2)

# Usage example
# signal_generator = TradingSignalGenerator(buy_threshold=7.5, sell_threshold=3.5)
# articles = fetch_news_rss(query="bitcoin ETF", count=25)
# trading_signal = signal_generator.generate_signal(articles)
# print(f"Trading Signal: {trading_signal['signal']} (Strength: {trading_signal['strength']:.2f})")
```

## Sentiment Score Framework

| Score Range | Signal | Description | Examples |
|-------------|--------|-------------|----------|
| 1.0-3.0 | 🔴 **STRONG SELL** | Major negative catalysts | Regulatory crackdowns, security breaches, institutional abandonment |
| 3.1-5.0 | 🟡 **WEAK SELL** | Moderate negative factors | Negative price action, minor regulatory concerns, reduced adoption |
| 5.1-6.0 | ⚪ **NEUTRAL/HOLD** | Mixed signals or routine movements | Technical consolidation, mixed news, no clear catalysts |
| 6.1-8.0 | 🟢 **WEAK BUY** | Positive indicators | Favorable developments, institutional interest, bullish technicals |
| 8.1-10.0 | 🟢 **STRONG BUY** | Major positive catalysts | Major institutional adoption, favorable regulation, breakthrough developments |

## Output Format

### News Files
Each news article contains:
```json
{
  "title": "Article title",
  "url": "Article URL",
  "timestamp": "Publication timestamp",
  "unix_timestamp": 1735089382,
  "source": "News Source Name",
  "body": "Full article content fetched via SearXNG (optional, may be empty)"
}
```

### Sentiment Files
Each analyzed article contains:
```json
{
  "title": "Article title",
  "url": "Article URL",
  "timestamp": "Publication timestamp",
  "unix_timestamp": 1735089382,
  "source": "News Source Name",
  "sentiment": {
    "score": 7.5,
    "reasoning": "Concise trading-focused explanation for the score"
  }
}
```

## Analysis Factors

The model considers:

- **Market Impact**: Effect on Bitcoin price and trading volume
- **Regulatory Environment**: Positive or negative regulatory implications
- **Institutional Adoption**: Signs of increased/decreased institutional interest
- **Technical Indicators**: Price levels, support/resistance, market structure
- **Market Sentiment**: Fear/greed indicators, social media sentiment
- **Adoption Metrics**: Real-world usage, merchant acceptance, network effects

## Statistical Analysis Features

The consistency tester provides comprehensive statistical analysis:

- **Coefficient of Variation (CV)**: Measures relative variability
- **Standard Deviation**: Absolute dispersion measure
- **Consistency Rate**: Percentage of scores within consistency threshold
- **Confidence Intervals**: Statistical confidence ranges
- **Range Analysis**: Score spread and outlier detection
- **Robust Statistics**: Median-based measures for outlier resistance

### Statistical Output Formats

- **JSON**: Complete statistical data for programmatic use
- **CSV**: Summary tables for spreadsheet analysis
- **HTML**: Interactive reports with visualizations
- **PNG Charts**: Statistical distribution and trend charts

## Error Handling

The system includes comprehensive error handling:

- **API Failures**: Falls back to neutral sentiment (5.0) with error explanation
- **Invalid Data**: Skips malformed articles with logging
- **Network Issues**: Retry logic for transient failures
- **Validation**: Ensures scores stay within 1.0-10.0 range
- **Missing Files**: Graceful handling of missing input files
- **Statistical Errors**: Fallback methods for failed calculations

## Performance and Optimization

### Benchmarks

- **Analysis Speed**: ~2-3 seconds per article (varies by model complexity)
- **Batch Processing**: Linear scaling with efficient memory usage
- **Memory Footprint**: ~50-100MB base usage + ~10MB per 100 articles
- **Statistical Analysis**: ~5-10 seconds for 1000 iterations (configurable)

### Optimization Features

- **Async Operations**: Non-blocking network requests for news fetching
- **Streaming Processing**: Articles processed in batches to minimize memory usage
- **Intelligent Caching**: RSS feed results cached for 30 minutes
- **Retry Logic**: Exponential backoff for failed requests (max 3 retries)
- **Connection Pooling**: Reused HTTP connections for multiple requests

### Scalability Considerations

- **Large Batch Processing**: Can handle 1000+ articles in single session
- **Statistical Testing**: Configurable iteration counts balance accuracy vs. speed
- **Memory Management**: Automatic cleanup of processed data
- **Rate Limiting**: Built-in delays prevent API throttling

### Performance Tuning

```python
# Example: Optimizing for large batches
from sentiment_analysis import analyze_articles_batch, setup_logging

logger = setup_logging(__name__, level=logging.WARNING)  # Reduce logging overhead

articles = fetch_news_rss(query="bitcoin", count=500)  # Large batch

# Process in smaller chunks to manage memory
chunk_size = 100
results = []
for i in range(0, len(articles), chunk_size):
    chunk = articles[i:i + chunk_size]
    chunk_results = analyze_articles_batch(chunk, logger=logger)
    results.extend(chunk_results)
    logger.info(f"Processed {i + len(chunk)}/{len(articles)} articles")
```

### Monitoring and Debugging

```python
# Enable performance monitoring
import time
from sentiment_analysis import analyze_articles_batch

start_time = time.time()
results = analyze_articles_batch(articles)
end_time = time.time()

print(f"Analyzed {len(results)} articles in {end_time - start_time:.2f} seconds")
print(f"Average: {(end_time - start_time) / len(results):.2f} seconds per article")
```

## Dependencies

### Core Dependencies
- `instructor`: Structured LLM outputs with Pydantic validation
- `python-dotenv`: Environment variable management for API configuration
- `matplotlib`: High-quality chart generation and data visualization
- `pandas`: Data manipulation and time-series analysis
- `numpy`: Numerical computing and statistical calculations
- `scipy`: Advanced statistical functions and scientific computing
- `statsmodels`: Statistical modeling and analysis
- `feedparser`: RSS feed parsing for news collection
- `aiohttp`: Asynchronous HTTP client for SearXNG search integration

### Development Dependencies
- `black`: Code formatting (line length: 88)
- `isort`: Import sorting and organization
- `ruff`: Fast Python linter with 40+ rule categories
- `mypy`: Static type checking with strict mode
- `pre-commit`: Git hooks for automated code quality

Note: `openai` is automatically included as a dependency of `instructor`.

## Troubleshooting and FAQ

### Common Issues

#### Q: Getting "API key not found" errors
**A:** Ensure your `.env` file is properly configured:
```bash
# Check if .env file exists in project root
ls -la .env

# Verify API key format
cat .env
# Should contain: OPENAI_API_KEY=sk-your-key-here
```

#### Q: Analysis returns all 5.0 (neutral) scores
**A:** This usually indicates API connectivity issues:
1. Check your internet connection
2. Verify API key validity
3. Check OpenAI API status at status.openai.com
4. Ensure sufficient API credits

#### Q: No news articles found
**A:** Try these solutions:
1. Use broader search terms (e.g., "bitcoin" instead of specific keywords)
2. Check RSS feed source availability
3. Verify network connectivity and firewall settings
4. Try different search terms or increase count parameter

#### Q: Memory errors with large batches
**A:** Process articles in smaller chunks:
```python
chunk_size = 50  # Reduce from default 100
for i in range(0, len(articles), chunk_size):
    chunk = articles[i:i + chunk_size]
    results = analyze_articles_batch(chunk)
```

#### Q: Slow processing speed
**A:** Optimize performance:
1. Use logging level WARNING to reduce overhead
2. Process in appropriate batch sizes (50-100 articles)
3. Ensure stable network connection
4. Consider API rate limits

### Error Messages Explained

#### "Failed to fetch RSS feed"
- **Cause**: Network issues or RSS source unavailability
- **Solution**: Check internet connection and try again later

#### "API request failed"
- **Cause**: OpenAI API issues or invalid key
- **Solution**: Verify API key and check OpenAI status

#### "Invalid JSON format"
- **Cause**: Corrupted input file or malformed data
- **Solution**: Validate input file format and content

#### "Statistical analysis failed"
- **Cause**: Insufficient data points or calculation errors
- **Solution**: Ensure at least 2 articles with valid sentiment scores

### Debug Mode

Enable detailed logging for troubleshooting:
```python
from sentiment_analysis import setup_logging

# Enable debug logging
logger = setup_logging(__name__, level=logging.DEBUG)

# Run analysis with verbose output
results = analyze_articles_batch(articles, logger=logger)
```

### Performance Issues

#### High Memory Usage
- Reduce batch size to 10 articles
- Close unused applications
- Monitor system resources during processing

#### Slow API Response
- Check network latency to OpenAI endpoints
- Try different OpenAI API regions if available
- Consider using faster models (gpt-4o-mini vs gpt-4o)

### Getting Help

1. **Check logs**: Error messages often contain specific guidance
2. **Verify configuration**: Ensure all environment variables are set
3. **Test connectivity**: Run simple API calls first
4. **Review examples**: Compare with working code in this README

### Recovery Procedures

**Corrupted Data Files**:
```bash
# Remove corrupted files
rm data/news/*.json
rm data/sentiments/*.json
rm data/charts/*.png

# Restart analysis with fresh data
python src/sentiment_analysis/__main__.py
```

**API Key Issues**:
```bash
# Reset environment variables
unset OPENAI_API_KEY

# Reload .env file
source .env

# Verify key is loaded
echo $OPENAI_API_KEY
```

## Code Quality and Development

### Development Workflow

This project follows modern Python development practices with automated code quality enforcement:

#### Pre-commit Hooks
```bash
# Install pre-commit hooks (automatic on first run)
pre-commit install

# Run all hooks manually
pre-commit run --all-files
```

#### Code Quality Tools
- **Black**: Opinionated code formatter (line-length: 88)
- **isort**: Import sorting with Black profile compatibility
- **ruff**: Fast linter covering 40+ rule categories
- **mypy**: Static type checking with strict mode

#### Code Style Standards
- Type hints for all function signatures
- Comprehensive docstrings with examples
- Consistent error handling patterns
- Modular architecture with clear separation of concerns

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd sentiment_analysis

# Install with development dependencies
poetry install --extras dev

# Set up pre-commit hooks
pre-commit install

# Run code quality checks
poetry run black src/ --check
poetry run isort src/ --check-only
poetry run ruff check src/
poetry run mypy src/
```

### Contributing Guidelines

1. **Code Quality**: All contributions must pass automated quality checks
2. **Type Safety**: Include comprehensive type hints
3. **Documentation**: Add docstrings for new functions and classes
4. **Testing**: Include tests for new functionality
5. **API Changes**: Maintain backward compatibility for public API

#### Before Submitting
```bash
# Run full quality check suite
poetry run black src/
poetry run isort src/
poetry run ruff check src/ --fix
poetry run mypy src/
pre-commit run --all-files
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
1. Check the demo script for usage examples
2. Review the code comments for detailed explanations
3. Check the logs for error messages and debug information
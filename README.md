# Bitcoin News Sentiment Analysis

## About

This project performs sentiment analysis of financial news articles using vector embeddings, similarity-based scoring, and structured LLM output. The system fetches news from RSS feeds, websockets, and social media. Then it processes the articles, and saves results to a SQL database efficiently.

## Installation

Install Python dependencies using Poetry.

Download, setup, and start SearXNG, PostgreSQL, and optionally LMStudio. For SearXNG, configure it to return results in JSON format and disable the rate limiter. For PostgreSQL, use an image with PostgreSQL version 18 that has pgvector installed, such as `pgvector/pgvector:pg18-trixie`. For LLM clients, either use a local LMStudio instance or connect to OpenRouter using your API key.

## Setup

Start SearXNG and PostgreSQL containers (and LMStudio if used).

Configure environment variables by renaming `.env.example` to `.env` and filling in your values.

Configure the configuration file by renaming `config.toml.example` to `config.toml` and filling in your values.

Run `db_utils.py` and choose option `setup_database` in the CLI. This will create a new preconfigured table called `articles`. Warning: this will drop the existing `articles` table before rebuilding it, so avoid data loss by preferably using a clean PostgreSQL instance.

## Usage

There are three main usage methods: run individual files for testing, run the `consistency_tester`, or run the `parallel_processor`.

### Individual Files

When running individual files in this order: `google_news` → `sentiment_analyzer` → `sentiment_grapher`, the system saves JSON files to respective folders (news, sentiment, charts). This approach is JSON file-based and is primarily used to test separate components to verify functionality.

The `google_news` loads a specified number of articles for a specified query. Article body loading is optional. Use either `searxng_search` or `smart_searxng_search` for body content. Default search uses a local SearXNG instance and gets rate limited quickly. Smart search loops through multiple public instances in addition to the local instance, avoiding rate limits longer but being much slower.

The `sentiment_analyzer` analyzes articles using structured LLM output and saves to a new JSON file. Parameters include temperature and whether to use reasoning. Using reasoning is slower but provides the model's thought process.

The `sentiment_grapher` produces a graph of sentiment from the latest file.

Each script loads the latest corresponding source file from `/news` and `/sentiment` based on generated timestamped filenames. New generated files (news, sentiment, and chart) have the same timestamp as the source file, unless there are issues extracting the timestamp.

Control these files using the `config.toml` file. Default values are loaded first, then config values overwrite them.

Check `config_example.toml` for parameters.

### Consistency Tester

The `consistency_tester` provides an interactive CLI to: 1. build a testing dataset, 2. run comprehensive testing with various sentiment analysis configurations, and 3. get statistical analysis of results. This is needed to fine-tune parameters for optimal results. Results are stored in a separate folder with subfolders for each specific run.

### Parallel processor

The `parallel_processor.py` performs the entire analysis process from start to end and saves results to the PostgreSQL database (no JSON files are saved). The pipeline: 1. gets articles from multiple sources, 2. filters out articles already present in the database, 3. adds vector embeddings for each article, 4. optionally adds similar articles from the database for each filtered article (used for similarity-based scoring), 5. performs sentiment analysis, and 6. saves results to the PostgreSQL database.

## Project Structure

The codebase follows a clean, modular architecture with comprehensive API design and database integration:

```
sentiment-analysis/
├── pyproject.toml                    # Project configuration and dependencies
├── ruff.toml                         # Linting and formatting rules
├── .mypy.ini                         # Type checking configuration
├── .pre-commit-config.yaml           # Pre-commit hooks configuration
├── README.md                         # This file
├── .env.example                      # Environment variables template
└── src/
    └── sentiment_analysis/
        ├── __init__.py               # Public API exports and package metadata (25+ functions)
        ├── alpaca_news.py            # Process news from Alpaca
        ├── client_manager.py         # AI client configuration (OpenRouter/LMStudio)
        ├── config_utils.py           # Configuration management
        ├── config.toml               # Configuration file (user-configured)
        ├── config.toml.example       # Configuration file template
        ├── consistency_tester.py     # Statistical consistency testing framework
        ├── db_utils.py               # PostgreSQL database operations and setup
        ├── embedding_model.py        # Manage the embedding model for vector generation
        ├── google_news.py            # Process news from Google RSS feed
        ├── google_supervisor.py      # Scheduler for automated execution
        ├── logging_utils.py          # Logging management
        ├── parallel_processor.py     # Process all news sources (async)
        ├── pipeline.py               # News processing pipeline
        ├── prompt_manager.py         # Specialized prompts for news sentiment analysis
        ├── searxng_search.py         # Search engine integration for article content
        ├── sentiment_analyzer.py     # LLM sentiment analysis engine with Pydantic models
        ├── sentiment_grapher.py      # Chart generation and data visualization
        ├── telegram_news.py          # Process news from Telegram
        ├── utils.py                  # Common utility functions and helpers
        ├── news/                     # Auto-generated news JSON files
        │   └── news_[sortable]_[timestamp].json
        ├── sentiments/               # Auto-generated sentiment analysis results
        │   └── sentiments_[sortable]_[timestamp].json
        ├── charts/                   # Auto-generated sentiment charts
        │   └── chart_[sortable]_[timestamp].png
        └── consistency/              # Auto-generated consistency reports
            └── run_[sortable]_[timestamp]/
                ├── analysis_[sortable]_[timestamp].txt
                ├── consistency_analysis.png
                ├── sample_distributions.png
                └── test_scores_[sortable]_[timestamp].json
```

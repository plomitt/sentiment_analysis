# Bitcoin News Sentiment Analysis Pipeline

A fully automated sentiment analysis pipeline for Bitcoin news articles using Instructor and large language models. This system fetches real-time news, performs trading-focused sentiment analysis, generates visualizations, and provides statistical consistency testing - all with automatic file management and timestamp tracking.

The pipeline provides sentiment scores on a 1-10 scale where 1 suggests "sell" and 10 suggests "buy", specifically designed for Bitcoin trading decisions.

## Features

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

## Project Structure

```
src/sentiment_analysis/
├── news_rss_fetcher.py          # RSS news fetcher for real-time Bitcoin news
├── sentiment_analyzer.py        # LLM sentiment analysis engine
├── sentiment_grapher.py         # Chart generation and visualization
├── consistency_tester.py        # Statistical consistency testing framework
├── client_manager.py            # LLM client management (OpenRouter/LMStudio)
├── prompt_manager.py            # Specialized prompts for Bitcoin sentiment analysis
├── news/                        # Auto-generated news JSON files
│   └── news_[sortable]_[timestamp].json
├── sentiments/                  # Auto-generated sentiment analysis results
│   └── sentiments_[sortable]_[timestamp].json
├── charts/                      # Auto-generated sentiment charts
│   └── chart_[sortable]_[timestamp].png
└── consistency/                 # Auto-generated consistency reports
    └── consistency_[sortable]_[timestamp]/
        ├── consistency_[sortable]_[timestamp].json
        ├── consistency_summary_[sortable]_[timestamp].csv
        ├── consistency_report_[sortable]_[timestamp].html
        └── *.png charts
```

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

Install dependencies using Poetry:

```bash
poetry install
```

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

Set up your environment variables in `.env`:

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
```

### Getting API Keys

1. **OpenRouter**: Sign up at [openrouter.ai](https://openrouter.ai) and get your API key from the dashboard
2. **LMStudio**: Download [LM Studio](https://lmstudio.ai), run it locally, and use the default settings

The system defaults to OpenRouter for production use. Switch to LMStudio for local, private processing.

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

## Performance

- **Processing Speed**: ~2-3 seconds per article (depends on model)
- **Batch Processing**: Efficient handling of multiple articles
- **Memory Usage**: Low memory footprint with streaming processing
- **Scalability**: Can handle hundreds of articles in a single batch
- **Statistical Testing**: Configurable iteration counts for reliability testing

## Dependencies

- `instructor`: Structured LLM outputs with Pydantic validation
- `python-dotenv`: Environment variable management for API configuration
- `matplotlib`: High-quality chart generation and data visualization
- `pandas`: Data manipulation and time-series analysis
- `numpy`: Numerical computing and statistical calculations
- `scipy`: Advanced statistical functions and scientific computing
- `statsmodels`: Statistical modeling and analysis
- `feedparser`: RSS feed parsing for news collection
- `aiohttp`: Asynchronous HTTP client for SearXNG search integration

Note: `openai` is automatically included as a dependency of `instructor`.

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
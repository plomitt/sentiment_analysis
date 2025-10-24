# Bitcoin News Sentiment Analysis

A sophisticated sentiment analysis system for Bitcoin news articles using Instructor and large language models. This system analyzes news from a trading perspective, providing sentiment scores on a 1-10 scale where 1 suggests "sell" and 10 suggests "buy".

## Features

- **Trading-Focused Analysis**: Sentiment scores are specifically designed for Bitcoin trading decisions
- **Structured Output**: Uses Pydantic models for consistent, validated results
- **Intelligent Prompting**: Few-shot learning with chain-of-thought reasoning
- **Batch Processing**: Efficiently analyze multiple articles
- **Detailed Reasoning**: Each score includes concise trading-focused explanations
- **Functional Design**: Simple, stateless functions without unnecessary class overhead
- **Twitter/X Integration**: Browser state storage for authenticated Twitter data access

## Project Structure

```
src/sentiment_analysis/
├── client_manager.py          # LLM client management (OpenRouter/LMStudio)
├── prompt_manager.py          # Specialized prompts for Bitcoin sentiment analysis
├── sentiment_analyzer.py      # Main analysis engine (functional approach)
├── demo.py                   # Demonstration script
├── news.json                 # Input news articles
└── news_with_sentiment.json  # Output with sentiment analysis

twitter_auth/
├── twitter_auth.py            # Twitter/X authentication with browser state storage
├── demo_manual_setup.py      # Manual authentication setup script
├── README.md                 # Twitter authentication documentation
└── .auth/                    # Authentication state storage (excluded from git)
```

## Installation

Install dependencies using Poetry:

```bash
poetry install
```

## Configuration

Set up your environment variables in `.env`:

```env
# OpenRouter Configuration
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_MODEL_ID=anthropic/claude-3-haiku

# LMStudio Configuration (optional)
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_API_KEY=123
LMSTUDIO_MODEL_ID=your_model_here

# Twitter/X Authentication
TWITTER_EMAIL=your_twitter_email
TWITTER_PASSWORD=your_twitter_password

USE_LMSTUDIO=false
```

## Quick Start

### 1. Analyze Existing News

```bash
cd src/sentiment_analysis
poetry run python sentiment_analyzer.py
```

This will:
- Load articles from `news.json`
- Analyze each article's sentiment
- Save results to `news_with_sentiment.json`
- Display a summary of the analysis

### 2. Run Demo

```bash
poetry run python demo.py
```

The demo showcases:
- Single article analysis
- Score interpretation guide
- Batch analysis results

### 3. Use in Your Code

```python
from sentiment_analyzer import analyze_article

# Analyze a single article
sentiment = analyze_article(
    title="Bitcoin Reaches New All-Time High",
    body="Bitcoin surged to new heights as institutional adoption accelerates..."
)

print(f"Score: {sentiment.score}/10")
print(f"Reasoning: {sentiment.reasoning}")

# Or analyze a complete file
from sentiment_analyzer import analyze_news_file
success = analyze_news_file('input.json', 'output.json')
```

## Functional API

The system provides a simple functional API:

### Core Functions

- `analyze_article(title, body, client=None)`: Analyze single article
- `analyze_articles_batch(articles, client=None)`: Analyze multiple articles
- `analyze_news_file(input_file, output_file, client=None)`: Complete pipeline
- `load_articles_from_json(file_path)`: Load articles from JSON
- `save_results_to_json(articles, output_path)`: Save results to JSON
- `create_client()`: Create Instructor client

### Usage Examples

```python
# Single article analysis
from sentiment_analyzer import analyze_article

sentiment = analyze_article(
    "Bitcoin Price Surges",
    "Bitcoin reached new heights as institutional adoption accelerates..."
)

# Batch analysis
articles = [
    {"title": "Article 1", "body": "Content 1", "timestamp": "2024-01-01", "url": "..."},
    {"title": "Article 2", "body": "Content 2", "timestamp": "2024-01-02", "url": "..."}
]

results = analyze_articles_batch(articles)

# Complete pipeline
success = analyze_news_file('news.json', 'results.json')
```

## Twitter/X Integration

The system includes browser state storage for authenticated Twitter/X access, enabling extraction of live Bitcoin-related tweets and sentiment data.

### Quick Start with Twitter Authentication

```bash
# Manual setup (recommended for first-time use)
poetry run python twitter_auth/demo_manual_setup.py

# Automated usage (after initial setup)
from twitter_auth import get_authenticated_search_results
content = get_authenticated_search_results()
```

### Twitter Authentication Features

- **Browser State Persistence**: Save and reuse authenticated browser sessions
- **Anti-Detection Measures**: Stealth browser configuration and human-like interaction
- **Automatic Validation**: Verify authentication status before accessing protected content
- **Secure Storage**: Authentication state stored securely and excluded from version control

### Detailed Setup

For complete Twitter/X authentication setup and usage instructions, see:

```
twitter_auth/README.md  # Comprehensive documentation and usage guide
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

Each article in the output JSON contains:

```json
{
  "title": "Article title",
  "body": "Article content",
  "timestamp": "Publication timestamp",
  "url": "Article URL",
  "sentiment": {
    "score": 7.5,
    "reasoning": "Concise trading-focused explanation for the score"
  }
}
```

## Prompt Engineering

The system uses advanced prompting techniques:

1. **Few-Shot Learning**: 5 examples of Bitcoin news with scores and reasoning
2. **Chain-of-Thought**: Models think through market impact before scoring
3. **Trading Context**: Emphasis on market implications rather than general sentiment
4. **Structured Framework**: Clear definitions for each score range
5. **Domain-Specific Examples**: Real Bitcoin news scenarios

## Analysis Factors

The model considers:

- **Market Impact**: Effect on Bitcoin price and trading volume
- **Regulatory Environment**: Positive or negative regulatory implications
- **Institutional Adoption**: Signs of increased/decreased institutional interest
- **Technical Indicators**: Price levels, support/resistance, market structure
- **Market Sentiment**: Fear/greed indicators, social media sentiment
- **Adoption Metrics**: Real-world usage, merchant acceptance, network effects

## Example Results

### Bullish Example (8.5/10)
**Title**: "WisdomTree Gains FCA Approval to Offer Bitcoin and Ethereum ETPs to UK Retail Investors"

**Reasoning**: "The FCA approval for WisdomTree's Bitcoin and Ethereum ETPs opens the door for significant institutional capital inflows from the UK market. This regulatory approval eliminates a key barrier to entry and provides legitimacy to cryptocurrency investments in a major financial market."

### Bearish Example (2.5/10)
**Title**: "Crypto market cap erases $110 billion in huge 1-day crash"

**Reasoning**: "The cryptocurrency market experienced a dramatic crash, losing approximately $110 billion in less than 24 hours. This massive sell-off is likely to trigger panic selling across various assets, including Bitcoin."

## Advanced Usage

### Custom Client Configuration

```python
from sentiment_analyzer import create_client, analyze_article

# Use custom configuration
client = create_client()  # Uses default configuration from client_manager
sentiment = analyze_article(title, body, client=client)
```

### Batch Processing with Custom Input

```python
articles = [
    {
        "title": "Your article title",
        "body": "Your article content",
        "timestamp": "2024-01-01",
        "url": "https://example.com"
    }
]

results = analyze_articles_batch(articles)
```

## Error Handling

The system includes comprehensive error handling:

- **API Failures**: Falls back to neutral sentiment (5.0) with error explanation
- **Invalid Data**: Skips malformed articles with logging
- **Network Issues**: Retry logic for transient failures
- **Validation**: Ensures scores stay within 1.0-10.0 range

## Performance

- **Processing Speed**: ~2-3 seconds per article (depends on model)
- **Batch Processing**: Efficient handling of multiple articles
- **Memory Usage**: Low memory footprint with streaming processing
- **Scalability**: Can handle hundreds of articles in a single batch

## Dependencies

- `instructor`: Structured LLM outputs
- `pydantic`: Data validation and settings management
- `openai`: OpenAI API client
- `python-dotenv`: Environment variable management
- `playwright`: Web automation (for news collection)

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
# AI News Extractor

A powerful AI agent that extracts structured news information (title, body, timestamp) from HTML content using Instructor and large language models.

## 🚀 Features

- **Single API Call**: Clean, simple implementation with just one LLM call
- **Robust Extraction**: Leverages AI to intelligently parse HTML and identify news content
- **Structured Output**: Returns validated Pydantic models with title, body, and timestamp
- **Flexible Input**: Works with any HTML content or URLs
- **Error Handling**: Comprehensive validation and graceful error handling
- **Integration Ready**: Seamlessly works with existing project tools

## 📁 Files

- `news_extractor.py` - Main AI agent for news extraction
- `extract_html.py` - HTML content extractor from URLs
- `test_integration.py` - Complete pipeline demonstration
- `client_manager.py` - LLM client management (existing)

## 🔧 Usage

### Basic News Extraction from HTML Content

```python
from news_extractor import extract_news_from_html

# Extract news from HTML content
html_content = "<html>...</html>"
news = extract_news_from_html(html_content)

print(f"Title: {news.title}")
print(f"Body: {news.body}")
print(f"Timestamp: {news.timestamp}")
```

### Extract News from HTML File

```python
from news_extractor import extract_news_from_file

# Extract news from saved HTML file
news = extract_news_from_file("article.html")

print(f"📰 {news.title}")
print(f"📄 {news.body[:200]}...")
```

### Complete Pipeline: URL to Structured News

```python
from extract_html import extract_page_html
from news_extractor import extract_news_from_html

# Step 1: Get HTML from URL
html_content = extract_page_html("https://news.example.com/article")

# Step 2: Extract news information
news = extract_news_from_html(html_content)

print(f"📰 Title: {news.title}")
print(f"⏰ Published: {news.timestamp}")
print(f"📄 Content: {len(news.body)} characters")
```

## 📋 Command Line Usage

### Run Demo
```bash
poetry run python news_extractor.py
```

### Extract from HTML File
```bash
poetry run python news_extractor.py path/to/article.html
```

### Test Complete Pipeline
```bash
poetry run python test_integration.py  # Tests with live URLs
poetry run python test_integration.py path/to/html_file.txt  # Tests with existing file
```

## 📊 Output Structure

The extractor returns a `NewsArticle` object with the following fields:

```python
class NewsArticle(BaseModel):
    title: str          # Main headline/title of the article
    body: str           # Main content body (cleaned text)
    timestamp: str      # Publication date/time (optional)
```

### Example Output
```python
NewsArticle(
    title="Bitcoin Faces Heavy Selling Pressure Despite Seasonal Bullish Expectations",
    body="Despite the prevailing seasonal bullish expectations among investors and market analysts, Bitcoin is currently experiencing significant selling pressure...",
    timestamp="2024-01-15T10:30:00Z"
)
```

## 🎯 Test Results

The extractor has been tested with various content types:

- ✅ **News Websites**: CryptoPanic, financial news
- ✅ **Documentation Sites**: Example.com, technical content
- ✅ **Literature Content**: Moby-Dick excerpts
- ✅ **Error Handling**: Invalid HTML, empty content

## 🤖 AI Capabilities

The AI agent can:

- **Identify Main Content**: Distinguish article content from navigation, ads, sidebars
- **Clean Text**: Remove HTML artifacts and normalize whitespace
- **Extract Metadata**: Find publication dates, authors, timestamps
- **Handle Multiple Articles**: Focus on the primary/most prominent article
- **Flexible Parsing**: Work with various HTML structures and formats

## 🔧 Configuration

The extractor uses your existing `client_manager.py` configuration:

- **OpenRouter**: Default, works with various models
- **LMStudio**: Local models support
- **Environment Variables**: Configure via `.env` file

### Using LMStudio
```python
news = extract_news_from_html(html_content, use_lmstudio=True)
```

## 📝 Best Practices

1. **Quality HTML**: Provide complete HTML content for best results
2. **Token Limits**: Large HTML (>10k chars) is automatically truncated
3. **Validation**: Built-in field validation ensures quality output
4. **Error Handling**: Always wrap in try-catch for production use

## 🔗 Integration Examples

### With HTML Extractor
```python
from extract_html import extract_page_html
from news_extractor import extract_news_from_html

def get_news_from_url(url):
    html = extract_page_html(url)
    return extract_news_from_html(html)
```

### Batch Processing
```python
import os
from news_extractor import extract_news_from_file

def process_html_files(directory):
    results = []
    for filename in os.listdir(directory):
        if filename.endswith('.html') or filename.endswith('.txt'):
            try:
                news = extract_news_from_file(os.path.join(directory, filename))
                results.append(news)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    return results
```

## 🚨 Error Handling

The extractor includes comprehensive error handling:

- **Input Validation**: Empty HTML, too short content
- **AI Extraction**: API failures, validation errors
- **File Operations**: Missing files, read errors

Example:
```python
try:
    news = extract_news_from_html(html_content)
    print(f"✅ Extracted: {news.title}")
except ValueError as e:
    print(f"❌ Invalid input: {e}")
except Exception as e:
    print(f"❌ Extraction failed: {e}")
```

## 🎉 Summary

This AI news extractor provides a clean, reliable way to convert raw HTML content into structured news data with just a single API call. It's designed to be simple to use while being powerful enough to handle real-world web content extraction needs.
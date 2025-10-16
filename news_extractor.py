#!/usr/bin/env python3
"""
AI Agent for extracting news information from HTML content using Instructor.

This module provides a simple AI agent that takes HTML content and extracts
structured news information (title, body, timestamp) using a single API call.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator
from client_manager import build_client


class NewsArticle(BaseModel):
    """
    Structured model for news article information extracted from HTML content.
    """
    title: str = Field(
        ...,
        description="The main headline or title of the news article. Extract the most prominent title from the page."
    )
    body: str = Field(
        ...,
        description="The main content body of the news article. Extract the primary article text, excluding navigation, ads, and other non-article content."
    )
    timestamp: Optional[str] = Field(
        None,
        description="The publication date and time of the news article. Extract any timestamp, date, or time information related to when this was published."
    )

    @field_validator('title')
    @classmethod
    def validate_title(cls, v):
        if not v or not v.strip():
            raise ValueError('Title cannot be empty')
        if len(v.strip()) < 3:
            raise ValueError('Title must be at least 3 characters long')
        return v.strip()

    @field_validator('body')
    @classmethod
    def validate_body(cls, v):
        if not v or not v.strip():
            raise ValueError('Body cannot be empty')
        if len(v.strip()) < 10:
            raise ValueError('Body must be at least 10 characters long')
        return v.strip()

    def __str__(self):
        """String representation of the news article."""
        timestamp_str = f" | {self.timestamp}" if self.timestamp else ""
        return f"📰 {self.title}{timestamp_str}\n📄 {len(self.body)} characters"


def extract_news_from_html(html_content: str, use_lmstudio: bool = False):
    """
    Extract structured news information from HTML content using an AI agent.

    This function uses a single API call to analyze HTML content and extract
    the news article's title, body, and timestamp using Instructor.

    Args:
        html_content (str): Raw HTML content to extract news information from
        use_lmstudio (bool): Whether to use LMStudio client (default: False)

    Returns:
        NewsArticle: Structured news article information

    Raises:
        ValueError: If HTML content is empty or invalid
        Exception: If AI extraction fails
    """
    # Input validation
    if not html_content or not html_content.strip():
        raise ValueError("HTML content cannot be empty")

    if len(html_content.strip()) < 50:
        raise ValueError("HTML content appears to be too short to contain a news article")

    print("🤖 Initializing AI agent for news extraction...")

    # Build the client
    client = build_client()

    # Create the system message
    system_message = """
    You are an expert news content extractor. Analyze the provided HTML content and extract the main news article information.

    Focus on identifying:
    1. The primary article title/headline (usually in h1, h2, or title tags)
    2. The main article body content (exclude navigation, ads, sidebars, comments)
    3. Publication timestamp/date information

    Guidelines:
    - Look for the main article content, not other page elements
    - Clean up extracted text by removing excessive whitespace and HTML artifacts
    - If multiple articles are present, focus on the primary/most prominent one
    - For timestamp, extract any date/time information related to publication
    - Preserve the meaning and key information while cleaning up formatting
    """

    # Create user message with HTML content
    user_message = f"""
    Please extract the news article information from the following HTML content:

    HTML Content:
    {html_content[:10000]}  # Limit to first 10k chars to avoid token limits
    {html_content[10000:] if len(html_content) > 10000 else ""}
    """

    print("🕷️  Analyzing HTML content with AI...")

    try:
        # Single API call to extract structured news data
        news_article = client.chat.completions.create(
            model="auto",  # Use the model from client configuration
            response_model=NewsArticle,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            extra_body={"provider": {"require_parameters": True}}
        )

        print("✅ News extraction completed successfully!")
        print(f"📰 Title: {news_article.title}")
        print(f"📄 Body Length: {len(news_article.body)} characters")
        if news_article.timestamp:
            print(f"⏰ Timestamp: {news_article.timestamp}")

        return news_article

    except Exception as e:
        print(f"❌ Error during news extraction: {e}")
        raise Exception(f"Failed to extract news information: {e}")


def extract_news_from_file(html_file_path: str, use_lmstudio: bool = False):
    """
    Convenience function to extract news from an HTML file.

    Args:
        html_file_path (str): Path to the HTML file
        use_lmstudio (bool): Whether to use LMStudio client

    Returns:
        NewsArticle: Structured news article information
    """
    try:
        with open(html_file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()

        print(f"📂 Loaded HTML from: {html_file_path}")
        print(f"📏 HTML content length: {len(html_content)} characters")

        return extract_news_from_html(html_content, use_lmstudio=use_lmstudio)

    except FileNotFoundError:
        raise FileNotFoundError(f"HTML file not found: {html_file_path}")
    except Exception as e:
        raise Exception(f"Error reading HTML file: {e}")


def demo_extraction():
    """
    Demonstration function showing how to use the news extractor.
    """
    print("🎯 News Extractor Demo")
    print("=" * 50)

    # Sample HTML content for demonstration
    sample_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Breaking News: AI Technology Advances</title>
    </head>
    <body>
        <header>
            <nav>Navigation menu</nav>
        </header>
        <main>
            <article>
                <h1>Major Breakthrough in AI Technology Announced</h1>
                <div class="meta">
                    <time datetime="2024-01-15T10:30:00Z">January 15, 2024 at 10:30 AM</time>
                    <span class="author">By Tech Reporter</span>
                </div>
                <div class="article-body">
                    <p>Scientists and engineers have announced a significant breakthrough in artificial intelligence technology that promises to revolutionize how we interact with machines.</p>
                    <p>The new system, developed after years of research, demonstrates capabilities that were previously thought to be decades away. Researchers report that the AI can understand complex instructions and perform sophisticated reasoning tasks with remarkable accuracy.</p>
                    <p>This development has far-reaching implications for industries ranging from healthcare to transportation, and experts believe it could accelerate progress in numerous fields.</p>
                </div>
            </article>
        </main>
        <aside>
            <div class="advertisement">Ad content</div>
        </aside>
    </body>
    </html>
    """

    try:
        # Extract news from sample HTML
        news = extract_news_from_html(sample_html)

        print("\n📊 Extraction Results:")
        print("=" * 50)
        print(news)
        print("\n📝 Full Title:")
        print(f"   {news.title}")
        print("\n📄 Full Body:")
        print(f"   {news.body}")
        print(f"\n⏰ Timestamp:")
        print(f"   {news.timestamp or 'Not found'}")

    except Exception as e:
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Command line usage: python news_extractor.py <html_file_path>
        html_file = sys.argv[1]
        try:
            news = extract_news_from_file(html_file)
            print(f"\n✅ Extraction completed!")
            print(f"📰 Title: {news.title}")
            print(f"📄 Body: {news.body[:200]}...")
            print(f"⏰ Timestamp: {news.timestamp}")
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    else:
        # Run demo
        demo_extraction()
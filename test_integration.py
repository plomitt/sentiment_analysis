#!/usr/bin/env python3
"""
Integration test for HTML extraction + AI news extraction pipeline.

This script demonstrates the complete workflow:
1. Extract HTML from a URL using extract_html.py
2. Extract news information from HTML using news_extractor.py
"""

from extract_html import extract_page_html, save_html_to_file, generate_filename
from news_extractor import extract_news_from_html
import os
import sys


def test_complete_pipeline(url):
    """
    Test the complete pipeline from URL to structured news data.

    Args:
        url (str): URL to extract and analyze
    """
    print("🚀 Testing Complete HTML + AI News Extraction Pipeline")
    print("=" * 60)
    print(f"🌐 Target URL: {url}")
    print()

    # Step 1: Extract HTML content
    print("📄 Step 1: Extracting HTML content...")
    try:
        html_content = extract_page_html(url, headless=True)

        if html_content:
            print(f"✅ HTML extraction successful! ({len(html_content):,} characters)")

            # Save HTML to file for reference
            filename = generate_filename(url)
            saved_path = save_html_to_file(html_content, filename)
            print(f"💾 HTML saved to: {saved_path}")
        else:
            print("❌ HTML extraction failed")
            return

    except Exception as e:
        print(f"❌ HTML extraction error: {e}")
        return

    print()

    # Step 2: Extract news information using AI
    print("🤖 Step 2: Extracting news information with AI...")
    try:
        news_article = extract_news_from_html(html_content)

        print("✅ AI extraction completed successfully!")
        print()

        # Display results
        print("📊 Final Results:")
        print("=" * 60)
        print(f"📰 Title: {news_article.title}")
        print(f"⏰ Timestamp: {news_article.timestamp or 'Not found'}")
        print(f"📄 Body Length: {len(news_article.body)} characters")
        print()
        print("📝 Body Preview:")
        print("-" * 40)
        if len(news_article.body) > 300:
            print(news_article.body[:300] + "...")
        else:
            print(news_article.body)
        print("-" * 40)

        return news_article

    except Exception as e:
        print(f"❌ AI extraction error: {e}")
        return None


def test_with_existing_file(html_file_path):
    """
    Test news extraction on an existing HTML file.

    Args:
        html_file_path (str): Path to existing HTML file
    """
    print(f"🎯 Testing AI News Extraction on Existing File")
    print("=" * 60)
    print(f"📂 File: {html_file_path}")

    if not os.path.exists(html_file_path):
        print(f"❌ File not found: {html_file_path}")
        return

    # Get file size
    file_size = os.path.getsize(html_file_path)
    print(f"📏 File size: {file_size:,} bytes")
    print()

    try:
        # Read HTML content
        with open(html_file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()

        # Extract news information
        news_article = extract_news_from_html(html_content)

        print("✅ Extraction completed!")
        print()
        print("📊 Results:")
        print("=" * 60)
        print(f"📰 Title: {news_article.title}")
        print(f"⏰ Timestamp: {news_article.timestamp or 'Not found'}")
        print(f"📄 Body Length: {len(news_article.body)} characters")
        print()
        print("📝 Full Body:")
        print("-" * 40)
        print(news_article.body)
        print("-" * 40)

        return news_article

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def main():
    """
    Main function to run integration tests.
    """
    if len(sys.argv) > 1:
        # Test with existing HTML file
        html_file = sys.argv[1]
        test_with_existing_file(html_file)
    else:
        # Test with a live URL
        test_urls = [
            "https://example.com",
            "https://httpbin.org/html"
        ]

        for url in test_urls:
            print(f"\n🔄 Testing URL: {url}")
            try:
                test_complete_pipeline(url)
                print("\n" + "=" * 80 + "\n")
            except Exception as e:
                print(f"❌ Test failed for {url}: {e}")
                print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
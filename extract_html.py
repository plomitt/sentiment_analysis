#!/usr/bin/env python3
"""
Extract and save the complete HTML content of a webpage as a .txt file.

This script uses Playwright to navigate to a URL, wait for the page to fully load,
extract the entire HTML source using page.content(), and save it to a text file.
"""

from playwright.sync_api import sync_playwright
from dotenv import load_dotenv
import os
import sys
from urllib.parse import urlparse
from datetime import datetime


def generate_filename(url):
    """
    Generate a descriptive filename for the HTML content file.

    Args:
        url (str): The URL of the webpage

    Returns:
        str: A filename based on the domain and timestamp
    """
    try:
        # Parse the URL to get the domain
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.replace('www.', '')

        # Clean the domain to make it filename-safe
        domain = domain.replace(':', '_').replace('/', '_')

        # Add timestamp to make filename unique
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        return f"{domain}_html_{timestamp}.txt"

    except Exception:
        # Fallback to timestamp-only filename if URL parsing fails
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"html_extract_{timestamp}.txt"


def extract_page_html(url, headless=True):
    """
    Extract the complete HTML content from a webpage.

    Args:
        url (str): The URL of the webpage to extract HTML from
        headless (bool): Whether to run browser in headless mode

    Returns:
        str: The complete HTML content of the page, or None if extraction failed
    """
    print(f"🚀 Starting HTML extraction from: {url}")

    with sync_playwright() as p:
        # Launch browser
        browser = p.chromium.launch(headless=headless)

        try:
            # Create a new page
            page = browser.new_page()

            # Navigate to the URL with extended timeout
            print("📄 Loading page...")
            page.goto(url, timeout=60000, wait_until='domcontentloaded')

            # Wait for page to fully load
            print("⏳ Waiting for page to complete loading...")
            try:
                page.wait_for_load_state('load', timeout=15000)
                print("✅ Page fully loaded")
            except Exception as e:
                print(f"⚠️  Page load timeout, proceeding anyway: {e}")

            # Additional wait for dynamic content
            page.wait_for_timeout(2000)

            # Extract the complete HTML content
            print("🕷️  Extracting HTML content...")
            html_content = page.content()

            if html_content:
                content_length = len(html_content)
                print(f"✅ HTML extraction successful! ({content_length:,} characters)")
                return html_content
            else:
                print("❌ HTML extraction failed - no content returned")
                return None

        except Exception as e:
            print(f"❌ Error during HTML extraction: {e}")
            return None

        finally:
            # Always close the browser
            browser.close()
            print("🔚 Browser closed")


def save_html_to_file(html_content, filename):
    """
    Save HTML content to a text file.

    Args:
        html_content (str): The HTML content to save
        filename (str): The filename to save to

    Returns:
        str: The path to the saved file, or None if saving failed
    """
    try:
        # Save to current directory
        file_path = os.path.join(os.getcwd(), filename)

        print(f"💾 Saving HTML content to: {filename}")

        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(html_content)

        # Verify file was created and get its size
        file_size = os.path.getsize(file_path)

        print(f"✅ File saved successfully!")
        print(f"📁 Path: {file_path}")
        print(f"📊 Size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")

        return file_path

    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return None


def main():
    """
    Main function to handle URL input and orchestrate HTML extraction.
    """
    # Load environment variables
    load_dotenv()

    # Get URL from command line argument or environment variable
    url = None

    # Try command line argument first
    if len(sys.argv) > 1:
        url = sys.argv[1]
        print(f"📥 Using URL from command line: {url}")
    else:
        # Try environment variable
        url = os.getenv("TEST_URL")
        if url:
            print(f"📥 Using URL from environment variable: {url}")

    # If no URL found, prompt user or show usage
    if not url:
        print("❌ No URL provided!")
        print("\nUsage:")
        print("  poetry run python extract_html.py <URL>")
        print("  # Or set TEST_URL in your .env file")
        print("\nExample:")
        print("  poetry run python extract_html.py https://example.com")
        sys.exit(1)

    # Validate URL format
    if not (url.startswith('http://') or url.startswith('https://')):
        print("❌ Invalid URL format! URL must start with http:// or https://")
        sys.exit(1)

    print("=" * 60)
    print("🕷️  PLAYWRIGHT HTML EXTRACTOR")
    print("=" * 60)

    # Extract HTML content
    html_content = extract_page_html(url, headless=True)

    if html_content:
        # Generate filename and save
        filename = generate_filename(url)
        saved_path = save_html_to_file(html_content, filename)

        if saved_path:
            print("\n" + "=" * 60)
            print("🎉 EXTRACTION COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"🌐 Source URL: {url}")
            print(f"📄 Output File: {saved_path}")
            print(f"📏 Content Size: {len(html_content):,} characters")
            print("\nYou can now view the complete HTML source in the text file.")
        else:
            print("\n❌ Extraction failed during file saving")
            sys.exit(1)
    else:
        print("\n❌ HTML extraction failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
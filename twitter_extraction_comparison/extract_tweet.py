"""
Twitter/X Tweet Data Extraction

A simple, reusable function to extract tweet text, author, and datetime from Twitter/X URLs using Playwright.
Based on Method 1 (inner_text) from the comparison script - proven to be reliable and fast.

Usage:
    from extract_tweet import extract_tweet_data, extract_tweet_text

    # Enhanced version - returns structured data
    tweet_data = extract_tweet_data("https://x.com/user/status/123456789")
    if tweet_data:
        print(f"Author: {tweet_data['author']}")
        print(f"Time: {tweet_data['datetime']}")
        print(f"Text: {tweet_data['text']}")

    # Simple version - returns just text (backward compatible)
    tweet_text = extract_tweet_text("https://x.com/user/status/123456789")
"""

import time
import traceback
import re
from typing import Optional, Dict
from playwright.sync_api import sync_playwright


def parse_engagement_count(text: str) -> int:
    """
    Parse Twitter's engagement count strings into integers.

    Handles formats like:
    - "1,234" → 1234
    - "1.2K" → 1200
    - "3.5M" → 3500000
    - "Hundreds" → 100 (approximation)

    Args:
        text: Formatted count string from Twitter

    Returns:
        Integer count, or 0 if parsing fails
    """
    if not text:
        return 0

    text = text.strip().upper()

    # Handle special cases
    if "HUNDRED" in text:
        return 100
    elif "THOUSAND" in text:
        return 1000
    elif "MILLION" in text:
        return 1000000

    # Extract numbers using regex
    match = re.search(r'([0-9,.]+)\s*([KM]?)', text)
    if not match:
        return 0

    number_str = match.group(1).replace(',', '')
    suffix = match.group(2)

    try:
        base_number = float(number_str)

        if suffix == 'K':  # Thousands
            return int(base_number * 1000)
        elif suffix == 'M':  # Millions
            return int(base_number * 1000000)
        else:
            return int(base_number)

    except (ValueError, TypeError):
        return 0


def extract_tweet_data(url: str, timeout: int = 30) -> Optional[Dict[str, any]]:
    """
    Extract comprehensive tweet data from Twitter/X URL using Playwright.

    This function extends the proven Method 1 (inner_text) approach to extract
    tweet content, metadata, and engagement metrics.

    Args:
        url: Twitter/X post URL (e.g., "https://x.com/user/status/123456789")
        timeout: Maximum wait time in seconds (default: 30)

    Returns:
        Dictionary with complete tweet data:
        {
            'text': str,           # Tweet text content
            'author': str,         # @username handle
            'datetime': str,       # Date/time text
            'url': str,           # Original URL
            'likes': int,         # Number of likes
            'comments': int,      # Number of comments
            'bookmarks': int,     # Number of bookmarks
            'retweets': int       # Number of retweets
        }
        or None if extraction failed

    Example:
        >>> tweet_data = extract_tweet_data("https://x.com/MerlijnTrader/status/1979585766515761455")
        >>> print(f"Author: {tweet_data['author']}")
        @MerlijnTrader
        >>> print(f"Time: {tweet_data['datetime']}")
        6:30 PM · Oct 18, 2025
        >>> print(f"Likes: {tweet_data['likes']}")
        1234
        >>> print(f"Text: {tweet_data['text']}")
        BREAKING: CHINA JUST WENT ALL-IN ON BITCOIN...
    """

    # Validate URL
    if not url or not isinstance(url, str):
        print("Error: Invalid URL provided")
        return None

    if not ('x.com/' in url or 'twitter.com/' in url):
        print("Error: URL must be from x.com or twitter.com")
        return None

    playwright = None
    browser = None
    context = None

    try:
        # Setup Playwright browser
        playwright = sync_playwright().start()
        browser = playwright.chromium.launch(
            headless=True,
            args=[
                '--no-sandbox',
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--disable-web-security',
                '--disable-features=VizDisplayCompositor',
                '--no-first-run',
                '--no-default-browser-check',
                '--disable-default-apps',
                '--disable-popup-blocking'
            ]
        )

        context = browser.new_context(
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            viewport={'width': 1280, 'height': 800},
            ignore_https_errors=True
        )

        page = context.new_page()

        # Add stealth measures
        page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
        """)

        # Load the page
        print(f"Loading tweet: {url}")

        try:
            page.goto(url, wait_until='domcontentloaded', timeout=timeout * 1000)
        except:
            # Fallback with less strict waiting
            page.goto(url, timeout=timeout * 1000)

        # Wait for dynamic content
        page.wait_for_timeout(3000)

        # Find the main tweet container
        tweet_container = None
        tweet_container_selectors = [
            '[data-testid="tweet"]',
            'article[role="article"]',
            '[data-testid="tweetDetail"]'
        ]

        for selector in tweet_container_selectors:
            try:
                elements = page.locator(selector)
                if elements.count() > 0:
                    tweet_container = elements.first
                    print(f"Found tweet container with selector: {selector}")
                    break
            except:
                continue

        if not tweet_container:
            print("No tweet container found")
            return None

        # Initialize result
        result = {
            'text': None,
            'author': None,
            'datetime': None,
            'url': url,
            'likes': 0,
            'comments': 0,
            'bookmarks': 0,
            'retweets': 0
        }

        # Extract tweet text
        print("Extracting tweet text...")
        text_selectors = [
            '[data-testid="tweetText"]',
            'div[lang]',
            '.css-1dbjc4n div[lang]',
            'article div[lang]'
        ]

        for selector in text_selectors:
            try:
                text_element = tweet_container.locator(selector).first
                if text_element.count() > 0:
                    result['text'] = text_element.inner_text().strip()
                    print(f"Found tweet text with selector: {selector}")
                    break
            except:
                continue

        # Extract author handle (@username)
        print("Extracting author handle...")
        author_selectors = [
            '[data-testid="User-Name"] a[href*="/"] span',
            '[data-testid="UserScreenName"]',
            'a[href*="/"] span',
            'div[data-testid="User-Name"] span'
        ]

        for selector in author_selectors:
            try:
                author_elements = tweet_container.locator(selector).all()
                for element in author_elements:
                    author_text = element.inner_text().strip()
                    # Look for @username format
                    if author_text.startswith('@') and len(author_text) > 1:
                        result['author'] = author_text
                        print(f"Found author with selector: {selector}")
                        break
                if result['author']:
                    break
            except:
                continue

        # Extract datetime
        print("Extracting datetime...")
        datetime_selectors = [
            'time',
            '[data-testid="User-Name"] time',
            'span[datetime]',
            'a[href*="/status/"] time'
        ]

        for selector in datetime_selectors:
            try:
                datetime_elements = tweet_container.locator(selector).all()
                for element in datetime_elements:
                    # Try different ways to get datetime
                    datetime_text = element.inner_text().strip()
                    datetime_attr = element.get_attribute('datetime')

                    # Prefer text content (usually shows human-readable format)
                    if datetime_text and ('AM' in datetime_text or 'PM' in datetime_text or
                                        any(month in datetime_text.upper() for month in
                                           ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                                            'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'])):
                        result['datetime'] = datetime_text
                        print(f"Found datetime with selector: {selector}")
                        break
                    # Fallback to datetime attribute
                    elif datetime_attr:
                        result['datetime'] = datetime_attr
                        print(f"Found datetime attribute with selector: {selector}")
                        break
                if result['datetime']:
                    break
            except:
                continue

        # Extract engagement metrics using inner_text() method
        print("Extracting engagement metrics...")

        # Helper function to extract engagement metrics using inner_text with improved validation
        def extract_engagement_metric(metric_name: str, selectors: list, required_keywords: list) -> int:
            print(f"Extracting {metric_name}...")
            for selector in selectors:
                try:
                    elements = tweet_container.locator(selector).all()
                    print(f"  Trying selector: {selector} (found {len(elements)} elements)")

                    for i, element in enumerate(elements):
                        # Try aria-label first (usually contains count with description)
                        aria_label = element.get_attribute('aria-label')
                        if aria_label:
                            print(f"    Element {i+1} aria-label: '{aria_label}'")
                            # Validate that aria-label contains required keywords
                            aria_label_lower = aria_label.lower()
                            if any(keyword.lower() in aria_label_lower for keyword in required_keywords):
                                count = parse_engagement_count(aria_label)
                                if count > 0:
                                    print(f"✅ Found {metric_name} with selector: {selector} ({count})")
                                    return count
                            else:
                                print(f"    ❌ aria-label doesn't contain required keywords: {required_keywords}")

                        # Fallback to inner_text() method (proven reliable)
                        text_content = element.inner_text().strip()
                        if text_content:
                            print(f"    Element {i+1} text_content: '{text_content}'")
                            # Validate that text content contains required keywords
                            text_lower = text_content.lower()
                            if any(keyword.lower() in text_lower for keyword in required_keywords):
                                count = parse_engagement_count(text_content)
                                if count > 0:
                                    print(f"✅ Found {metric_name} with selector: {selector} ({count})")
                                    return count
                            else:
                                print(f"    ❌ text_content doesn't contain required keywords: {required_keywords}")
                except Exception as e:
                    print(f"    Error with selector {selector}: {str(e)}")
                    continue
            print(f"❌ No {metric_name} found")
            return 0

        # Extract likes (heart icon)
        likes_selectors = [
            '[data-testid="like"]',
            '[data-testid="unlike"]',
            '[aria-label*="Like"]',
            '[aria-label*="like"]',
            'div[role="button"][aria-label*="like"]',
            'div[role="button"][data-testid="like"] span'
        ]
        result['likes'] = extract_engagement_metric('likes', likes_selectors, ['like', 'heart', 'favorite'])

        # Extract comments/replies (speech bubble icon)
        comments_selectors = [
            '[data-testid="reply"]',
            '[aria-label*="Reply"]',
            '[aria-label*="repl"]',
            '[aria-label*="comment"]',
            'div[role="button"][aria-label*="reply"]',
            'div[role="button"][data-testid="reply"] span'
        ]
        result['comments'] = extract_engagement_metric('comments', comments_selectors, ['reply', 'comment', 'repl'])

        # Extract bookmarks (bookmark icon)
        bookmarks_selectors = [
            '[data-testid="bookmark"]',
            '[data-testid="unbookmark"]',
            '[aria-label*="Bookmark"]',
            '[aria-label*="Save"]',
            '[aria-label*="bookmark"]',
            'div[role="button"][aria-label*="bookmark"]',
            'div[role="button"][data-testid="bookmark"] span'
        ]
        result['bookmarks'] = extract_engagement_metric('bookmarks', bookmarks_selectors, ['bookmark', 'save', 'saved'])

        # Extract retweets (retweet/recycle icon)
        retweets_selectors = [
            '[data-testid="retweet"]',
            '[data-testid="unretweet"]',
            '[aria-label*="Retweet"]',
            '[aria-label*="retweet"]',
            '[aria-label*="Share"]',
            'div[role="button"][aria-label*="retweet"]',
            'div[role="button"][data-testid="retweet"] span'
        ]
        result['retweets'] = extract_engagement_metric('retweets', retweets_selectors, ['retweet', 'share', 'repost'])

        # Print extraction results
        print(f"Extraction complete:")
        print(f"  Text: {len(result['text']) if result['text'] else 0} chars")
        print(f"  Author: {result['author'] if result['author'] else 'Not found'}")
        print(f"  DateTime: {result['datetime'] if result['datetime'] else 'Not found'}")
        print(f"  Likes: {result['likes']}")
        print(f"  Comments: {result['comments']}")
        print(f"  Bookmarks: {result['bookmarks']}")
        print(f"  Retweets: {result['retweets']}")

        # Return result if we at least got the text
        return result if result['text'] else None

    except Exception as e:
        print(f"Error extracting tweet data: {str(e)}")
        traceback.print_exc()
        return None

    finally:
        # Clean up resources
        if context:
            context.close()
        if browser:
            browser.close()
        if playwright:
            playwright.stop()


def extract_tweet_text(url: str, timeout: int = 30) -> Optional[str]:
    """
    Extract tweet text from Twitter/X URL using Playwright inner_text() method.

    This function provides backward compatibility by using the enhanced extract_tweet_data()
    function and returning just the text component.

    Args:
        url: Twitter/X post URL (e.g., "https://x.com/user/status/123456789")
        timeout: Maximum wait time in seconds (default: 30)

    Returns:
        Tweet text as string, or None if extraction failed

    Example:
        >>> tweet_url = "https://x.com/MerlijnTrader/status/1979585766515761455"
        >>> text = extract_tweet_text(tweet_url)
        >>> print(text)
        BREAKING:

        CHINA JUST WENT ALL-IN ON BITCOIN.

        XI JINPING ANNOUNCES A PLAN TO LEGALIZE & BUY $40 BILLION IN $BTC FOR THE NATIONAL CRYPTO RESERVE.

        THE EAST JUST SHOOK THE GLOBAL MARKETS.

        BITCOIN'S NEXT CHAPTER HAS BEGUN.
    """

    # Use the enhanced function and return just the text
    tweet_data = extract_tweet_data(url, timeout)
    return tweet_data['text'] if tweet_data else None


# Example usage and testing
if __name__ == "__main__":
    # Test with the previously successful URL
    test_url = "https://x.com/Jasminetrder/status/1980347321708888186"

    print("Testing enhanced tweet data extraction...")
    print("=" * 60)

    # Test the new enhanced function
    print("\n1. Testing extract_tweet_data() - Enhanced version:")
    print("-" * 50)
    result = extract_tweet_data(test_url)

    if result:
        print("✅ Enhanced extraction successful!")
        print(f"Author: {result['author']}")
        print(f"DateTime: {result['datetime']}")
        print(f"Engagement:")
        print(f"  ❤️  Likes: {result['likes']}")
        print(f"  💬 Comments: {result['comments']}")
        print(f"  🔖 Bookmarks: {result['bookmarks']}")
        print(f"  🔄 Retweets: {result['retweets']}")
        print(f"Tweet text ({len(result['text'])} chars):")
        print("-" * 30)
        print(result['text'])
        print("-" * 30)
    else:
        print("❌ Enhanced extraction failed!")

    # Test the backward-compatible function
    print("\n2. Testing extract_tweet_text() - Simple version (backward compatible):")
    print("-" * 50)
    text_result = extract_tweet_text(test_url)

    if text_result:
        print("✅ Simple extraction successful!")
        print(f"Tweet text ({len(text_result)} chars):")
        print("-" * 30)
        print(text_result)
        print("-" * 30)
    else:
        print("❌ Simple extraction failed!")

    print("\n" + "=" * 60)
    print("USAGE EXAMPLES:")
    print("=" * 60)
    print("\n# Enhanced version - returns complete tweet data:")
    print("from extract_tweet import extract_tweet_data")
    print("tweet_data = extract_tweet_data('https://x.com/user/status/123456789')")
    print("if tweet_data:")
    print("    print(f\"Author: {tweet_data['author']}\")")
    print("    print(f\"Time: {tweet_data['datetime']}\")")
    print("    print(f\"❤️  Likes: {tweet_data['likes']}\")")
    print("    print(f\"💬 Comments: {tweet_data['comments']}\")")
    print("    print(f\"🔖 Bookmarks: {tweet_data['bookmarks']}\")")
    print("    print(f\"🔄 Retweets: {tweet_data['retweets']}\")")
    print("    print(f\"Text: {tweet_data['text']}\")")

    print("\n# Simple version - returns just text (backward compatible):")
    print("from extract_tweet import extract_tweet_text")
    print("text = extract_tweet_text('https://x.com/user/status/123456789')")
    print("if text:")
    print("    print(f\"Tweet: {text}\")")
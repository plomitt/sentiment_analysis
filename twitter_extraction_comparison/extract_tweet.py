"""
Twitter/X Tweet Text Extraction

A simple, reusable function to extract tweet text from Twitter/X URLs using Playwright.
Based on Method 1 (inner_text) from the comparison script - proven to be reliable and fast.

Usage:
    from extract_tweet import extract_tweet_text

    tweet_url = "https://x.com/user/status/123456789"
    tweet_text = extract_tweet_text(tweet_url)
    if tweet_text:
        print(f"Tweet text: {tweet_text}")
"""

import time
import traceback
from typing import Optional
from playwright.sync_api import sync_playwright


def extract_tweet_text(url: str, timeout: int = 30) -> Optional[str]:
    """
    Extract tweet text from Twitter/X URL using Playwright inner_text() method.

    This function uses the proven Method 1 (inner_text) approach from the comparison
    script, which was fast (0.01s), reliable, and consistently extracted clean text.

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

        # Try multiple selectors for tweet text
        tweet_selectors = [
            '[data-testid="tweetText"]',
            '[data-testid="tweet"] div[lang]',
            'div[lang] span',
            '.css-1dbjc4n div[lang]',
            'div[lang]',
            'article div[lang]'
        ]

        tweet_element = None
        for selector in tweet_selectors:
            try:
                elements = page.locator(selector)
                if elements.count() > 0:
                    tweet_element = elements.first
                    print(f"Found tweet with selector: {selector}")
                    break
            except:
                continue

        if not tweet_element:
            print("No tweet content found with known selectors")
            return None

        # Extract text using Method 1: inner_text()
        print("Extracting tweet text...")
        start_time = time.time()
        tweet_text = tweet_element.inner_text()
        end_time = time.time()

        # Clean up the result
        tweet_text = tweet_text.strip()

        print(f"Success! Extracted {len(tweet_text)} characters in {end_time - start_time:.2f}s")

        return tweet_text if tweet_text else None

    except Exception as e:
        print(f"Error extracting tweet text: {str(e)}")
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


# Example usage and testing
if __name__ == "__main__":
    # Test with the previously successful URL
    test_url = "https://x.com/MerlijnTrader/status/1979585766515761455"

    print("Testing tweet extraction function...")
    print("=" * 50)

    result = extract_tweet_text(test_url)

    if result:
        print("✅ Extraction successful!")
        print(f"Tweet text ({len(result)} chars):")
        print("-" * 30)
        print(result)
        print("-" * 30)
    else:
        print("❌ Extraction failed!")

    print("\nYou can now use this function in your own code:")
    print("from extract_tweet import extract_tweet_text")
    print("text = extract_tweet_text('https://x.com/user/status/123456789')")
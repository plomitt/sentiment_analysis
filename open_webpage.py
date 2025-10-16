from playwright.sync_api import sync_playwright
from dotenv import load_dotenv
import os

def detect_sponsored_content(page):
    """
    Detect if the page contains sponsored content.

    Args:
        page: Playwright page object

    Returns:
        str or None: Sponsor text if found, None otherwise
    """
    try:
        # Method 1: Try the specific selector first (most reliable)
        sponsor_locator = page.locator("div.pane-toolbar span.button.button-empty.button-title span")
        if sponsor_locator.count() > 0:
            sponsor_text = sponsor_locator.inner_text()
            print(f"✓ Sponsored content detected (primary method): {sponsor_text}")
            return sponsor_text

        # Method 2: Look for sponsor text patterns in the page (more specific to avoid false positives)
        sponsor_patterns = [
            "this is sponsored by",
            "paid promotion by",
            "sponsored content by",
            "advertisement by",
            "paid content by"
        ]

        page_text = page.inner_text("body").lower() if page.inner_text("body") else ""

        for pattern in sponsor_patterns:
            if pattern in page_text:
                print(f"✓ Sponsored content detected (text pattern): {pattern}")
                return f"Sponsored content detected: {pattern}"

        # Method 3: Check for sponsor-related CSS classes or elements
        sponsor_selectors = [
            ".sponsored",
            "[data-sponsored]",
            ".advertisement",
            ".ad-content",
            ".promo",
            "[data-ad]"
        ]

        for selector in sponsor_selectors:
            elements = page.locator(selector)
            if elements.count() > 0:
                print(f"✓ Sponsored content detected (CSS selector): {selector}")
                return f"Sponsored content detected via {selector}"

        # Method 4: Check for specific sponsor-related indicators
        sponsor_indicators = page.evaluate('''
            () => {
                const text = document.body.innerText.toLowerCase();
                const indicators = [
                    "sponsored content",
                    "paid partnership",
                    "promoted content",
                    "advertisement",
                    "sponsored post",
                    "this is a sponsored",
                    "paid promotion"
                ];

                for (const indicator of indicators) {
                    if (text.includes(indicator)) {
                        return indicator;
                    }
                }
                return null;
            }
        ''')

        if sponsor_indicators:
            print(f"✓ Sponsored content detected (page indicators): {sponsor_indicators}")
            return f"Sponsored content: {sponsor_indicators}"

        # Method 5: Check URL patterns that might indicate sponsored content
        current_url = page.url
        if current_url and any(pattern in current_url.lower() for pattern in ["sponsored", "promotion", "advert"]):
            print(f"✓ Sponsored content detected (URL pattern): {current_url}")
            return "Sponsored content detected via URL"

    except Exception as e:
        print(f"Error detecting sponsored content: {e}")

    return None

def detect_embedded_tweet(page):
    """
    Detect if the page contains an embedded tweet and extract the tweet URL.

    Args:
        page: Playwright page object

    Returns:
        str or None: Tweet URL if found, None otherwise
    """
    try:
        # Method 1: Look for specific CryptoPanic embedded tweet pattern (most reliable)
        specific_tweet_links = page.locator('a.css-4rbku5.css-18t94o4.css-1dbjc4n.r-1loqt21[href*="/status/"]')
        if specific_tweet_links.count() > 0:
            raw_url = specific_tweet_links.first.get_attribute('href')
            print(f"✓ Found specific CryptoPanic tweet link: {raw_url[:100]}...")

            # Clean the URL to remove tracking parameters
            clean_url = page.evaluate('''
                (url) => {
                    // Extract base Twitter URL and remove query parameters
                    const urlObj = new URL(url);
                    const baseUrl = `${urlObj.protocol}//${urlObj.host}${urlObj.pathname}`;
                    return baseUrl;
                }
            ''', raw_url)

            if clean_url and ('twitter.com' in clean_url or 'x.com' in clean_url):
                print(f"✓ Cleaned tweet URL: {clean_url}")
                return clean_url

        # Method 2: Look for links with aria-label="Visit this post on X"
        aria_label_links = page.locator('a[aria-label="Visit this post on X"][href*="/status/"]')
        if aria_label_links.count() > 0:
            raw_url = aria_label_links.first.get_attribute('href')
            print(f"✓ Found aria-label tweet link: {raw_url[:100]}...")

            # Clean the URL
            clean_url = page.evaluate('''
                (url) => {
                    const urlObj = new URL(url);
                    const baseUrl = `${urlObj.protocol}//${urlObj.host}${urlObj.pathname}`;
                    return baseUrl;
                }
            ''', raw_url)

            if clean_url and ('twitter.com' in clean_url or 'x.com' in clean_url):
                print(f"✓ Cleaned tweet URL: {clean_url}")
                return clean_url

        # Method 2.5: Try broader CSS class patterns (the class names might vary)
        broader_tweet_links = page.locator('a[href*="/status/"][class*="css-"]')
        if broader_tweet_links.count() > 0:
            raw_url = broader_tweet_links.first.get_attribute('href')
            if raw_url and ('twitter.com' in raw_url or 'x.com' in raw_url):
                print(f"✓ Found broader CSS tweet link: {raw_url[:100]}...")

                clean_url = page.evaluate('''
                    (url) => {
                        const urlObj = new URL(url);
                        const baseUrl = `${urlObj.protocol}//${urlObj.host}${urlObj.pathname}`;
                        return baseUrl;
                    }
                ''', raw_url)

                if clean_url and ('twitter.com' in clean_url or 'x.com' in clean_url):
                    print(f"✓ Cleaned tweet URL: {clean_url}")
                    return clean_url

        # Method 3: Look for Twitter/X iframe elements (legacy method)
        tweet_iframes = page.locator('iframe[src*="twitter.com"], iframe[src*="x.com"]')
        if tweet_iframes.count() > 0:
            # Extract URL from iframe src
            iframe_src = tweet_iframes.first.get_attribute('src')
            print(f"✓ Found Twitter iframe: {iframe_src}")

            # Extract tweet URL from iframe parameters
            if 'twitter.com' in iframe_src or 'x.com' in iframe_src:
                # Try to extract the actual tweet URL
                tweet_url = page.evaluate('''
                    () => {
                        const iframe = document.querySelector('iframe[src*="twitter.com"], iframe[src*="x.com"]');
                        if (!iframe) return null;

                        // Check if the iframe src contains the tweet URL
                        const src = iframe.src;

                        // Look for tweet URL in the src
                        const tweetMatch = src.match(/(https?:\\/\\/(?:twitter\\.com|x\\.com)\\/[^\\/]+\\/status\\/\\d+)/);
                        if (tweetMatch) return tweetMatch[1];

                        // Look for tweet ID
                        const tweetIdMatch = src.match(/tweet_id=(\\d+)/);
                        if (tweetIdMatch) {
                            // Reconstruct tweet URL - we'll need to find the username
                            const links = Array.from(document.querySelectorAll('a[href*="twitter.com"], a[href*="x.com"]'));
                            for (const link of links) {
                                const href = link.href;
                                if (href.includes('/status/')) {
                                    return href;
                                }
                            }
                            return `https://twitter.com/i/web/status/${tweetIdMatch[1]}`;
                        }

                        return null;
                    }
                ''')

                if tweet_url:
                    print(f"✓ Extracted tweet URL: {tweet_url}")
                    return tweet_url

        # Method 4: Look for Twitter widget containers
        tweet_widgets = page.locator('[data-tweet-id], .twitter-tweet, blockquote.twitter-tweet')
        if tweet_widgets.count() > 0:
            tweet_url = page.evaluate('''
                () => {
                    const widgets = document.querySelectorAll('[data-tweet-id], .twitter-tweet, blockquote.twitter-tweet');
                    for (const widget of widgets) {
                        // Try to find the tweet URL from the widget
                        const tweetId = widget.getAttribute('data-tweet-id');
                        if (tweetId) {
                            const links = widget.querySelectorAll('a[href*="twitter.com"], a[href*="x.com"]');
                            for (const link of links) {
                                if (link.href.includes('/status/')) {
                                    return link.href;
                                }
                            }
                            // Fallback: construct URL from tweet ID
                            return `https://twitter.com/i/web/status/${tweetId}`;
                        }

                        // Check for links within the widget
                        const links = widget.querySelectorAll('a[href*="twitter.com"], a[href*="x.com"]');
                        for (const link of links) {
                            if (link.href.includes('/status/')) {
                                return link.href;
                            }
                        }
                    }
                    return null;
                }
            ''')

            if tweet_url:
                print(f"✓ Extracted tweet URL from widget: {tweet_url}")
                return tweet_url

        # Method 5: Look for any Twitter/X links in the description area
        description_links = page.locator('div.description a[href*="twitter.com"], div.description a[href*="x.com"]')
        if description_links.count() > 0:
            tweet_url = description_links.first.get_attribute('href')
            if tweet_url and '/status/' in tweet_url:
                print(f"✓ Found tweet link in description: {tweet_url}")
                return tweet_url

        # Method 6: Search for any Twitter/X status links on the page (fallback)
        all_tweet_links = page.locator('a[href*="twitter.com"][href*="/status/"], a[href*="x.com"][href*="/status/"]')
        if all_tweet_links.count() > 0:
            tweet_url = all_tweet_links.first.get_attribute('href')
            print(f"✓ Found tweet link on page: {tweet_url}")
            return tweet_url

        # Method 7: If we found a Twitter iframe but no direct link, try to extract from page content
        tweet_iframes = page.locator('iframe[src*="twitter.com"], iframe[src*="x.com"]')
        if tweet_iframes.count() > 0:
            print("✓ Found Twitter iframe but no direct link, searching page content...")

            # Debug: Show all Twitter-related links found
            debug_links = page.evaluate('''
                () => {
                    const allLinks = Array.from(document.querySelectorAll('a'));
                    const twitterLinks = [];
                    for (const link of allLinks) {
                        const href = link.href;
                        if (href && (href.includes('twitter.com') || href.includes('x.com'))) {
                            twitterLinks.push({
                                href: href,
                                class: link.className,
                                text: link.textContent?.substring(0, 100) || ''
                            });
                        }
                    }
                    return twitterLinks;
                }
            ''')

            print(f"Debug: Found {len(debug_links)} Twitter-related links:")
            for i, link_info in enumerate(debug_links[:5]):  # Show first 5
                print(f"  {i+1}. Class: {link_info['class'][:50]}...")
                print(f"     Href: {link_info['href'][:100]}...")
                print(f"     Text: {link_info['text']}")
            tweet_url = page.evaluate('''
                () => {
                    // Search all links on the page for Twitter status URLs
                    const allLinks = Array.from(document.querySelectorAll('a'));
                    for (const link of allLinks) {
                        const href = link.href;
                        if (href && (href.includes('twitter.com') || href.includes('x.com')) && href.includes('/status/')) {
                            // Clean the URL
                            const urlObj = new URL(href);
                            const cleanUrl = `${urlObj.protocol}//${urlObj.host}${urlObj.pathname}`;
                            return cleanUrl;
                        }
                    }
                    return null;
                }
            ''')

            if tweet_url:
                print(f"✓ Found tweet URL in page content: {tweet_url}")
                return tweet_url

    except Exception as e:
        print(f"Error detecting embedded tweet: {e}")

    return None

def extract_cryptopanic_info(page, url):
    """
    Extract information from a CryptoPanic post page.

    Args:
        page: Playwright page object
        url: URL of the CryptoPanic post

    Returns:
        dict: Extracted information - for sponsored content, embedded tweets, or news
    """
    # First check if this is sponsored content
    sponsor_text = detect_sponsored_content(page)
    if sponsor_text:
        return {
            'type': 'sponsor',
            'sponsor_text': sponsor_text,
            'url': url
        }

    # Check if this contains an embedded tweet
    tweet_url = detect_embedded_tweet(page)
    if tweet_url:
        # Extract basic info for embedded tweet case
        info = {
            'type': 'embedded_tweet',
            'title': None,
            'time': None,
            'cryptopanic_url': url,
            'tweet_url': tweet_url
        }

        # Try to extract title and time even for embedded tweets
        try:
            # Extract title: h1.post-title a span.text (use first element if multiple found)
            # Wait for the element to be available
            page.wait_for_selector("h1.post-title a span.text", timeout=15000)
            title_locator = page.locator("h1.post-title a span.text").first
            info['title'] = title_locator.inner_text()
            print("✓ Title extracted successfully for embedded tweet")

        except Exception as e:
            print(f"⚠ Could not extract title for embedded tweet: {e}")
            # Try alternative selectors for title
            fallback_selectors = [
                "h1.post-title span.text",
                "h1.post-title a",
                "h1.post-title",
                ".post-title span.text",
                ".post-title"
            ]

            for selector in fallback_selectors:
                try:
                    fallback_locator = page.locator(selector)
                    if fallback_locator.count() > 0:
                        info['title'] = fallback_locator.inner_text()
                        print(f"✓ Title extracted using fallback selector: {selector}")
                        break
                except:
                    continue

        # Try to extract time
        try:
            # Extract time: span.post-source time[datetime]
            # Wait for the element to be available
            page.wait_for_selector("span.post-source time[datetime]", timeout=8000)
            time_locator = page.locator("span.post-source time[datetime]")
            if time_locator.count() > 0:
                info['time'] = time_locator.get_attribute('datetime')
                print("✓ Time extracted successfully for embedded tweet")

        except Exception as e:
            print(f"⚠ Could not extract time for embedded tweet: {e}")

        return info

    # If not sponsored or embedded tweet, proceed with regular news extraction
    info = {
        'type': 'news',
        'title': None,
        'body': None,
        'time': None,
        'url': url
    }

    try:
        # Extract title: h1.post-title a span.text (use first element if multiple found)
        # Wait for the element to be available
        page.wait_for_selector("h1.post-title a span.text", timeout=15000)
        title_locator = page.locator("h1.post-title a span.text").first
        info['title'] = title_locator.inner_text()
        print("✓ Title extracted successfully")

    except Exception as e:
        print(f"⚠ Primary title selector failed, trying alternatives... Error: {e}")

        # Try alternative selectors for title
        fallback_selectors = [
            "h1.post-title span.text",
            "h1.post-title a",
            "h1.post-title",
            ".post-title span.text",
            ".post-title"
        ]

        for selector in fallback_selectors:
            try:
                fallback_locator = page.locator(selector)
                if fallback_locator.count() > 0:
                    info['title'] = fallback_locator.inner_text()
                    print(f"✓ Title extracted using fallback selector: {selector}")
                    break
            except:
                continue
        else:
            print("⚠ Title element not found with any selector")

    # Try one more time with a simple h1 selector as last resort
    if not info['title']:
        try:
            simple_title = page.locator("h1").first.inner_text()
            if simple_title:
                info['title'] = simple_title
                print("✓ Title extracted using simple h1 selector")
        except:
            pass

    try:
        # Extract body: div.description div.description-body
        # Wait for the element to be available
        page.wait_for_selector("div.description div.description-body", timeout=10000)
        body_locator = page.locator("div.description div.description-body")
        if body_locator.count() > 0:
            info['body'] = body_locator.inner_text()
            print("✓ Body extracted successfully")
        else:
            print("⚠ Primary body selector not found, trying alternatives...")

            # Try alternative selectors for body
            fallback_body_selectors = [
                "div.description-body",
                ".description-body",
                "div.description",
                ".description",
                ".post-content",
                ".content"
            ]

            for selector in fallback_body_selectors:
                try:
                    fallback_locator = page.locator(selector)
                    if fallback_locator.count() > 0:
                        info['body'] = fallback_locator.inner_text()
                        print(f"✓ Body extracted using fallback selector: {selector}")
                        break
                except:
                    continue
            else:
                print("⚠ Body element not found with any selector")

    except Exception as e:
        print(f"✗ Error extracting body: {e}")
        # Try to get any substantial text content as fallback
        try:
            main_content = page.locator("main, article, .main-content").first.inner_text()
            if main_content and len(main_content.strip()) > 50:
                info['body'] = main_content
                print("✓ Body extracted using main content selector")
        except:
            pass

    try:
        # Extract time: span.post-source time[datetime]
        # Wait for the element to be available
        page.wait_for_selector("span.post-source time[datetime]", timeout=8000)
        time_locator = page.locator("span.post-source time[datetime]")
        if time_locator.count() > 0:
            info['time'] = time_locator.get_attribute('datetime')
            print("✓ Time extracted successfully")
        else:
            print("⚠ Primary time selector not found, trying alternatives...")
            # Try alternative time selectors
            alternative_selectors = [
                "time[datetime]",
                ".post-source time",
                "[datetime]",
                "time",
                ".post-date",
                ".timestamp",
                ".date",
                "time[datetime]",
                ".publish-date"
            ]

            for selector in alternative_selectors:
                try:
                    alt_locator = page.locator(selector)
                    if alt_locator.count() > 0:
                        if selector == "time[datetime]" or selector == "[datetime]":
                            info['time'] = alt_locator.get_attribute('datetime')
                        else:
                            info['time'] = alt_locator.inner_text()
                        print(f"✓ Time found using alternative selector: {selector}")
                        break
                except:
                    continue
            else:
                print("⚠ Time element not found with any selector")

    except Exception as e:
        print(f"✗ Error extracting time: {e}")
        # Try to find any date/time related elements
        try:
            date_elements = page.locator("[datetime], .date, .time, time").all()
            for element in date_elements:
                try:
                    datetime_attr = element.get_attribute('datetime')
                    if datetime_attr:
                        info['time'] = datetime_attr
                        print("✓ Time extracted from datetime attribute")
                        break
                    else:
                        text = element.inner_text()
                        if text and any(char.isdigit() for char in text):
                            info['time'] = text
                            print("✓ Time extracted from date element text")
                            break
                except:
                    continue
        except:
            pass

    return info

def parse_cryptopanic_url(user_url=None):
    """
    Open the CryptoPanic webpage and extract post information.

    Args:
        url: URL of the CryptoPanic post (optional, uses default if not provided)
    """
    load_dotenv()

    url = user_url or os.getenv("TEST_URL")

    if url is None:
        raise ValueError("No URL provided. Please set TEST_URL in .env or pass as argument.")

    with sync_playwright() as p:
        # Launch Chromium browser in headless mode for faster loading
        browser = p.chromium.launch(headless=False)

        try:
            # Create a new page
            page = browser.new_page()

            # Navigate to the URL with a 60-second timeout
            print(f"Navigating to: {url}")
            page.goto(url, timeout=60000)

            # Wait for page to load - use domcontentloaded instead of networkidle
            try:
                page.wait_for_load_state('load', timeout=10000)
                print("Page loaded successfully")
            except:
                print("Page may still be loading, proceeding anyway...")

            # Extract information
            print("\n=== Extracting Information ===")
            extracted_info = extract_cryptopanic_info(page, url)

            # Display results based on content type
            print("\n=== Extracted Information ===")
            if extracted_info['type'] == 'sponsor':
                print(f"📢 Content Type: SPONSORED")
                print(f"🔗 URL: {extracted_info['url']}")
                print(f"💰 Sponsor Message: {extracted_info['sponsor_text']}")
                print("=" * 50)

                # Hold for 20 seconds for sponsor content
                print("Holding for 20 seconds...")
                page.wait_for_timeout(20000)

            elif extracted_info['type'] == 'embedded_tweet':
                print(f"🐦 Content Type: EMBEDDED TWEET")
                print(f"📰 Title: {extracted_info['title']}")
                print(f"🔗 CryptoPanic URL: {extracted_info['cryptopanic_url']}")
                print(f"🐦 Tweet URL: {extracted_info['tweet_url']}")
                print(f"⏰ Time: {extracted_info['time']}")
                print("=" * 50)

                # Navigate to the tweet URL
                print(f"\n🔄 Navigating to tweet: {extracted_info['tweet_url']}")
                try:
                    page.goto(extracted_info['tweet_url'], timeout=30000)
                    page.wait_for_load_state('domcontentloaded', timeout=10000)
                    print("✓ Tweet page loaded successfully")

                    # Hold for 5 seconds on tweet page
                    print("🐦 Holding on tweet page for 5 seconds...")
                    page.wait_for_timeout(5000)

                except Exception as e:
                    print(f"⚠ Error navigating to tweet: {e}")
                    print("Holding on CryptoPanic page for 5 seconds instead...")
                    page.wait_for_timeout(5000)

            else:
                print(f"📰 Content Type: NEWS")
                print(f"🔗 URL: {extracted_info['url']}")
                print(f"📰 Title: {extracted_info['title']}")

                # Show body preview
                body = extracted_info['body']
                if body:
                    if len(body) > 200:
                        print(f"📄 Body ({len(body)} chars): {body[:200]}...")
                    else:
                        print(f"📄 Body: {body}")
                else:
                    print("📄 Body: Not found")

                print(f"⏰ Time: {extracted_info['time']}")
                print("=" * 50)

                # Hold for 20 seconds for regular news
                print("Holding for 20 seconds...")
                page.wait_for_timeout(20000)

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            # Always close the browser
            browser.close()

if __name__ == "__main__":
    import sys

    # Allow URL to be passed as command line argument
    url = sys.argv[1] if len(sys.argv) > 1 else None
    parse_cryptopanic_url(url)
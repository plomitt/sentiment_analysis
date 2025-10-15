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
        # Try the specific selector first
        sponsor_locator = page.locator("div.pane-toolbar span.button.button-empty.button-title span")
        if sponsor_locator.count() > 0:
            sponsor_text = sponsor_locator.inner_text()
            print(f"✓ Sponsored content detected: {sponsor_text}")
            return sponsor_text

        # Additional fallback: Check if this might be sponsored content by looking at URL patterns
        # or other indicators, but for now we'll proceed with news extraction

    except Exception as e:
        print(f"Error detecting sponsored content: {e}")

    return None

def extract_cryptopanic_info(page, url):
    """
    Extract information from a CryptoPanic post page.

    Args:
        page: Playwright page object
        url: URL of the CryptoPanic post

    Returns:
        dict: Extracted information - either for sponsored content or news
    """
    # First check if this is sponsored content
    sponsor_text = detect_sponsored_content(page)
    if sponsor_text:
        return {
            'type': 'sponsor',
            'sponsor_text': sponsor_text,
            'url': url
        }

    # If not sponsored, proceed with news extraction
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
        browser = p.chromium.launch(headless=True)

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
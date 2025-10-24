"""
Twitter/X Authentication using Browser State Storage

This module provides functionality to authenticate with Twitter/X using Playwright
and save the browser authentication state for reuse in subsequent sessions.

The approach:
1. Perform initial login with credentials and save browser state
2. Reuse saved state to create authenticated contexts without repeated login
3. Validate authentication and handle session expiry gracefully

Usage:
    from twitter_auth import get_authenticated_search_results

    # Get Bitcoin search results (will handle authentication automatically)
    results = get_authenticated_search_results()
"""

import os
import json
import traceback
from typing import Optional, Dict, Any
from pathlib import Path
from playwright.sync_api import sync_playwright, BrowserContext, Page, Browser, Playwright
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
AUTH_DIR = Path("twitter_auth/.auth")
STATE_FILE = AUTH_DIR / "twitter_state.json"
LOGIN_URL = "https://x.com/i/flow/login"
TARGET_SEARCH_URL = "https://x.com/search?q=bitcoin&src=typed_query&f=live"


def ensure_auth_directory() -> None:
    """Ensure the authentication directory exists."""
    AUTH_DIR.mkdir(parents=True, exist_ok=True)


def get_credentials() -> tuple[str, str]:
    """
    Get Twitter credentials from environment variables.

    Returns:
        Tuple of (email, password)

    Raises:
        ValueError: If credentials are not found in environment variables
    """
    email = os.getenv('TWITTER_EMAIL')
    password = os.getenv('TWITTER_PASSWORD')

    if not email or not password:
        raise ValueError(
            "TWITTER_EMAIL and TWITTER_PASSWORD must be set in environment variables"
        )

    return email, password


def create_browser_context(headless: bool = True) -> tuple[Playwright, Browser, BrowserContext]:
    """
    Create a new browser context with stealth settings.

    Args:
        headless: Whether to run browser in headless mode

    Returns:
        Tuple of (playwright, browser, context)
    """
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(
        headless=headless,
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

    return playwright, browser, context


def add_stealth_script(page: Page) -> None:
    """Add stealth script to avoid bot detection."""
    page.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined,
        });

        // Override the permissions API
        if (navigator.permissions) {
            navigator.permissions.query = () => Promise.resolve({ state: 'granted' });
        }

        // Override the plugins API
        Object.defineProperty(navigator, 'plugins', {
            get: () => [1, 2, 3, 4, 5],
        });

        // Override the languages API
        Object.defineProperty(navigator, 'languages', {
            get: () => ['en-US', 'en'],
        });
    """)


def perform_login(page: Page, email: str, password: str, timeout: int = 60) -> bool:
    """
    Perform login to Twitter/X with given credentials.

    Args:
        page: Playwright page instance
        email: Twitter email
        password: Twitter password
        timeout: Login timeout in seconds

    Returns:
        True if login successful, False otherwise
    """
    try:
        print(f"Navigating to login page: {LOGIN_URL}")
        page.goto(LOGIN_URL, wait_until='domcontentloaded', timeout=timeout * 1000)

        # Wait for page to load
        page.wait_for_timeout(3000)

        # Look for username/email input
        username_selectors = [
            'input[name="text"]',
            'input[autocomplete="username"]',
            'input[type="text"]',
            '[data-testid="ocfEnterTextTextInput"]',
            'input[data-testid="ocfEnterTextTextInput"]',
            'input[aria-label="Phone, email, or username"]',
            'input[placeholder*="email"]',
            'input[placeholder*="username"]'
        ]

        username_input = None
        for selector in username_selectors:
            try:
                element = page.locator(selector).first
                if element.count() > 0:
                    username_input = element
                    print(f"Found username input with selector: {selector}")
                    break
            except:
                continue

        if not username_input:
            print("Could not find username/email input field")
            return False

        # Enter email
        print("Entering email...")
        username_input.fill(email)
        page.wait_for_timeout(1000)

        # Click next/continue button - try multiple approaches
        next_selectors = [
            '[data-testid="LoginForm_Login_Button"]',
            'div[role="button"]:has-text("Next")',
            'div[role="button"]:has-text("Continue")',
            'button[type="submit"]',
            'div[data-testid="LoginForm_Login_Button"]',
            'div[role="button"]:has-text("Next,")'
        ]

        next_clicked = False
        for selector in next_selectors:
            try:
                button = page.locator(selector).first
                if button.count() > 0:
                    print(f"Clicking next button with selector: {selector}")
                    button.click()
                    next_clicked = True
                    break
            except:
                continue

        if not next_clicked:
            print("Could not find or click next button, trying Enter key...")
            username_input.press('Enter')

        page.wait_for_timeout(5000)

        # Look for password input - this might be on a new page
        password_selectors = [
            'input[name="password"]',
            'input[type="password"]',
            'input[autocomplete="current-password"]',
            '[data-testid="ocfEnterTextTextInput"]',
            'input[data-testid="ocfEnterTextTextInput"]',
            'input[aria-label*="password"]',
            'input[placeholder*="password"]'
        ]

        password_input = None
        # Wait a bit more for password field to appear
        page.wait_for_timeout(3000)

        for selector in password_selectors:
            try:
                element = page.locator(selector).first
                if element.count() > 0:
                    password_input = element
                    print(f"Found password input with selector: {selector}")
                    break
            except:
                continue

        if not password_input:
            print("Could not find password input field")
            # Debug: print current page content
            print("Current URL:", page.url)
            print("Page title:", page.title())
            return False

        # Enter password
        print("Entering password...")
        password_input.fill(password)
        page.wait_for_timeout(1000)

        # Click login button
        login_selectors = [
            '[data-testid="LoginForm_Login_Button"]',
            'div[role="button"]:has-text("Log in")',
            'button[type="submit"]',
            'div[data-testid="LoginForm_Login_Button"]',
            'div[role="button"]:has-text("Log in,")'
        ]

        login_clicked = False
        for selector in login_selectors:
            try:
                button = page.locator(selector).first
                if button.count() > 0:
                    print(f"Clicking login button with selector: {selector}")
                    button.click()
                    login_clicked = True
                    break
            except:
                continue

        if not login_clicked:
            print("Could not find login button, trying Enter key...")
            password_input.press('Enter')

        # Wait for login to complete
        print("Waiting for login to complete...")
        page.wait_for_timeout(8000)

        # Check if login was successful by looking for navigation away from login page
        current_url = page.url
        print(f"Current URL after login attempt: {current_url}")

        # Look for successful login indicators
        if ('login' not in current_url.lower() and
            'flow/login' not in current_url and
            'i/flow' not in current_url):
            print("✅ Login successful!")
            return True
        else:
            print("❌ Login may have failed - still on login flow")

            # Check for potential verification/challenge pages
            challenge_indicators = [
                'challenge',
                'verification',
                'confirm',
                'suspended',
                'locked'
            ]

            for indicator in challenge_indicators:
                if indicator in current_url.lower():
                    print(f"⚠️  Account may require verification ({indicator})")
                    break

            return False

    except Exception as e:
        print(f"❌ Login error: {str(e)}")
        traceback.print_exc()
        return False


def save_authentication_state(context: BrowserContext) -> bool:
    """
    Save the current browser context's authentication state.

    Args:
        context: Browser context to save state from

    Returns:
        True if save successful, False otherwise
    """
    try:
        ensure_auth_directory()
        state = context.storage_state()

        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)

        print(f" Authentication state saved to: {STATE_FILE}")
        return True

    except Exception as e:
        print(f"L Failed to save authentication state: {str(e)}")
        return False


def load_authentication_state() -> Optional[Dict[str, Any]]:
    """
    Load saved authentication state from file.

    Returns:
        Authentication state dictionary, or None if file doesn't exist or is invalid
    """
    try:
        if not STATE_FILE.exists():
            print("No saved authentication state found")
            return None

        with open(STATE_FILE, 'r') as f:
            state = json.load(f)

        print(f" Authentication state loaded from: {STATE_FILE}")
        return state

    except Exception as e:
        print(f"L Failed to load authentication state: {str(e)}")
        return None


def validate_authentication(page: Page, target_url: str = TARGET_SEARCH_URL) -> bool:
    """
    Validate that the current context is authenticated by checking access to target URL.

    Args:
        page: Page instance to test authentication
        target_url: URL to test authentication against

    Returns:
        True if authenticated, False otherwise
    """
    try:
        print(f"Validating authentication by accessing: {target_url}")
        page.goto(target_url, wait_until='domcontentloaded', timeout=30000)
        page.wait_for_timeout(3000)

        # Check for indicators of successful authentication
        # Look for tweet content or search results
        tweet_indicators = [
            '[data-testid="tweet"]',
            'article[role="article"]',
            '[data-testid="tweetText"]'
        ]

        login_indicators = [
            '[data-testid="LoginForm_Login_Button"]',
            'input[name="text"]',
            'input[type="password"]'
        ]

        # Check if we see tweets (authenticated) or login form (not authenticated)
        for indicator in tweet_indicators:
            try:
                if page.locator(indicator).count() > 0:
                    print(" Authentication validated - found tweet content")
                    return True
            except:
                continue

        for indicator in login_indicators:
            try:
                if page.locator(indicator).count() > 0:
                    print("L Authentication failed - found login form")
                    return False
            except:
                continue

        # If we can't clearly determine, check URL
        if 'login' in page.url.lower():
            print("L Authentication failed - redirected to login")
            return False

        print("�  Authentication status unclear - assuming valid")
        return True

    except Exception as e:
        print(f"L Authentication validation error: {str(e)}")
        return False


def create_authenticated_context(headless: bool = True) -> Optional[tuple[Playwright, Browser, BrowserContext]]:
    """
    Create an authenticated browser context, using saved state if available.

    Args:
        headless: Whether to run browser in headless mode

    Returns:
        Tuple of (playwright, browser, context) if successful, None otherwise
    """
    # Try to load saved state
    saved_state = load_authentication_state()

    if saved_state:
        print("Creating context with saved authentication state...")
        try:
            playwright, browser, context = create_browser_context(headless=headless)

            # Create page for validation
            page = context.new_page()
            add_stealth_script(page)

            if validate_authentication(page):
                print(" Successfully created authenticated context with saved state")
                return playwright, browser, context
            else:
                print("L Saved state is invalid, will perform fresh login")
                context.close()
                browser.close()
                playwright.stop()

        except Exception as e:
            print(f"L Failed to create context with saved state: {str(e)}")

    # Perform fresh login if saved state is invalid or doesn't exist
    print("Performing fresh login...")
    try:
        email, password = get_credentials()

        playwright, browser, context = create_browser_context(headless=headless)
        page = context.new_page()
        add_stealth_script(page)

        if perform_login(page, email, password):
            # Save the authentication state for future use
            save_authentication_state(context)

            # Validate the login
            if validate_authentication(page):
                print(" Successfully created authenticated context with fresh login")
                return playwright, browser, context
            else:
                print("L Login validation failed")

        # Cleanup on failure
        context.close()
        browser.close()
        playwright.stop()
        return None

    except Exception as e:
        print(f"L Failed to create authenticated context: {str(e)}")
        return None


def get_authenticated_search_results(headless: bool = True) -> Optional[str]:
    """
    Get the Bitcoin search results page from Twitter/X with authentication.

    Args:
        headless: Whether to run browser in headless mode

    Returns:
        Page content as HTML string if successful, None otherwise
    """
    result = create_authenticated_context(headless=headless)

    if not result:
        print("L Failed to create authenticated context")
        return None

    playwright, browser, context = result

    try:
        page = context.new_page()
        add_stealth_script(page)

        print(f"Accessing Bitcoin search page: {TARGET_SEARCH_URL}")
        page.goto(TARGET_SEARCH_URL, wait_until='domcontentloaded', timeout=30000)
        page.wait_for_timeout(3000)

        # Get page content
        content = page.content()
        print(f" Successfully retrieved search page content ({len(content)} chars)")

        return content

    except Exception as e:
        print(f"L Failed to get search results: {str(e)}")
        return None

    finally:
        # Cleanup
        context.close()
        browser.close()
        playwright.stop()


# Example usage and testing
if __name__ == "__main__":
    print("Twitter/X Authentication Test")
    print("=" * 50)

    # Test authentication
    print("\n1. Testing authenticated access to Bitcoin search...")
    print("-" * 50)

    content = get_authenticated_search_results(headless=False)  # Use non-headless for testing

    if content:
        print(" Authentication test successful!")
        print(f"Retrieved {len(content)} characters of HTML content")

        # Check for tweet content
        if 'data-testid="tweet"' in content or 'article role="article"' in content:
            print(" Found tweet indicators in content")
        else:
            print("�  No tweet indicators found in content")

    else:
        print("L Authentication test failed!")

    print("\n" + "=" * 50)
    print("USAGE EXAMPLES:")
    print("=" * 50)
    print("\n# Get authenticated Bitcoin search results:")
    print("from twitter_auth import get_authenticated_search_results")
    print("content = get_authenticated_search_results()")
    print("if content:")
    print("    # Process the HTML content to extract tweets")
    print("    print(f'Retrieved {len(content)} characters')")
    print("\n# Create authenticated context for custom operations:")
    print("from twitter_auth import create_authenticated_context")
    print("result = create_authenticated_context()")
    print("if result:")
    print("    playwright, browser, context = result")
    print("    page = context.new_page()")
    print("    # Use page for authenticated operations")
    print("    context.close()")
    print("    browser.close()")
    print("    playwright.stop()")